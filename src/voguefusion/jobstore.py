"""Durable job queue and job state, in one SQLite table (WAL mode).

State machine:

    queued --claim--> running --complete--> succeeded
                        |  \\--fail(retryable), attempts left--> queued (after backoff)
                        |   \\-fail(retryable), no attempts left--> dead
                        |    \\fail(permanent)--> failed
                        \\--lease expires (worker died)--> queued, or dead if no attempts left

Every mutation runs in a BEGIN IMMEDIATE transaction, so claim is atomic
across processes. A running job is owned by (lease_owner, attempts); every
write by a worker is conditional on both, so a worker whose lease expired
and whose job was handed to someone else cannot overwrite the new owner's
work (fencing).

All SQL for the queue lives in this module. Moving to Postgres would mean
rewriting this file with SELECT ... FOR UPDATE SKIP LOCKED in claim().
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTIVE = ("queued", "running")
TERMINAL = ("succeeded", "failed", "dead")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    state            TEXT NOT NULL,
    substate         TEXT,
    input_json       TEXT NOT NULL,
    input_key        TEXT NOT NULL,
    idempotency_key  TEXT UNIQUE,
    result_json      TEXT,
    error            TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL,
    lease_owner      TEXT,
    lease_expires_at REAL,
    available_at     REAL NOT NULL,
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_claim ON jobs (kind, state, available_at, created_at);
CREATE INDEX IF NOT EXISTS ix_jobs_input ON jobs (kind, input_key, state);
CREATE INDEX IF NOT EXISTS ix_jobs_finished ON jobs (kind, state, finished_at);
"""


@dataclass(frozen=True)
class KindLimits:
    max_depth: int
    max_attempts: int
    lease_s: float
    retry_backoff_s: float


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    state: str
    substate: str | None
    input: dict[str, Any]
    input_key: str
    idempotency_key: str | None
    result: dict[str, Any] | None
    error: str | None
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    available_at: float
    created_at: float
    started_at: float | None
    finished_at: float | None
    updated_at: float

    def public(self) -> dict[str, Any]:
        """The view returned by the API: no lease internals."""
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "substate": self.substate,
            "input": self.input,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class QueueFull(Exception):
    def __init__(self, kind: str, depth: int, limit: int) -> None:
        super().__init__(f"{kind} queue full: depth {depth} >= {limit}")
        self.kind, self.depth, self.limit = kind, depth, limit


class IdempotencyConflict(Exception):
    """The idempotency key was already used with a different request body."""


def _row_to_job(r: sqlite3.Row) -> Job:
    return Job(
        id=r["id"],
        kind=r["kind"],
        state=r["state"],
        substate=r["substate"],
        input=json.loads(r["input_json"]),
        input_key=r["input_key"],
        idempotency_key=r["idempotency_key"],
        result=json.loads(r["result_json"]) if r["result_json"] else None,
        error=r["error"],
        attempts=r["attempts"],
        max_attempts=r["max_attempts"],
        lease_owner=r["lease_owner"],
        lease_expires_at=r["lease_expires_at"],
        available_at=r["available_at"],
        created_at=r["created_at"],
        started_at=r["started_at"],
        finished_at=r["finished_at"],
        updated_at=r["updated_at"],
    )


def percentile(values: list[float], q: float) -> float | None:
    """Linearly interpolated percentile, q in [0, 100]; None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class JobStore:
    def __init__(
        self,
        path: Path,
        kinds: dict[str, KindLimits],
        busy_timeout_ms: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.kinds = kinds
        self.busy_timeout_ms = busy_timeout_ms
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)}")
        c.execute("PRAGMA synchronous=FULL")
        return c

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    # -- submit -------------------------------------------------------------

    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        input_key: str,
        idempotency_key: str | None = None,
        dedupe: bool = True,
    ) -> tuple[Job, bool]:
        """Enqueue a job. Returns (job, created).

        An existing job is returned instead of creating one when the
        idempotency key was seen before, or when dedupe is on and a queued,
        running or succeeded job has the same input_key. Raises QueueFull
        when queued + running jobs of this kind are at the limit.
        """
        limits = self.kinds[kind]
        now = self.clock()
        with self._tx() as c:
            if idempotency_key is not None:
                r = c.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if r is not None:
                    if r["input_key"] != input_key or r["kind"] != kind:
                        raise IdempotencyConflict(idempotency_key)
                    return _row_to_job(r), False
            if dedupe:
                r = c.execute(
                    "SELECT * FROM jobs WHERE kind = ? AND input_key = ? "
                    "AND state IN ('queued', 'running', 'succeeded') "
                    "ORDER BY created_at DESC LIMIT 1",
                    (kind, input_key),
                ).fetchone()
                if r is not None:
                    return _row_to_job(r), False
            depth = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = ? AND state IN ('queued', 'running')",
                (kind,),
            ).fetchone()[0]
            if depth >= limits.max_depth:
                raise QueueFull(kind, depth, limits.max_depth)
            job_id = uuid.uuid4().hex
            c.execute(
                "INSERT INTO jobs (id, kind, state, input_json, input_key, idempotency_key, "
                "max_attempts, available_at, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    kind,
                    json.dumps(payload, sort_keys=True),
                    input_key,
                    idempotency_key,
                    limits.max_attempts,
                    now,
                    now,
                    now,
                ),
            )
            r = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return _row_to_job(r), True

    # -- worker side ----------------------------------------------------------

    def _reap_expired(self, c: sqlite3.Connection, kind: str, now: float) -> None:
        """Return jobs whose lease expired to the queue, or mark them dead."""
        limits = self.kinds[kind]
        c.execute(
            "UPDATE jobs SET state = 'dead', substate = NULL, lease_owner = NULL, "
            "lease_expires_at = NULL, finished_at = ?, updated_at = ?, "
            "error = 'lease expired (worker lost); no attempts left' "
            "WHERE kind = ? AND state = 'running' AND lease_expires_at < ? "
            "AND attempts >= max_attempts",
            (now, now, kind, now),
        )
        c.execute(
            "UPDATE jobs SET state = 'queued', substate = NULL, lease_owner = NULL, "
            "lease_expires_at = NULL, available_at = ?, updated_at = ?, "
            "error = 'lease expired (worker lost); requeued' "
            "WHERE kind = ? AND state = 'running' AND lease_expires_at < ?",
            (now + limits.retry_backoff_s, now, kind, now),
        )

    def claim(self, kind: str, worker_id: str) -> Job | None:
        """Atomically take the oldest available queued job of kind, or None."""
        limits = self.kinds[kind]
        now = self.clock()
        with self._tx() as c:
            self._reap_expired(c, kind, now)
            r = c.execute(
                "SELECT id FROM jobs WHERE kind = ? AND state = 'queued' AND available_at <= ? "
                "ORDER BY created_at LIMIT 1",
                (kind, now),
            ).fetchone()
            if r is None:
                return None
            c.execute(
                "UPDATE jobs SET state = 'running', substate = 'claimed', "
                "attempts = attempts + 1, lease_owner = ?, lease_expires_at = ?, "
                "started_at = ?, updated_at = ? WHERE id = ?",
                (worker_id, now + limits.lease_s, now, now, r["id"]),
            )
            return _row_to_job(c.execute("SELECT * FROM jobs WHERE id = ?", (r["id"],)).fetchone())

    def _owned(self) -> str:
        return "id = ? AND state = 'running' AND lease_owner = ? AND attempts = ?"

    def renew(self, job: Job, substate: str | None = None) -> bool:
        """Extend the lease and optionally set the substate. False if the lease was lost."""
        now = self.clock()
        lease = self.kinds[job.kind].lease_s
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET lease_expires_at = ?, substate = COALESCE(?, substate), "
                f"updated_at = ? WHERE {self._owned()} AND lease_expires_at >= ?",
                (now + lease, substate, now, job.id, job.lease_owner, job.attempts, now),
            )
            return cur.rowcount == 1

    def complete(self, job: Job, result: dict[str, Any]) -> bool:
        now = self.clock()
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET state = 'succeeded', substate = NULL, result_json = ?, "
                "error = NULL, lease_owner = NULL, lease_expires_at = NULL, finished_at = ?, "
                f"updated_at = ? WHERE {self._owned()}",
                (
                    json.dumps(result, sort_keys=True),
                    now,
                    now,
                    job.id,
                    job.lease_owner,
                    job.attempts,
                ),
            )
            return cur.rowcount == 1

    def fail(self, job: Job, error: str, retryable: bool) -> str | None:
        """Record a failure. Returns the new state, or None if the lease was lost."""
        now = self.clock()
        limits = self.kinds[job.kind]
        if not retryable:
            new_state = "failed"
        elif job.attempts >= job.max_attempts:
            new_state = "dead"
        else:
            new_state = "queued"
        finished = now if new_state != "queued" else None
        with self._tx() as c:
            cur = c.execute(
                "UPDATE jobs SET state = ?, substate = NULL, error = ?, lease_owner = NULL, "
                "lease_expires_at = NULL, available_at = ?, finished_at = ?, updated_at = ? "
                f"WHERE {self._owned()}",
                (
                    new_state,
                    error[:2000],
                    now + limits.retry_backoff_s * job.attempts,
                    finished,
                    now,
                    job.id,
                    job.lease_owner,
                    job.attempts,
                ),
            )
            return new_state if cur.rowcount == 1 else None

    # -- reads ----------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        c = self._conn()
        try:
            r = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return _row_to_job(r) if r else None
        finally:
            c.close()

    def depth(self, kind: str) -> int:
        c = self._conn()
        try:
            return c.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = ? AND state IN ('queued', 'running')",
                (kind,),
            ).fetchone()[0]
        finally:
            c.close()

    def stats(self, kind: str, window: int) -> dict[str, Any]:
        """Counts by state, depth, and latency percentiles over the last window successes.

        queue_wait is started_at - created_at of the final attempt, so for a
        retried job it includes the earlier attempts.
        """
        c = self._conn()
        try:
            counts = {s: 0 for s in (*ACTIVE, *TERMINAL)}
            for r in c.execute(
                "SELECT state, COUNT(*) AS n FROM jobs WHERE kind = ? GROUP BY state", (kind,)
            ):
                counts[r["state"]] = r["n"]
            rows = c.execute(
                "SELECT created_at, started_at, finished_at FROM jobs "
                "WHERE kind = ? AND state = 'succeeded' ORDER BY finished_at DESC LIMIT ?",
                (kind, window),
            ).fetchall()
            oldest = c.execute(
                "SELECT MIN(created_at) FROM jobs WHERE kind = ? AND state = 'queued'", (kind,)
            ).fetchone()[0]
        finally:
            c.close()
        wait = [r["started_at"] - r["created_at"] for r in rows]
        service = [r["finished_at"] - r["started_at"] for r in rows]
        e2e = [r["finished_at"] - r["created_at"] for r in rows]

        def pct(v: list[float]) -> dict[str, float | None]:
            return {"p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v, default=None)}

        return {
            "kind": kind,
            "counts": counts,
            "depth": counts["queued"] + counts["running"],
            "max_depth": self.kinds[kind].max_depth,
            "oldest_queued_age_s": None if oldest is None else self.clock() - oldest,
            "window": len(rows),
            "queue_wait_s": pct(wait),
            "service_s": pct(service),
            "end_to_end_s": pct(e2e),
        }
