"""Blob store and job store properties: atomic writes, leases, fencing, retries, backpressure."""

import os
import sqlite3
import threading

import pytest

from voguefusion.blobstore import BlobCorrupt, BlobNotFound, BlobStore, sha256_hex
from voguefusion.jobstore import IdempotencyConflict, JobStore, KindLimits, QueueFull, percentile

LIMITS = {"k": KindLimits(max_depth=3, max_attempts=2, lease_s=10.0, retry_backoff_s=1.0)}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return JobStore(tmp_path / "jobs.sqlite3", LIMITS, busy_timeout_ms=5000, clock=clock)


# -- blob store ---------------------------------------------------------------


def test_blob_address_is_content_hash(tmp_path):
    blobs = BlobStore(tmp_path)
    sha = blobs.put(b"hello")
    assert sha == sha256_hex(b"hello")
    assert blobs.get(sha, verify=True) == b"hello"
    assert blobs.put(b"hello") == sha  # idempotent


def test_interrupted_blob_write_leaves_nothing_at_final_address(tmp_path, monkeypatch):
    blobs = BlobStore(tmp_path)

    def crash(*_a, **_k):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        blobs.put(b"payload")
    sha = sha256_hex(b"payload")
    assert not blobs.exists(sha)
    assert list(blobs.path(sha).parent.iterdir()) == []  # temp file cleaned up


def test_corrupt_blob_detected_on_verified_read(tmp_path):
    blobs = BlobStore(tmp_path)
    sha = blobs.put(b"original")
    blobs.path(sha).write_bytes(b"tampered")
    with pytest.raises(BlobCorrupt):
        blobs.get(sha, verify=True)


def test_blob_path_rejects_non_hash_names(tmp_path):
    with pytest.raises(BlobNotFound):
        BlobStore(tmp_path).get("../../etc/passwd")


# -- job store ----------------------------------------------------------------


def test_submit_claim_complete(store):
    job, created = store.submit("k", {"a": 1}, "key1")
    assert created and job.state == "queued"
    claimed = store.claim("k", "w1")
    assert claimed.id == job.id and claimed.state == "running" and claimed.attempts == 1
    assert store.complete(claimed, {"out": 1})
    done = store.get(job.id)
    assert done.state == "succeeded" and done.result == {"out": 1}


def test_claim_is_fifo_and_empty_returns_none(store, clock):
    a, _ = store.submit("k", {}, "a")
    clock.t += 1
    b, _ = store.submit("k", {}, "b")
    assert store.claim("k", "w").id == a.id
    assert store.claim("k", "w").id == b.id
    assert store.claim("k", "w") is None


def test_concurrent_claims_never_hand_out_a_job_twice(store):
    for i in range(3):
        store.submit("k", {"i": i}, f"c{i}")
    got: list[str] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        while (job := store.claim("k", f"w{n}")) is not None:
            with lock:
                got.append(job.id)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 3 and len(set(got)) == 3


def test_queue_full_raises_at_max_depth(store):
    for i in range(3):
        store.submit("k", {}, f"d{i}")
    with pytest.raises(QueueFull) as e:
        store.submit("k", {}, "d3")
    assert e.value.depth == 3


def test_finished_jobs_do_not_count_toward_depth(store):
    for i in range(3):
        store.submit("k", {}, f"e{i}")
    store.complete(store.claim("k", "w"), {})
    store.submit("k", {}, "e3")  # does not raise


def test_idempotency_key_returns_original_job(store):
    a, created_a = store.submit("k", {"x": 1}, "same", idempotency_key="idem-1")
    b, created_b = store.submit("k", {"x": 1}, "same", idempotency_key="idem-1", dedupe=False)
    assert created_a and not created_b and a.id == b.id


def test_idempotency_key_with_different_body_conflicts(store):
    store.submit("k", {"x": 1}, "one", idempotency_key="idem-2")
    with pytest.raises(IdempotencyConflict):
        store.submit("k", {"x": 2}, "two", idempotency_key="idem-2")


def test_identical_input_coalesces_and_caches(store):
    a, _ = store.submit("k", {"x": 1}, "same-input")
    b, created = store.submit("k", {"x": 1}, "same-input")
    assert not created and b.id == a.id  # in-flight duplicate
    store.complete(store.claim("k", "w"), {"r": 1})
    c, created = store.submit("k", {"x": 1}, "same-input")
    assert not created and c.id == a.id and c.state == "succeeded"  # cached result


def test_failed_input_is_not_served_from_cache(store):
    a, _ = store.submit("k", {}, "bad")
    store.fail(store.claim("k", "w"), "boom", retryable=False)
    b, created = store.submit("k", {}, "bad")
    assert created and b.id != a.id


def test_expired_lease_requeues_job(store, clock):
    job, _ = store.submit("k", {}, "lease")
    store.claim("k", "dead-worker")
    clock.t += LIMITS["k"].lease_s + 0.1
    assert store.claim("k", "w2") is None  # requeued, but inside retry backoff
    clock.t += LIMITS["k"].retry_backoff_s
    again = store.claim("k", "w2")
    assert again.id == job.id and again.attempts == 2 and "lease expired" in again.error


def expire_and_reap(store, clock):
    """Let the running job's lease lapse, let claim() reap it, then wait out the backoff."""
    clock.t += LIMITS["k"].lease_s + 0.1
    assert store.claim("k", "reaper") is None
    clock.t += LIMITS["k"].retry_backoff_s


def test_expired_lease_with_no_attempts_left_goes_dead(store, clock):
    job, _ = store.submit("k", {}, "lease-dead")
    assert store.claim("k", "w") is not None
    expire_and_reap(store, clock)
    assert store.claim("k", "w") is not None
    expire_and_reap(store, clock)
    assert store.claim("k", "w") is None
    dead = store.get(job.id)
    assert dead.state == "dead" and "no attempts left" in dead.error


def test_stale_worker_cannot_complete_after_losing_its_lease(store, clock):
    store.submit("k", {}, "fence")
    stale = store.claim("k", "slow-worker")
    expire_and_reap(store, clock)
    fresh = store.claim("k", "fast-worker")
    assert not store.renew(stale)
    assert not store.complete(stale, {"from": "stale"})
    assert store.fail(stale, "x", retryable=True) is None
    assert store.complete(fresh, {"from": "fresh"})
    assert store.get(fresh.id).result == {"from": "fresh"}


def test_retryable_failure_requeues_with_backoff_then_dies(store, clock):
    job, _ = store.submit("k", {}, "retry")
    assert store.fail(store.claim("k", "w"), "transient", retryable=True) == "queued"
    assert store.claim("k", "w") is None  # backoff
    clock.t += LIMITS["k"].retry_backoff_s
    assert store.fail(store.claim("k", "w"), "transient again", retryable=True) == "dead"
    assert store.get(job.id).error == "transient again"


def test_permanent_failure_is_not_retried(store):
    job, _ = store.submit("k", {}, "perm")
    assert store.fail(store.claim("k", "w"), "bad input", retryable=False) == "failed"
    assert store.claim("k", "w") is None


def test_renew_sets_substate_and_extends_lease(store, clock):
    store.submit("k", {}, "renew")
    job = store.claim("k", "w")
    clock.t += 5
    assert store.renew(job, "sampling")
    clock.t += 9  # past the original lease, within the renewed one
    assert store.claim("k", "other") is None
    assert store.get(job.id).substate == "sampling"


def test_jobs_survive_reopening_the_database(tmp_path, clock):
    path = tmp_path / "durable.sqlite3"
    job, _ = JobStore(path, LIMITS, 5000, clock).submit("k", {"keep": True}, "durable")
    reopened = JobStore(path, LIMITS, 5000, clock)
    assert reopened.get(job.id).input == {"keep": True}
    assert sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_stats_latency_percentiles(store, clock):
    for i, (wait, service) in enumerate([(1, 2), (3, 4), (5, 6)]):
        store.submit("k", {}, f"s{i}")
        clock.t += wait
        job = store.claim("k", "w")
        clock.t += service
        store.complete(job, {})
    s = store.stats("k", window=10)
    assert s["counts"]["succeeded"] == 3 and s["depth"] == 0
    assert s["queue_wait_s"]["p50"] == pytest.approx(3)
    assert s["service_s"]["p50"] == pytest.approx(4)
    assert s["end_to_end_s"]["max"] == pytest.approx(11)


def test_percentile_closed_form():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
