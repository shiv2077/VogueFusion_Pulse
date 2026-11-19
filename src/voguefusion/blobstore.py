"""Content-addressed blob store on a local directory.

A blob's name is the SHA-256 of its bytes, stored at <root>/<aa>/<sha256>.
Writes go to a temporary file in the same directory, are fsynced, then
renamed into place. rename is atomic on POSIX, so a crash leaves either no
file or a complete one at the final address, never a partial one. The same
layout maps directly onto an object store key.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BlobNotFound(KeyError):
    pass


class BlobCorrupt(ValueError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        if not SHA256_RE.match(sha):
            raise BlobNotFound(sha)
        return self.root / sha[:2] / sha

    def exists(self, sha: str) -> bool:
        try:
            return self.path(sha).is_file()
        except BlobNotFound:
            return False

    def put(self, data: bytes) -> str:
        """Store bytes, return their sha256. Idempotent: an existing blob is not rewritten."""
        sha = sha256_hex(data)
        final = self.path(sha)
        if final.is_file():
            return sha
        final.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
            dir_fd = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return sha

    def get(self, sha: str, verify: bool = False) -> bytes:
        try:
            data = self.path(sha).read_bytes()
        except FileNotFoundError as e:
            raise BlobNotFound(sha) from e
        if verify and sha256_hex(data) != sha:
            raise BlobCorrupt(sha)
        return data
