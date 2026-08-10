"""Checksummed, bounded, atomic artifact/log transport with quarantine.

Bytes only ever land at the final content-addressed path after the complete
stream has been consumed and its declared size and digest have been
independently verified against what was actually written.  Any failure --
oversized transfer, digest mismatch, an unsafe relative path, or an
interrupted/disconnected source -- leaves the partial data quarantined
on-disk (never deleted silently, never published) and raises a typed,
provenance-carrying error rather than discarding the cause (H-12).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from repository_manager.development import ArtifactReference, LogReference

_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_DEFAULT_LOG_TAIL_BYTES = 64 * 1024


class ArtifactTransferOutcome(StrEnum):
    """Terminal disposition of one staged transfer."""

    PUBLISHED = "published"
    QUARANTINED_SIZE = "quarantined_size_exceeded"
    QUARANTINED_DIGEST = "quarantined_digest_mismatch"
    QUARANTINED_PARTIAL = "quarantined_partial_transfer"
    QUARANTINED_PATH = "quarantined_path_traversal"
    QUARANTINED_INVALID = "quarantined_invalid_reference"


class ArtifactTransferError(ValueError):
    """A transfer could not be published; ``quarantine_path`` names the
    on-disk evidence retained, if any."""

    def __init__(
        self,
        outcome: ArtifactTransferOutcome,
        message: str,
        *,
        quarantine_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.quarantine_path = quarantine_path


class PathTraversalError(ArtifactTransferError):
    """The declared relative path attempted to escape the artifact root."""

    def __init__(self, message: str) -> None:
        super().__init__(ArtifactTransferOutcome.QUARANTINED_PATH, message)


@dataclass(frozen=True)
class ArtifactTransferReceipt:
    """Source/host/result provenance for one published artifact or log."""

    reference: ArtifactReference | LogReference
    host_id: str
    source_description: str
    outcome: ArtifactTransferOutcome


def _require_relative(relative_path: str) -> str:
    path = Path(relative_path)
    if (
        not relative_path
        or relative_path.strip() != relative_path
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise PathTraversalError(f"unsafe relative artifact path: {relative_path!r}")
    return relative_path


class ArtifactStagingReceiver:
    """Stream bytes into a bounded, checksummed, content-addressed store."""

    def __init__(
        self, root: str | Path, *, max_bytes: int = _MAX_ARTIFACT_BYTES
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)
        (self.root / ".quarantine").mkdir(parents=True, exist_ok=True)

    def receive(
        self,
        relative_path: str,
        chunks: Iterable[bytes],
        *,
        declared_size: int,
        host_id: str,
        source_description: str,
        declared_digest: str | None = None,
        media_type: str = "application/octet-stream",
        producer_job_id: str | None = None,
    ) -> ArtifactTransferReceipt:
        """Stream, verify, and atomically publish one bounded artifact."""

        relative_path = _require_relative(relative_path)
        if declared_size < 0 or declared_size > self.max_bytes:
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_SIZE,
                f"declared size {declared_size} exceeds bound {self.max_bytes}",
            )

        temp_path, written, computed_digest = self._stream_to_temp(
            chunks, declared_size=declared_size, declared_digest=declared_digest
        )

        try:
            reference = ArtifactReference(
                content_address=computed_digest,
                relative_path=relative_path,
                size_bytes=written,
                media_type=media_type,
                producer_job_id=producer_job_id,
            )
        except ValidationError as exc:
            quarantine = self._quarantine(temp_path)
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_INVALID,
                f"artifact reference failed validation: {exc.error_count()} error(s)",
                quarantine_path=str(quarantine),
            ) from exc

        final_path = self.root / computed_digest
        os.replace(temp_path, final_path)
        return ArtifactTransferReceipt(
            reference=reference,
            host_id=host_id,
            source_description=source_description,
            outcome=ArtifactTransferOutcome.PUBLISHED,
        )

    def receive_log(
        self,
        relative_path: str,
        chunks: Iterable[bytes],
        *,
        declared_size: int,
        host_id: str,
        source_description: str,
        declared_digest: str | None = None,
        media_type: str = "text/plain",
        tail_bytes_limit: int = _DEFAULT_LOG_TAIL_BYTES,
    ) -> ArtifactTransferReceipt:
        """Stream, verify, and atomically publish one bounded log file."""

        relative_path = _require_relative(relative_path)
        if declared_size < 0 or declared_size > self.max_bytes:
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_SIZE,
                f"declared size {declared_size} exceeds bound {self.max_bytes}",
            )

        temp_path, written, computed_digest = self._stream_to_temp(
            chunks, declared_size=declared_size, declared_digest=declared_digest
        )

        try:
            reference = LogReference(
                content_address=computed_digest,
                relative_path=relative_path,
                size_bytes=written,
                tail_bytes=min(written, tail_bytes_limit),
                media_type=media_type,
            )
        except ValidationError as exc:
            quarantine = self._quarantine(temp_path)
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_INVALID,
                f"log reference failed validation: {exc.error_count()} error(s)",
                quarantine_path=str(quarantine),
            ) from exc

        final_path = self.root / computed_digest
        os.replace(temp_path, final_path)
        return ArtifactTransferReceipt(
            reference=reference,
            host_id=host_id,
            source_description=source_description,
            outcome=ArtifactTransferOutcome.PUBLISHED,
        )

    def _stream_to_temp(
        self,
        chunks: Iterable[bytes],
        *,
        declared_size: int,
        declared_digest: str | None,
    ) -> tuple[Path, int, str]:
        temp_path = self.root / ".staging" / f"{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        written = 0
        try:
            with temp_path.open("wb") as handle:
                for chunk in chunks:
                    written += len(chunk)
                    if written > self.max_bytes or written > declared_size:
                        raise ArtifactTransferError(
                            ArtifactTransferOutcome.QUARANTINED_SIZE,
                            "transfer exceeded its declared/bounded size",
                        )
                    digest.update(chunk)
                    handle.write(chunk)
        except ArtifactTransferError as exc:
            quarantine = self._quarantine(temp_path)
            raise ArtifactTransferError(
                exc.outcome, str(exc), quarantine_path=str(quarantine)
            ) from exc
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            quarantine = self._quarantine(temp_path)
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_PARTIAL,
                f"artifact source disconnected or failed: {type(exc).__name__}",
                quarantine_path=str(quarantine),
            ) from exc

        if written != declared_size:
            quarantine = self._quarantine(temp_path)
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_PARTIAL,
                f"transfer produced {written} bytes, declared {declared_size}",
                quarantine_path=str(quarantine),
            )
        computed_digest = digest.hexdigest()
        if declared_digest is not None and declared_digest != computed_digest:
            quarantine = self._quarantine(temp_path)
            raise ArtifactTransferError(
                ArtifactTransferOutcome.QUARANTINED_DIGEST,
                "computed digest does not match declared digest",
                quarantine_path=str(quarantine),
            )
        return temp_path, written, computed_digest

    def _quarantine(self, temp_path: Path) -> Path:
        if not temp_path.exists():
            return temp_path
        quarantine_path = self.root / ".quarantine" / temp_path.name
        try:
            os.replace(temp_path, quarantine_path)
        except OSError:
            temp_path.unlink(missing_ok=True)
            return temp_path
        return quarantine_path


__all__ = [
    "ArtifactStagingReceiver",
    "ArtifactTransferError",
    "ArtifactTransferOutcome",
    "ArtifactTransferReceipt",
    "PathTraversalError",
]
