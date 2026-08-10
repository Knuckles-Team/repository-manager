"""RMDD-15: checksummed, bounded, atomic artifact/log transport with quarantine.

Every failure mode -- oversized transfer, digest mismatch, path traversal,
and a partial/interrupted stream -- must quarantine the partial bytes (never
silently delete them) and never publish to the final content-addressed path.
"""

from __future__ import annotations

import hashlib

import pytest

from repository_manager.remote_execution.artifact_transport import (
    ArtifactStagingReceiver,
    ArtifactTransferError,
    ArtifactTransferOutcome,
    PathTraversalError,
)


def _receiver(tmp_path, max_bytes: int = 1024) -> ArtifactStagingReceiver:
    return ArtifactStagingReceiver(tmp_path / "artifacts", max_bytes=max_bytes)


def test_receive_publishes_a_verified_artifact_content_addressed(tmp_path) -> None:
    receiver = _receiver(tmp_path)
    payload = b"artifact-bytes-0123456789"
    digest = hashlib.sha256(payload).hexdigest()

    receipt = receiver.receive(
        "logs/build.txt",
        [payload],
        declared_size=len(payload),
        host_id="worker:remote-1",
        source_description="test fixture",
    )

    assert receipt.outcome == ArtifactTransferOutcome.PUBLISHED
    assert receipt.reference.content_address == digest
    assert receipt.reference.relative_path == "logs/build.txt"
    published = tmp_path / "artifacts" / digest
    assert published.is_file()
    assert published.read_bytes() == payload


@pytest.mark.parametrize(
    "traversal_path",
    ["../escape.txt", "a/../../etc/passwd", "/absolute/path", "a/../b"],
)
def test_receive_refuses_path_traversal_and_never_writes_outside_root(
    tmp_path, traversal_path: str
) -> None:
    receiver = _receiver(tmp_path)
    payload = b"payload"

    with pytest.raises(PathTraversalError) as excinfo:
        receiver.receive(
            traversal_path,
            [payload],
            declared_size=len(payload),
            host_id="worker:remote-1",
            source_description="test fixture",
        )
    assert excinfo.value.outcome == ArtifactTransferOutcome.QUARANTINED_PATH
    # Nothing was streamed for a path-traversal refusal: it is rejected before
    # any bytes are written to a temp file, so there is no quarantine file --
    # confirm no unexpected content landed anywhere under the root either.
    written_files = list((tmp_path / "artifacts").rglob("*"))
    assert not any(f.is_file() and f.stat().st_size > 0 for f in written_files)


def test_receive_quarantines_and_refuses_a_declared_digest_mismatch(tmp_path) -> None:
    receiver = _receiver(tmp_path)
    payload = b"payload-bytes"
    wrong_digest = "0" * 64

    with pytest.raises(ArtifactTransferError) as excinfo:
        receiver.receive(
            "artifact.bin",
            [payload],
            declared_size=len(payload),
            host_id="worker:remote-1",
            source_description="test fixture",
            declared_digest=wrong_digest,
        )
    assert excinfo.value.outcome == ArtifactTransferOutcome.QUARANTINED_DIGEST
    assert excinfo.value.quarantine_path is not None
    # The quarantine directory holds the evidence; nothing was published.
    assert list((tmp_path / "artifacts").glob(wrong_digest)) == []
    quarantine_dir = tmp_path / "artifacts" / ".quarantine"
    assert any(quarantine_dir.iterdir())


def test_receive_quarantines_an_oversized_transfer(tmp_path) -> None:
    receiver = _receiver(tmp_path, max_bytes=8)
    payload = b"this payload is far too large for the bound"

    with pytest.raises(ArtifactTransferError) as excinfo:
        receiver.receive(
            "artifact.bin",
            [payload],
            declared_size=len(payload),
            host_id="worker:remote-1",
            source_description="test fixture",
        )
    assert excinfo.value.outcome == ArtifactTransferOutcome.QUARANTINED_SIZE
    assert list((tmp_path / "artifacts").glob("*")) != []  # .staging/.quarantine exist
    published_digest = hashlib.sha256(payload).hexdigest()
    assert not (tmp_path / "artifacts" / published_digest).exists()


def test_receive_quarantines_a_declared_size_that_exceeds_the_bound_up_front(
    tmp_path,
) -> None:
    receiver = _receiver(tmp_path, max_bytes=8)

    with pytest.raises(ArtifactTransferError) as excinfo:
        receiver.receive(
            "artifact.bin",
            [b"short"],
            declared_size=10_000,
            host_id="worker:remote-1",
            source_description="test fixture",
        )
    assert excinfo.value.outcome == ArtifactTransferOutcome.QUARANTINED_SIZE


def test_receive_quarantines_a_partial_transfer_shorter_than_declared(tmp_path) -> None:
    receiver = _receiver(tmp_path)
    payload = b"short"

    with pytest.raises(ArtifactTransferError) as excinfo:
        receiver.receive(
            "artifact.bin",
            [payload],
            declared_size=len(payload) + 50,  # declares more than is ever sent
            host_id="worker:remote-1",
            source_description="test fixture",
        )
    assert excinfo.value.outcome == ArtifactTransferOutcome.QUARANTINED_PARTIAL
    assert excinfo.value.quarantine_path is not None


def test_receive_quarantines_a_disconnected_source_stream(tmp_path) -> None:
    """A source iterator that raises mid-stream must never publish partial bytes."""

    receiver = _receiver(tmp_path)

    def _disconnecting_chunks():
        yield b"partial-"
        raise ConnectionError("source host disconnected")

    with pytest.raises(ArtifactTransferError) as excinfo:
        receiver.receive(
            "artifact.bin",
            _disconnecting_chunks(),
            declared_size=100,
            host_id="worker:remote-1",
            source_description="test fixture",
        )
    assert excinfo.value.outcome == ArtifactTransferOutcome.QUARANTINED_PARTIAL
    assert excinfo.value.quarantine_path is not None
    # The exception cause is preserved (H-12): never a discarded exception.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    published = list((tmp_path / "artifacts").glob("[0-9a-f]" * 64))
    assert published == []


def test_receive_log_bounds_the_tail_and_publishes_content_addressed(tmp_path) -> None:
    receiver = _receiver(tmp_path)
    payload = b"log line 1\nlog line 2\n" * 10

    receipt = receiver.receive_log(
        "logs/job.log",
        [payload],
        declared_size=len(payload),
        host_id="worker:remote-1",
        source_description="test fixture",
        tail_bytes_limit=16,
    )
    assert receipt.outcome == ArtifactTransferOutcome.PUBLISHED
    assert receipt.reference.tail_bytes == 16
    assert receipt.reference.content_address == hashlib.sha256(payload).hexdigest()


def test_receiver_root_directories_are_created_and_isolated(tmp_path) -> None:
    _receiver(tmp_path)
    assert (tmp_path / "artifacts" / ".staging").is_dir()
    assert (tmp_path / "artifacts" / ".quarantine").is_dir()


def test_republishing_the_same_artifact_after_a_disconnect_is_idempotent(
    tmp_path,
) -> None:
    """Disconnect-after-publication proof: a retry of an already-published
    artifact must be safe.  Content addressing makes this trivially
    idempotent -- the second ``receive`` computes the same digest and
    ``os.replace``s onto the same final path, never corrupting or duplicating
    it.  This is the "disconnect ... after artifact publication yields
    deterministic retry" required test.
    """

    receiver = _receiver(tmp_path)
    payload = b"idempotent-republish-payload"
    digest = hashlib.sha256(payload).hexdigest()

    first = receiver.receive(
        "artifact.bin",
        [payload],
        declared_size=len(payload),
        host_id="worker:remote-1",
        source_description="attempt 1",
    )
    # Simulate the caller never seeing attempt 1's ack (as if the connection
    # dropped immediately after the remote write completed) and retrying.
    second = receiver.receive(
        "artifact.bin",
        [payload],
        declared_size=len(payload),
        host_id="worker:remote-1",
        source_description="attempt 2 (retry after disconnect)",
    )

    assert first.reference.content_address == second.reference.content_address == digest
    published = tmp_path / "artifacts" / digest
    assert published.is_file()
    assert published.read_bytes() == payload
    # Exactly one published file exists for this content -- no duplicate.
    assert len(list((tmp_path / "artifacts").glob(f"{digest}*"))) == 1
