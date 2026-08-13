"""RMDD-15 remote placement, worker lifecycle, and artifact transport.

This package composes the frozen RMDD-07 local execution contract
(:mod:`repository_manager.execution`), the frozen RMDD-08 resource scheduler
(:mod:`repository_manager.resource_scheduler` /
:mod:`repository_manager.capacity`), and the frozen RMDD-14 tunnel execution
seam (:mod:`tunnel_manager.remote_execution`) into Repository Manager worker
semantics: durable capacity/heartbeat state, immutable source
materialization, bounded artifact transport, cancellation, and host-loss
recovery for a job that may run locally or on an authorized inventory host.

It does not modify tunnel-manager, does not add a public MCP/CLI entrypoint,
does not implement build/validation/merge policy, and never acquires a
controller-local filesystem lock or moves a target branch.  See
``README.md`` in this package for the architecture and the exact frozen
interfaces consumed.
"""

from __future__ import annotations

from .artifact_transport import (
    ArtifactStagingReceiver,
    ArtifactTransferError,
    ArtifactTransferOutcome,
    ArtifactTransferReceipt,
    PathTraversalError,
)
from .bootstrap import (
    RemoteWorkerBootstrap,
    RemoteWorkerBootstrapError,
    build_bootstrap_command,
)
from .executor import (
    RemoteExecutorPort,
    RemoteWorkerExecutor,
    from_remote_result,
    to_remote_request,
)
from .host_loss import HostLossDecision, HostLossReconciler, ReservationReleasePort
from .registry import (
    RemoteWorkerProfile,
    RemoteWorkerRegistry,
    RemoteWorkerRegistryError,
)
from .source_staging import (
    DirtySourceError,
    ImmutableSourceStaging,
    SourceVerificationError,
    StagedSource,
)
from .ssh_executor import RemoteSshExecutionUnavailableError, TunnelSSHExecutor

__all__ = [
    "ArtifactStagingReceiver",
    "ArtifactTransferError",
    "ArtifactTransferOutcome",
    "ArtifactTransferReceipt",
    "DirtySourceError",
    "HostLossDecision",
    "HostLossReconciler",
    "ImmutableSourceStaging",
    "PathTraversalError",
    "RemoteExecutorPort",
    "RemoteSshExecutionUnavailableError",
    "RemoteWorkerBootstrap",
    "RemoteWorkerBootstrapError",
    "RemoteWorkerExecutor",
    "RemoteWorkerProfile",
    "RemoteWorkerRegistry",
    "RemoteWorkerRegistryError",
    "ReservationReleasePort",
    "SourceVerificationError",
    "StagedSource",
    "TunnelSSHExecutor",
    "build_bootstrap_command",
    "from_remote_result",
    "to_remote_request",
]
