"""Stable string codes used by repository-development contracts.

The values in this module are persisted and cross the MCP, CLI, WorkItem, and
worker boundaries.  Rename a Python member only with a compatibility decision;
the string value is the wire-level API.
"""

from enum import StrEnum


class OperationKind(StrEnum):
    """Repository-development operation families."""

    LANE_ALLOCATE = "lane.allocate"
    REPOSITORY = "repository"
    VALIDATION = "validation"
    BUILD = "build"
    MERGE = "merge"
    RELEASE = "release"
    CANDIDATE_SUBMIT = "candidate.submit"
    GENERATION_CERTIFY = "generation.certify"
    BRANCH_LAND = "branch.land"
    WORKSPACE_VALIDATE = "workspace.validate"
    WORKSPACE_BUMP = "workspace.bump"
    WORKSPACE_PUSH = "workspace.push"


class ValidationStage(StrEnum):
    """Ordered validation confidence stages."""

    FEEDBACK = "feedback"
    INTEGRATION = "integration"
    CERTIFICATION = "certification"
    SMOKE = "smoke"
    RELEASE = "release"


class JobState(StrEnum):
    """The graph-os WorkItem lifecycle projected for repository jobs."""

    SUBMITTED = "submitted"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead-letter"


class LaneState(StrEnum):
    """Lifecycle of an isolated development lane."""

    ALLOCATING = "allocating"
    ACTIVE = "active"
    SUBMITTED = "submitted"
    LANDED = "landed"
    ABORTED = "aborted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class CandidateState(StrEnum):
    """Lifecycle of a committed branch candidate."""

    QUEUED = "queued"
    VALIDATING = "validating"
    READY = "ready"
    LANDING = "landing"
    LANDED = "landed"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    FAILED = "failed"


class GenerationState(StrEnum):
    """Lifecycle of an immutable candidate generation."""

    OPEN = "open"
    SEALED = "sealed"
    INTEGRATING = "integrating"
    CERTIFIED = "certified"
    LANDING = "landing"
    LANDED = "landed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ReleasePlanState(StrEnum):
    """Workspace release-plan lifecycle."""

    DRAFT = "draft"
    FROZEN = "frozen"
    APPLIED = "applied"
    REJECTED = "rejected"


class ReservationState(StrEnum):
    """Resource reservation lifecycle."""

    RESERVED = "reserved"
    RELEASED = "released"
    EXPIRED = "expired"
    REFUSED = "refused"


class ExecutionOutcome(StrEnum):
    """Outcome of one fixed-argv execution."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    REFUSED = "refused"


class LandingOutcome(StrEnum):
    """Fenced target-branch landing result."""

    LANDED = "landed"
    TARGET_MOVED = "target_moved"
    REFUSED = "refused"
    FAILED = "failed"


class EvidenceOutcome(StrEnum):
    """Outcome of a validation gate or evidence collection run."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REFUSED = "refused"


class BuildOutcome(StrEnum):
    """Build-cache result classes shared by build consumers."""

    HIT = "hit"
    WAITED_HIT = "waited_hit"
    PRODUCED_MISS = "produced_miss"
    DEGRADED_UNCACHEABLE = "degraded_uncacheable"
    CORRUPTED_ENTRY = "corrupted_entry"
    REFUSED = "refused"
    FAILED = "failed"


class TargetKind(StrEnum):
    """Public execution target forms.

    ``INVENTORY_ALIAS`` is deliberately the only remote form.  Connection
    material belongs to tunnel-manager and never appears in this contract.
    ``REMOTE`` remains a Python spelling for the same wire value so callers
    can use the natural term without introducing a second protocol value.
    """

    LOCAL = "local"
    INVENTORY_ALIAS = "inventory_alias"
    REMOTE = "inventory_alias"


class FailureClass(StrEnum):
    """Stable failure categories from C-10."""

    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED_TARGET = "unauthorized_target"
    CONFLICT_BASE_MOVED = "conflict_base_moved"
    CAPACITY_DISK_DEFERRED = "capacity_disk_deferred"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    CANCELLED_DEADLINE = "cancelled_deadline"
    WORKER_ENVIRONMENT_FAILURE = "worker_environment_failure"
    VALIDATION_CANDIDATE_FAILURE = "validation_candidate_failure"
    STALE_FENCE_DUPLICATE_EFFECT = "stale_fence_duplicate_effect"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    INTERNAL_ERROR = "internal_error"


class RefusalCode(StrEnum):
    """Stable refusal codes shared by all public surfaces.

    The first eleven values are the C-10 category codes.  The remaining
    values identify common, actionable refinements without changing those
    category values.
    """

    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED_TARGET = "unauthorized_target"
    CONFLICT_BASE_MOVED = "conflict_base_moved"
    CAPACITY_DISK_DEFERRED = "capacity_disk_deferred"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    CANCELLED_DEADLINE = "cancelled_deadline"
    WORKER_ENVIRONMENT_FAILURE = "worker_environment_failure"
    VALIDATION_CANDIDATE_FAILURE = "validation_candidate_failure"
    STALE_FENCE_DUPLICATE_EFFECT = "stale_fence_duplicate_effect"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    INTERNAL_ERROR = "internal_error"
    PATH_OUTSIDE_CONFIGURED_ROOT = "path_outside_configured_root"
    INVALID_GIT_REF = "invalid_git_ref"
    INVALID_GIT_SHA = "invalid_git_sha"
    REMOTE_ALIAS_REQUIRED = "remote_alias_required"
    REMOTE_CREDENTIALS_FORBIDDEN = "remote_credentials_forbidden"
    SHELL_COMMAND_FORBIDDEN = "shell_command_forbidden"
    RESOURCE_LIMIT_INVALID = "resource_limit_invalid"
    INVALID_STATE_COMBINATION = "invalid_state_combination"
    DUPLICATE_REQUEST = "duplicate_request"


C10_FAILURE_CODES: tuple[str, ...] = tuple(member.value for member in FailureClass)
