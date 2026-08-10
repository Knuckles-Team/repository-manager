"""Refusal taxonomy for concept-claim coordination (RMDD-17).

Every refusal maps onto a :class:`repository_manager.development.RefusalCode`
(C-10, ``CONTRACT-FREEZES.md``) category so MCP/CLI/WorkItem/skill/metrics
surfaces share one vocabulary instead of RMDD-17 inventing a second one.
"""

from __future__ import annotations

from repository_manager.development import RefusalCode

__all__ = [
    "ConceptAuthorityUnavailable",
    "ConceptClaimConflict",
    "ConceptClaimFenceConflict",
    "ConceptClaimIdUnavailable",
    "ConceptClaimNotFound",
    "ConceptClaimUnauthorized",
    "ConceptCoordinationError",
    "ConceptVerificationRefused",
]


class ConceptCoordinationError(Exception):
    """Base error for every concept-claim coordination refusal.

    ``code`` is one of the stable C-10 categories; human ``str(exc)`` text may
    improve without breaking automation, but ``code`` is versioned.
    """

    code: RefusalCode = RefusalCode.INTERNAL_ERROR

    def __init__(self, message: str, *, code: RefusalCode | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConceptAuthorityUnavailable(ConceptCoordinationError):
    """The central concept authority (RMDD-16) could not be reached.

    Repository-side fragments are the audit projection, never a fallback
    allocator (lane brief, "Non-negotiable correctness constraints"): this
    error is raised instead of minting or reusing a local ID. ``unreachable``
    names exactly what could not be reached (an import, a port method, a
    configured endpoint) so the refusal is actionable rather than opaque.
    The original exception, if any, is preserved as ``__cause__`` — never
    swallowed (H-12).
    """

    code = RefusalCode.DEPENDENCY_BLOCKED

    def __init__(self, unreachable: str, *, cause: BaseException | None = None) -> None:
        super().__init__(
            f"central concept authority unreachable: {unreachable}",
            code=RefusalCode.DEPENDENCY_BLOCKED,
        )
        self.unreachable = unreachable
        if cause is not None:
            self.__cause__ = cause


class ConceptClaimConflict(ConceptCoordinationError):
    """A request key or concept id already names different immutable input."""

    code = RefusalCode.CONFLICT_BASE_MOVED


class ConceptClaimIdUnavailable(ConceptClaimConflict):
    """The concept id is already owned by another active/terminal claim."""

    code = RefusalCode.DUPLICATE_REQUEST


class ConceptClaimFenceConflict(ConceptClaimConflict):
    """A lifecycle mutation used a stale owner or fencing token."""

    code = RefusalCode.STALE_FENCE_DUPLICATE_EFFECT


class ConceptClaimUnauthorized(ConceptCoordinationError):
    """The caller does not own the tenant/owner scope of this claim."""

    code = RefusalCode.UNAUTHORIZED_TARGET


class ConceptClaimNotFound(ConceptCoordinationError):
    """The requested reservation is not present in the authority/projection."""

    code = RefusalCode.INVALID_REQUEST


class ConceptVerificationRefused(ConceptCoordinationError):
    """A candidate/generation could not be certified against concept claims."""

    code = RefusalCode.VALIDATION_CANDIDATE_FAILURE
