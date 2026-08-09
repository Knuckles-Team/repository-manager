"""Read-only expiry assessment and guarded lane reclamation.

Assessment never removes a path or changes a ref.  An execution plan is fenced
to the owner and current lane token, re-checks every safety condition, and then
delegates removal to ``WorktreeManager.remove`` so its existing occupancy and
branch-retention guards remain the final authority.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .lane_record import LaneLifecycleState, LaneRecord


class ReclamationReason(StrEnum):
    EXPIRED = "expired"
    LIVE_PROCESS = "live_process"
    RECENT_HEARTBEAT = "recent_heartbeat"
    DIRTY_WORKTREE = "dirty_worktree"
    UNMERGED_COMMITS = "unmerged_commits"
    ACTIVE_JOB = "active_job"
    ACTIVE_CANDIDATE = "active_candidate"
    CONCEPT_CLAIM = "concept_claim"
    MISSING_BACKUP_ANCHOR = "missing_backup_anchor"
    OCCUPIED = "occupied"
    WORKTREE_MISSING = "worktree_missing"
    GIT_UNAVAILABLE = "git_unavailable"
    FENCE_STALE = "fence_stale"


@dataclass(frozen=True)
class SafetyCheck:
    """One explainable cleanup condition."""

    name: str
    allowed: bool
    reason: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed": self.allowed,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ExpiryCandidate:
    """Assessment result for one expired lane."""

    lane: LaneRecord
    eligible: bool
    checks: tuple[SafetyCheck, ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(check.reason for check in self.checks if not check.allowed)

    @property
    def refusal_codes(self) -> tuple[str, ...]:
        return tuple(
            check.name for check in self.checks if not check.allowed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.model_dump(mode="json"),
            "eligible": self.eligible,
            "checks": [check.as_dict() for check in self.checks],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CleanupPlan:
    """Previewable cleanup intent; no filesystem operation is implied."""

    plan_id: str
    lane_id: str
    repository_path: str
    branch: str
    worktree_path: str
    owner_id: str
    fence: str
    created_at: datetime
    assessment: ExpiryCandidate
    guarded_remove: bool = True
    requested_job_id: str | None = None
    requested_job_fence: str | None = None

    @property
    def ok(self) -> bool:
        return self.assessment.eligible and self.guarded_remove

    @property
    def preview_only(self) -> bool:
        return self.requested_job_id is None or self.requested_job_fence is None

    @property
    def executable(self) -> bool:
        return self.ok and not self.preview_only

    @property
    def eligible(self) -> bool:
        return self.assessment.eligible

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "lane_id": self.lane_id,
            "repository_path": self.repository_path,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "owner_id": self.owner_id,
            "fence": self.fence,
            "created_at": self.created_at.isoformat(),
            "eligible": self.eligible,
            "ok": self.ok,
            "guarded_remove": self.guarded_remove,
            "requested_job_id": self.requested_job_id,
            "requested_job_fence": self.requested_job_fence,
            "preview_only": self.preview_only,
            "executable": self.executable,
            "assessment": self.assessment.as_dict(),
        }


class CleanupRefused(ValueError):
    """A cleanup plan failed one or more safety checks."""

    def __init__(self, plan: CleanupPlan):
        self.plan = plan
        code = plan.assessment.refusal_codes
        if not code and plan.preview_only:
            code = ("durable_cleanup_job",)
        super().__init__(
            "lane cleanup refused: "
            + ", ".join(code or ("unknown",))
        )


class ReconciliationClass(StrEnum):
    """Read-only durable-record versus Git/filesystem classifications."""

    MANAGED = "managed"
    OBSERVED_LEGACY = "observed_legacy"
    MISSING_WORKTREE = "missing_worktree"
    MISSING_BRANCH = "missing_branch"
    PATH_MISMATCH = "path_mismatch"
    BRANCH_MISMATCH = "branch_mismatch"
    STATE_MISMATCH = "state_mismatch"
    ORPHAN_WORKTREE = "orphan_worktree"


@dataclass(frozen=True)
class ReconciliationFinding:
    lane_id: str | None
    classification: ReconciliationClass
    details: Mapping[str, Any] = field(default_factory=dict)
    repair_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "classification": self.classification.value,
            "details": dict(self.details),
            "repair_required": self.repair_required,
        }


class LaneReclaimer:
    """Assess expired lanes and execute only guarded cleanup plans."""

    def __init__(
        self,
        registry: Any,
        *,
        worktree_manager: Any | None = None,
        git: Any | None = None,
        process_probe: Callable[[LaneRecord], bool | None] | None = None,
        job_probe: Callable[[LaneRecord], bool | None] | None = None,
        candidate_probe: Callable[[LaneRecord], bool | None] | None = None,
        concept_probe: Callable[[LaneRecord], bool | None] | None = None,
        occupancy_probe: Callable[[LaneRecord], bool | None] | None = None,
        cleanup_authority: Any | None = None,
        recent_heartbeat_seconds: int = 60,
        clock: Callable[[], datetime] | Any | None = None,
    ) -> None:
        if recent_heartbeat_seconds < 0:
            raise ValueError("recent_heartbeat_seconds cannot be negative")
        self.registry = registry
        self.worktree_manager = worktree_manager
        self.git = git or getattr(worktree_manager, "git", None)
        self.process_probe = process_probe
        self.job_probe = job_probe
        self.candidate_probe = candidate_probe
        self.concept_probe = concept_probe
        self.occupancy_probe = occupancy_probe
        self.cleanup_authority = cleanup_authority
        self.recent_heartbeat_seconds = recent_heartbeat_seconds
        self._clock = clock

    def _now(self, value: datetime | None = None) -> datetime:
        if value is not None:
            current = value
        elif self._clock is None:
            current = datetime.now(UTC)
        elif callable(self._clock):
            current = self._clock()
        else:
            current = self._clock.now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("reclaimer clock must return an aware datetime")
        return current.astimezone(UTC)

    @staticmethod
    def _result_ok(value: Any) -> bool:
        return getattr(value, "status", "") == "success" or bool(
            isinstance(value, Mapping) and value.get("ok")
        )

    def _git(self, command: str, path: str) -> tuple[bool, str]:
        try:
            if self.git is not None and callable(getattr(self.git, "git_action", None)):
                result = self.git.git_action(
                    command=command, path=path, quiet=True, raw_output=True
                )
                return self._result_ok(result), str(getattr(result, "data", "") or "")
        except Exception:  # noqa: BLE001 - unavailable evidence must refuse cleanup
            return False, ""
        try:
            proc = subprocess.run(
                shlex.split(command),
                cwd=path,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False, ""
        return proc.returncode == 0, (proc.stdout or "").strip()

    def _check_process(self, lane: LaneRecord) -> SafetyCheck:
        if self.process_probe is None:
            return SafetyCheck(
                "live_process",
                False,
                ReclamationReason.LIVE_PROCESS.value + ": liveness evidence unavailable",
            )
        try:
            alive = self.process_probe(lane)
        except Exception:  # noqa: BLE001 - an unknown process is still unsafe
            return SafetyCheck(
                "live_process",
                False,
                ReclamationReason.LIVE_PROCESS.value + ": liveness probe failed",
            )
        if alive is None:
            return SafetyCheck(
                "live_process",
                False,
                ReclamationReason.LIVE_PROCESS.value + ": process liveness unknown",
            )
        return SafetyCheck(
            "live_process",
            not alive,
            "" if not alive else ReclamationReason.LIVE_PROCESS.value,
        )

    def _check_recent(self, lane: LaneRecord, now: datetime) -> SafetyCheck:
        age = (now - lane.heartbeat_at).total_seconds()
        allowed = age >= self.recent_heartbeat_seconds
        return SafetyCheck(
            "recent_heartbeat",
            allowed,
            "" if allowed else ReclamationReason.RECENT_HEARTBEAT.value,
            {"age_seconds": age, "threshold_seconds": self.recent_heartbeat_seconds},
        )

    def _check_git_state(self, lane: LaneRecord) -> tuple[SafetyCheck, ...]:
        if not Path(lane.worktree_path).is_dir():
            return (
                SafetyCheck(
                    "worktree_present",
                    False,
                    ReclamationReason.WORKTREE_MISSING.value,
                ),
            )
        status_ok, status = self._git("git status --porcelain", lane.worktree_path)
        if not status_ok:
            return (
                SafetyCheck(
                    "git_state",
                    False,
                    ReclamationReason.GIT_UNAVAILABLE.value,
                ),
            )
        dirty = bool(status.strip())
        checks: list[SafetyCheck] = [
            SafetyCheck(
                "dirty_worktree",
                not dirty,
                "" if not dirty else ReclamationReason.DIRTY_WORKTREE.value,
                {"status": status.splitlines()[:25]},
            )
        ]
        ancestor_ok, _ = self._git(
            "git merge-base --is-ancestor "
            + shlex.quote(lane.branch)
            + " "
            + shlex.quote(lane.base_ref),
            lane.worktree_path,
        )
        checks.append(
            SafetyCheck(
                "unmerged_commits",
                ancestor_ok,
                "" if ancestor_ok else ReclamationReason.UNMERGED_COMMITS.value,
            )
        )
        return tuple(checks)

    def _check_anchor(self, lane: LaneRecord) -> SafetyCheck:
        anchors = lane.cleanup_anchors or (
            "refs/lane-backup/" + lane.branch.replace("/", "-") ,
        )
        tip_ok, tip = self._git(
            "git rev-parse --verify --quiet "
            + shlex.quote("refs/heads/" + lane.branch),
            lane.repository_path,
        )
        tip_sha = tip.strip()
        if not tip_ok or not re.fullmatch(r"[0-9a-fA-F]{40}", tip_sha):
            return SafetyCheck(
                "backup_anchor",
                False,
                ReclamationReason.MISSING_BACKUP_ANCHOR.value + ": lane tip unavailable",
                {"anchors": list(anchors), "lane_tip": tip_sha},
            )
        for anchor in anchors:
            if not isinstance(anchor, str) or not anchor.strip() or anchor != anchor.strip():
                continue
            is_ref = bool(
                re.fullmatch(r"refs/(?:lane-backup|heads|tags|remotes)/[A-Za-z0-9._/-]+", anchor)
                and ".." not in anchor
                and "//" not in anchor
            )
            is_sha = bool(re.fullmatch(r"[0-9a-fA-F]{40}", anchor))
            if not (is_ref or is_sha):
                continue
            ok, value = self._git(
                "git rev-parse --verify --quiet " + shlex.quote(anchor + "^{commit}"),
                lane.repository_path,
            )
            anchor_sha = value.strip()
            if not ok or not re.fullmatch(r"[0-9a-fA-F]{40}", anchor_sha):
                continue
            preserved, _ = self._git(
                "git merge-base --is-ancestor "
                + shlex.quote(tip_sha)
                + " "
                + shlex.quote(anchor_sha),
                lane.repository_path,
            )
            if preserved:
                return SafetyCheck(
                    "backup_anchor",
                    True,
                    evidence={"anchor": anchor, "sha": anchor_sha, "lane_tip": tip_sha},
                )
        return SafetyCheck(
            "backup_anchor",
            False,
            ReclamationReason.MISSING_BACKUP_ANCHOR.value,
            {"anchors": list(anchors), "lane_tip": tip_sha},
        )

    def _check_claims(self, lane: LaneRecord) -> tuple[SafetyCheck, ...]:
        job_claimed = bool(lane.active_job_ids)
        job_unknown = self.job_probe is None
        if self.job_probe is not None:
            try:
                probed = self.job_probe(lane)
            except Exception:  # noqa: BLE001 - unknown claim state refuses cleanup
                probed = None
            if probed is None:
                job_unknown = True
            else:
                job_claimed = bool(probed)
        candidate_claimed = bool(lane.active_candidate_id)
        candidate_unknown = self.candidate_probe is None
        if self.candidate_probe is not None:
            try:
                probed = self.candidate_probe(lane)
            except Exception:  # noqa: BLE001 - unknown claim state refuses cleanup
                probed = None
            if probed is None:
                candidate_unknown = True
            else:
                candidate_claimed = bool(probed)
        concept_claimed = bool(lane.concept_ids)
        concept_unknown = self.concept_probe is None
        if self.concept_probe is not None:
            try:
                probed = self.concept_probe(lane)
            except Exception:  # noqa: BLE001 - unknown claim state refuses cleanup
                probed = None
            if probed is None:
                concept_unknown = True
            else:
                concept_claimed = bool(probed)
        return (
            SafetyCheck(
                "active_job",
                not job_claimed and not job_unknown,
                ""
                if not job_claimed and not job_unknown
                else (
                    ReclamationReason.ACTIVE_JOB.value
                    if job_claimed
                    else ReclamationReason.ACTIVE_JOB.value + ": claim evidence unavailable"
                ),
                {"job_ids": list(lane.active_job_ids), "evidence_available": not job_unknown},
            ),
            SafetyCheck(
                "active_candidate",
                not candidate_claimed and not candidate_unknown,
                ""
                if not candidate_claimed and not candidate_unknown
                else (
                    ReclamationReason.ACTIVE_CANDIDATE.value
                    if candidate_claimed
                    else ReclamationReason.ACTIVE_CANDIDATE.value + ": claim evidence unavailable"
                ),
                {"candidate_id": lane.active_candidate_id, "evidence_available": not candidate_unknown},
            ),
            SafetyCheck(
                "concept_claim",
                not concept_claimed and not concept_unknown,
                ""
                if not concept_claimed and not concept_unknown
                else (
                    ReclamationReason.CONCEPT_CLAIM.value
                    if concept_claimed
                    else ReclamationReason.CONCEPT_CLAIM.value + ": claim evidence unavailable"
                ),
                {"concept_ids": list(lane.concept_ids), "evidence_available": not concept_unknown},
            ),
        )

    def assess(
        self,
        lane: LaneRecord,
        *,
        now: datetime | None = None,
    ) -> ExpiryCandidate:
        current = self._now(now)
        checks: list[SafetyCheck] = [
            SafetyCheck(
                "expired",
                current >= lane.expires_at,
                ""
                if current >= lane.expires_at
                else ReclamationReason.EXPIRED.value + ": TTL has not elapsed",
            ),
            self._check_process(lane),
            self._check_recent(lane, current),
        ]
        checks.extend(self._check_git_state(lane))
        checks.extend(self._check_claims(lane))
        checks.append(self._check_anchor(lane))
        checks.append(self._check_occupied(lane))
        return ExpiryCandidate(lane, all(check.allowed for check in checks), tuple(checks))

    def select_expiry_candidates(
        self,
        *,
        repository_id: str | None = None,
        now: datetime | None = None,
        include_refused: bool = False,
    ) -> tuple[ExpiryCandidate, ...]:
        records = self.registry.list_records(
            repository_id=repository_id, include_terminal=False
        )
        assessed = tuple(self.assess(record, now=now) for record in records)
        if include_refused:
            return assessed
        return tuple(candidate for candidate in assessed if candidate.eligible)

    def plan_cleanup(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        now: datetime | None = None,
    ) -> CleanupPlan:
        lane = self.registry.require(lane_id)
        if owner_id != lane.owner_id or fence != lane.fence:
            raise ValueError(ReclamationReason.FENCE_STALE.value)
        assessment = self.assess(lane, now=now)
        current = self._now(now)
        digest = f"{lane_id}\0{fence}\0{current.isoformat()}"
        import hashlib

        plan_id = "cleanup:" + hashlib.sha256(digest.encode()).hexdigest()
        return CleanupPlan(
            plan_id=plan_id,
            lane_id=lane_id,
            repository_path=lane.repository_path,
            branch=lane.branch,
            worktree_path=lane.worktree_path,
            owner_id=owner_id,
            fence=fence,
            created_at=current,
            assessment=assessment,
        )

    def request_cleanup(
        self,
        plan: CleanupPlan,
        *,
        submit: Callable[[CleanupPlan], Any] | None = None,
    ) -> CleanupPlan:
        """Create a separate cleanup job request without mutating a tree."""

        if not plan.ok:
            raise CleanupRefused(plan)
        if submit is None:
            # A preview has no durable authorization and must never be
            # accepted by execute_cleanup, even if the process is restarted.
            return plan
        if self.cleanup_authority is None:
            raise CleanupRefused(plan)
        submitted = submit(plan)
        job_id = getattr(submitted, "job_id", None)
        job_fence = getattr(submitted, "lease_fence", None) or getattr(
            submitted, "fence", None
        )
        if isinstance(submitted, Mapping):
            job_id = submitted.get("job_id") or submitted.get("id")
            job_fence = (
                submitted.get("lease_fence")
                or submitted.get("fence")
                or submitted.get("lease_token")
            )
        if not job_id or not job_fence:
            raise CleanupRefused(plan)
        return CleanupPlan(
            plan_id=plan.plan_id,
            lane_id=plan.lane_id,
            repository_path=plan.repository_path,
            branch=plan.branch,
            worktree_path=plan.worktree_path,
            owner_id=plan.owner_id,
            fence=plan.fence,
            created_at=plan.created_at,
            assessment=plan.assessment,
            guarded_remove=plan.guarded_remove,
            requested_job_id=str(job_id),
            requested_job_fence=str(job_fence),
        )

    def _cleanup_job_current(self, plan: CleanupPlan, lane: LaneRecord) -> bool:
        """Verify the separate cleanup WorkItem lease immediately before remove."""

        authority = self.cleanup_authority
        if authority is None or not plan.requested_job_id or not plan.requested_job_fence:
            return False
        checker = getattr(authority, "is_current", None)
        if checker is None:
            checker = getattr(authority, "revalidate", None)
        if not callable(checker):
            return False
        try:
            result = checker(
                plan.requested_job_id,
                lane_id=lane.lane_id,
                owner_id=plan.owner_id,
                lane_fence=plan.fence,
                job_fence=plan.requested_job_fence,
                plan_id=plan.plan_id,
            )
        except TypeError:
            try:
                result = checker(
                    plan.requested_job_id,
                    plan.lane_id,
                    plan.owner_id,
                    plan.fence,
                    plan.requested_job_fence,
                )
            except Exception:  # noqa: BLE001 - unknown lease state refuses remove
                return False
        except Exception:  # noqa: BLE001 - unknown lease state refuses remove
            return False
        return bool(result)

    def _cleanup_receipt(self, plan: CleanupPlan, lane: LaneRecord) -> bool:
        """Return true only for an exact durable removal receipt."""

        authority = self.cleanup_authority
        if authority is None or not plan.requested_job_id or not plan.requested_job_fence:
            return False
        getter = getattr(authority, "get_removal_receipt", None)
        if not callable(getter):
            getter = getattr(authority, "cleanup_receipt", None)
        if not callable(getter):
            return False
        try:
            receipt = getter(plan.requested_job_id, plan_id=plan.plan_id)
        except TypeError:
            try:
                receipt = getter(plan.requested_job_id, plan.plan_id)
            except Exception:  # noqa: BLE001 - unavailable receipt is not completion
                return False
        except Exception:  # noqa: BLE001 - unavailable receipt is not completion
            return False
        if not isinstance(receipt, Mapping):
            return False
        return (
            receipt.get("plan_id") == plan.plan_id
            and receipt.get("lane_id") == lane.lane_id
            and receipt.get("lane_fence") == plan.fence
            and receipt.get("job_fence") == plan.requested_job_fence
            and receipt.get("worktree_path") == lane.worktree_path
            and receipt.get("removed") is True
        )

    def _record_cleanup_complete(
        self,
        plan: CleanupPlan,
        lane: LaneRecord,
        result: Mapping[str, Any],
    ) -> bool:
        authority = self.cleanup_authority
        if authority is None:
            return False
        recorder = getattr(authority, "record_cleanup_complete", None)
        if not callable(recorder):
            return False
        payload = {
            "plan_id": plan.plan_id,
            "lane_id": lane.lane_id,
            "lane_fence": plan.fence,
            "job_id": plan.requested_job_id,
            "job_fence": plan.requested_job_fence,
            "repository_path": lane.repository_path,
            "branch": lane.branch,
            "worktree_path": lane.worktree_path,
            "removed": True,
            "result": dict(result),
        }
        try:
            return bool(recorder(payload))
        except TypeError:
            try:
                return bool(
                    recorder(
                        plan.requested_job_id,
                        plan_id=plan.plan_id,
                        lane_id=lane.lane_id,
                        lane_fence=plan.fence,
                        job_fence=plan.requested_job_fence,
                        worktree_path=lane.worktree_path,
                        result=result,
                    )
                )
            except Exception:  # noqa: BLE001 - durable receipt is required
                return False
        except Exception:  # noqa: BLE001 - durable receipt is required
            return False

    def _check_occupied(self, lane: LaneRecord) -> SafetyCheck:
        try:
            if self.occupancy_probe is not None:
                locked = self.occupancy_probe(lane)
            else:
                from repository_manager import prune_guard

                locked = prune_guard.worktree_is_locked(lane.worktree_path)
        except Exception:  # noqa: BLE001 - unknown occupancy refuses cleanup
            locked = None
        allowed = locked is False
        return SafetyCheck(
            "occupied",
            allowed,
            ""
            if allowed
            else (
                ReclamationReason.OCCUPIED.value
                if locked is True
                else ReclamationReason.OCCUPIED.value + ": occupancy evidence unavailable"
            ),
        )

    def _before_remove_checks(self, lane: LaneRecord) -> tuple[SafetyCheck, ...]:
        checks: list[SafetyCheck] = []
        checks.extend((self._check_process(lane),))
        checks.extend(self._check_git_state(lane))
        checks.extend(self._check_claims(lane))
        checks.append(self._check_anchor(lane))
        checks.append(self._check_occupied(lane))
        return tuple(checks)

    def execute_cleanup(self, plan: CleanupPlan, *, now: datetime | None = None) -> dict[str, Any]:
        """Re-check and remove through the existing guarded worktree adapter."""

        if not plan.executable:
            raise CleanupRefused(plan)
        current = self.registry.require(plan.lane_id)
        if current.owner_id != plan.owner_id or current.fence != plan.fence:
            raise ValueError(ReclamationReason.FENCE_STALE.value)
        if self._cleanup_receipt(plan, current):
            if current.state != LaneLifecycleState.QUARANTINED:
                try:
                    current = self.registry.quarantine(
                        current.lane_id,
                        owner_id=plan.owner_id,
                        fence=plan.fence,
                        reason="durable cleanup receipt reconciled",
                        now=now,
                    )
                except Exception as exc:  # noqa: BLE001 - do not claim completion
                    return {
                        "ok": False,
                        "lane_id": current.lane_id,
                        "reason": "cleanup receipt exists but quarantine was not durable",
                        "error": type(exc).__name__,
                    }
            return {
                "ok": True,
                "idempotent": True,
                "lane_id": current.lane_id,
                "receipt": True,
            }
        if current.state == LaneLifecycleState.QUARANTINED:
            if not self._cleanup_job_current(plan, current):
                raise CleanupRefused(plan)
            # Quarantine is a lifecycle state, not proof that the guarded
            # remove completed. A lane may be quarantined after an operator
            # action or an earlier failed cleanup, so only an exact durable
            # removal receipt can authorize an idempotent success above.
            return {
                "ok": False,
                "idempotent": False,
                "lane_id": current.lane_id,
                "reason": "quarantined lane lacks exact durable removal receipt",
                "reconciliation_pending": True,
                "removal_performed": False,
            }
        if current.state not in {
            LaneLifecycleState.ALLOCATING,
            LaneLifecycleState.ACTIVE,
            LaneLifecycleState.SUBMITTED,
            LaneLifecycleState.EXPIRED,
        }:
            raise CleanupRefused(plan)
        if not self._cleanup_job_current(plan, current):
            raise CleanupRefused(plan)
        fresh = self.assess(current, now=now)
        if not fresh.eligible:
            blocked = CleanupPlan(
                plan_id=plan.plan_id,
                lane_id=plan.lane_id,
                repository_path=plan.repository_path,
                branch=plan.branch,
                worktree_path=plan.worktree_path,
                owner_id=plan.owner_id,
                fence=plan.fence,
                created_at=plan.created_at,
                assessment=fresh,
                guarded_remove=plan.guarded_remove,
                requested_job_id=plan.requested_job_id,
                requested_job_fence=plan.requested_job_fence,
            )
            raise CleanupRefused(blocked)
        expired = self.registry.expire(
            current.lane_id,
            owner_id=plan.owner_id,
            fence=plan.fence,
            now=now,
        )
        # The expire transition is durable, but the lane/job/process/anchor
        # state can change in the gap before remove.  Re-read both authorities
        # and re-run every destructive guard immediately before delegation.
        expired = self.registry.require(plan.lane_id)
        if (
            expired.state != LaneLifecycleState.EXPIRED
            or expired.owner_id != plan.owner_id
            or expired.fence != plan.fence
            or not self._cleanup_job_current(plan, expired)
        ):
            raise CleanupRefused(plan)
        final_checks = self._before_remove_checks(expired)
        if not all(check.allowed for check in final_checks):
            blocked = CleanupPlan(
                plan_id=plan.plan_id,
                lane_id=plan.lane_id,
                repository_path=plan.repository_path,
                branch=plan.branch,
                worktree_path=plan.worktree_path,
                owner_id=plan.owner_id,
                fence=plan.fence,
                created_at=plan.created_at,
                assessment=ExpiryCandidate(expired, False, final_checks),
                guarded_remove=plan.guarded_remove,
                requested_job_id=plan.requested_job_id,
                requested_job_fence=plan.requested_job_fence,
            )
            raise CleanupRefused(blocked)
        if self.worktree_manager is None:
            return {
                "ok": False,
                "lane_id": expired.lane_id,
                "reason": "guarded worktree manager is unavailable",
            }
        result = self.worktree_manager.remove(
            expired.repository_path,
            expired.branch,
            force=False,
            delete_branch=True,
            base=expired.base_ref,
        )
        if not result.get("ok"):
            return {"ok": False, "lane_id": expired.lane_id, "result": result}
        anchors = list(expired.cleanup_anchors)
        if result.get("branch_anchor"):
            anchors.append(str(result["branch_anchor"]))
            try:
                self.registry.record_cleanup_anchor(
                    expired.lane_id,
                    str(result["branch_anchor"]),
                    owner_id=plan.owner_id,
                    fence=plan.fence,
                    now=now,
                )
            except Exception as exc:  # noqa: BLE001 - durable evidence is required
                return {
                    "ok": False,
                    "lane_id": expired.lane_id,
                    "reason": "cleanup anchor was not durably recorded",
                    "error": type(exc).__name__,
                }
        if not self._record_cleanup_complete(plan, expired, result):
            return {
                "ok": False,
                "lane_id": expired.lane_id,
                "reason": "guarded removal succeeded but durable cleanup receipt failed",
                "removal_performed": True,
                "result": result,
            }
        try:
            quarantined = self.registry.quarantine(
                expired.lane_id,
                owner_id=plan.owner_id,
                fence=plan.fence,
                reason="guarded worktree cleanup completed",
                now=now,
            )
        except Exception as exc:  # noqa: BLE001 - do not claim completion
            return {
                "ok": False,
                "lane_id": expired.lane_id,
                "reason": "durable quarantine transition failed after removal",
                "removal_performed": True,
                "error": type(exc).__name__,
                "result": result,
            }
        return {
            "ok": True,
            "lane_id": quarantined.lane_id,
            "idempotent": False,
            "result": result,
            "anchors": sorted(set(anchors)),
        }

    def reconcile(
        self,
        observed_worktrees: Iterable[Mapping[str, Any]],
    ) -> tuple[ReconciliationFinding, ...]:
        """Compare durable records with a read-only worktree listing."""

        observed = tuple(observed_worktrees)
        by_path = {str(item.get("path")): item for item in observed if item.get("path")}
        findings: list[ReconciliationFinding] = []
        matched: set[str] = set()
        for lane in self.registry.list_records():
            if lane.state == LaneLifecycleState.OBSERVED_LEGACY:
                findings.append(
                    ReconciliationFinding(
                        lane.lane_id,
                        ReconciliationClass.OBSERVED_LEGACY,
                        {"path": lane.worktree_path},
                        repair_required=False,
                    )
                )
                continue
            item = by_path.get(lane.worktree_path)
            if item is None:
                findings.append(
                    ReconciliationFinding(
                        lane.lane_id,
                        ReconciliationClass.MISSING_WORKTREE,
                        {"expected_path": lane.worktree_path},
                        repair_required=True,
                    )
                )
                continue
            matched.add(lane.worktree_path)
            if str(item.get("branch", "")) != lane.branch:
                classification = ReconciliationClass.BRANCH_MISMATCH
            elif not Path(lane.worktree_path).exists():
                classification = ReconciliationClass.PATH_MISMATCH
            elif lane.state not in {
                LaneLifecycleState.ALLOCATING,
                LaneLifecycleState.ACTIVE,
                LaneLifecycleState.SUBMITTED,
                LaneLifecycleState.EXPIRED,
            }:
                classification = ReconciliationClass.STATE_MISMATCH
            else:
                classification = ReconciliationClass.MANAGED
            findings.append(
                ReconciliationFinding(
                    lane.lane_id,
                    classification,
                    {"path": lane.worktree_path, "branch": item.get("branch")},
                    repair_required=classification != ReconciliationClass.MANAGED,
                )
            )
        for item in observed:
            path = str(item.get("path", ""))
            if path and path not in matched and not any(
                finding.details.get("path") == path for finding in findings
            ):
                findings.append(
                    ReconciliationFinding(
                        None,
                        ReconciliationClass.ORPHAN_WORKTREE,
                        {"path": path, "branch": item.get("branch")},
                        repair_required=False,
                    )
                )
        return tuple(findings)


LaneRegistryReconciler = LaneReclaimer


__all__ = [
    "CleanupPlan",
    "CleanupRefused",
    "ExpiryCandidate",
    "LaneReclaimer",
    "LaneRegistryReconciler",
    "ReconciliationClass",
    "ReconciliationFinding",
    "ReclamationReason",
    "SafetyCheck",
]
