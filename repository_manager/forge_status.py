"""Forge-abstracted CI-run status — is the tag that triggered a publish still
running, or did it already conclude (success or failure)?

CONCEPT:RM-DEP-READY ci-run-barrier

The gap this closes. ``dependency_readiness.await_gate_readiness`` (Layer 2 of
CONCEPT:RM-DEP-READY) decides a ``phased_push`` phase transition by retrying a
downstream repo's own pre-push gate — which, underneath, polls the package
index (PyPI Simple API) for the just-published version. That is the right
oracle for "is it installable right now", but it is blind to *why* a version
never shows up: a publish workflow that is still running looks identical, from
the index's point of view, to one that already failed outright — both report
"not there yet", so the barrier burns its entire ``wait_minutes`` retry
ceiling on both, even when the CI run that would have produced the artifact
already failed minutes ago and nothing further is coming. On 2026-08-12 that
same blind-sleep failure mode (a different gap in the same barrier — see
``dependency_readiness``'s own module docstring) let phase 3 push against an
unsatisfiable ``epistemic-graph`` constraint nothing detected; a CI run that
concludes failed is the earliest, cheapest signal available that the same
"wait and hope" pattern is about to happen again.

This module is the ONE thing that answers "what is the tag's release run
doing right now" — never duplicated per forge. ``ForgeBackend`` is a
one-method Protocol; :func:`await_gate_readiness` (Layer 2) calls it, when
given one, BEFORE it starts retrying each downstream repo's gate: a
``completed`` run with a non-success ``conclusion`` aborts immediately with
the run's own URL, instead of retrying an artifact that is never coming. A
``queued``/``in_progress`` run is polled (bounded by the SAME retry ceiling
Layer 2 already enforces — no second, parallel deadline). An ``unknown``
status — no client installed, the forge unreachable, the ref never ran,
anything this module cannot positively resolve — degrades to today's
behavior (proceed straight to index-polling) rather than blocking a push on
a signal this module could not obtain. Fail-closed on a CONFIRMED failure,
fail-open (degrade) on an UNKNOWN one: those are different problems and must
never be conflated into one bit.

Two forges, one contract, both real (never a stub):

* :class:`GitHubActionsBackend` — over ``github_agent``'s
  ``api_client_workflows.Api`` (``get_workflow_runs`` / ``get_workflow_run``).
  This is the live path for every ``agent-packages/*`` repo.
* :class:`GitLabPipelineBackend` — over ``gitlab_api``'s
  ``api.api_client_pipelines.GitLabApiPipelines`` (``get_pipelines`` /
  ``get_pipeline``), for the fleet's internal GitLab-hosted repos (see the
  workspace ``AGENTS.md``: GitHub is public/strict, GitLab is internal/lax —
  both still get a real CI-run barrier, not a GitHub-only one).

Both client imports are OPTIONAL, guarded with the fleet's standard
``try/except ImportError`` pattern (see e.g.
``repository_manager.prune_guard``'s guard on ``agent_utilities.governance
.lanes``): a repository-manager install without the ``github-agent``/
``gitlab-api`` extras must still import and run this module — every call
just degrades to ``state="unknown"`` with a loud ``FORGE_STATUS_UNAVAILABLE``
log line, never a silent skip and never a hard failure that would itself
become a new way to wedge the barrier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

__all__ = [
    "RunState",
    "RunStatus",
    "ForgeBackend",
    "GitHubActionsBackend",
    "GitLabPipelineBackend",
    "UnknownForgeBackend",
    "backend_for_remote",
    "owner_repo_from_remote",
]

# --------------------------------------------------------------------------- #
# Optional forge clients — guarded imports, the fleet's standard pattern
# (see repository_manager.prune_guard's guard on agent_utilities.governance
# .lanes): missing/unavailable degrades every call to state="unknown", never
# an ImportError propagating out of this module at import time or call time.
# --------------------------------------------------------------------------- #

try:
    from github_agent.api.api_client_workflows import Api as _GitHubWorkflowsApi
except ImportError as exc:  # pragma: no cover - exercised via the degrade path
    _GitHubWorkflowsApi = None  # type: ignore[assignment,misc]
    _GITHUB_AGENT_UNAVAILABLE = str(exc)
else:
    _GITHUB_AGENT_UNAVAILABLE = ""

try:
    from gitlab_api.api.api_client_pipelines import (
        GitLabApiPipelines as _GitLabPipelinesApi,
    )
except ImportError as exc:  # pragma: no cover - exercised via the degrade path
    _GitLabPipelinesApi = None  # type: ignore[assignment,misc]
    _GITLAB_API_UNAVAILABLE = str(exc)
else:
    _GITLAB_API_UNAVAILABLE = ""


RunState = Literal["queued", "in_progress", "completed", "unknown"]


@dataclass(frozen=True)
class RunStatus:
    """What one forge says about the latest CI run for one ref, right now.

    ``state="unknown"`` is the universal degrade target — a missing client,
    an unreachable forge, a ref with no run at all, or a response this
    backend could not positively interpret are all reported as ``"unknown"``,
    never as ``"completed"``/failed, since claiming a conclusion this module
    does not actually have would fail closed on a signal that was never
    really obtained (the exact inversion of what this module exists to add).
    """

    state: RunState
    conclusion: str | None
    url: str | None
    started_at: str | None


_UNKNOWN = RunStatus(state="unknown", conclusion=None, url=None, started_at=None)


class ForgeBackend(Protocol):
    """Pluggable CI-forge backend (CONCEPT:RM-DEP-READY ci-run-barrier).

    One method: the latest run for a ref. Never raises — every failure mode
    (no client, unreachable forge, malformed response, ref never ran) is
    represented as :data:`RunStatus` with ``state="unknown"``, so a caller
    never needs a second try/except around this call.
    """

    def latest_run_for_ref(self, owner: str, repo: str, ref: str) -> RunStatus:
        """The most recent CI run triggered by ``ref`` in ``owner/repo``."""
        ...


# --------------------------------------------------------------------------- #
# GitHub Actions
# --------------------------------------------------------------------------- #

#: GitHub Actions' own run-status vocabulary that this module trusts as-is
#: (``docs.github.com`` "About status checks"). Anything else observed on a
#: response is reported as "unknown" rather than guessed at.
_GITHUB_RUN_STATES = frozenset({"queued", "in_progress", "completed"})


class GitHubActionsBackend:
    """:class:`ForgeBackend` over ``github_agent.api.api_client_workflows.Api``.

    The live path for every ``agent-packages/*`` repo (GitHub is this
    fleet's public, strict-standards forge — see the workspace ``AGENTS.md``
    "GitHub public vs GitLab internal standards" note). Accepts a
    pre-constructed client (the normal path when a caller already holds an
    authenticated ``github_agent`` client and wants it reused rather than a
    second one built here) or builds its own from ``url``/``token``.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        url: str = "https://api.github.com",
        token: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        if _GitHubWorkflowsApi is None:
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: github_agent is not installed "
                "(%s) -- every GitHubActionsBackend.latest_run_for_ref call "
                "will degrade to state='unknown'",
                _GITHUB_AGENT_UNAVAILABLE,
            )
            self._client = None
            return
        self._client = _GitHubWorkflowsApi(url=url, token=token)

    def latest_run_for_ref(self, owner: str, repo: str, ref: str) -> RunStatus:
        if self._client is None:
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: no github_agent client available "
                "for %s/%s@%s -- degrading to state='unknown'",
                owner,
                repo,
                ref,
            )
            return _UNKNOWN
        try:
            response = self._client.get_workflow_runs(
                owner=owner, repo=repo, branch=ref
            )
            runs = response.data or []
        except Exception as exc:  # noqa: BLE001 - a forge outage must never raise
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: could not list workflow runs for "
                "%s/%s@%s: %s: %s -- degrading to state='unknown'",
                owner,
                repo,
                ref,
                type(exc).__name__,
                exc,
            )
            return _UNKNOWN
        if not runs:
            logger.info(
                "forge_status: %s/%s@%s has no workflow runs yet -- state='unknown'",
                owner,
                repo,
                ref,
            )
            return _UNKNOWN

        run = runs[0]  # GitHub returns runs newest-first
        status = getattr(run, "status", None)
        state: RunState = status if status in _GITHUB_RUN_STATES else "unknown"
        return RunStatus(
            state=state,
            conclusion=getattr(run, "conclusion", None),
            url=getattr(run, "html_url", None),
            started_at=getattr(run, "run_started_at", None)
            or getattr(run, "created_at", None),
        )


# --------------------------------------------------------------------------- #
# GitLab CI pipelines
# --------------------------------------------------------------------------- #

#: GitLab's own pipeline-status vocabulary (GitLab REST API "Pipelines"
#: reference) mapped onto the smaller :data:`RunState` this module exposes.
#: Terminal states (a pipeline that will not transition further) map to
#: "completed" with the terminal status itself carried as the conclusion;
#: everything still moving maps to "queued"/"in_progress".
_GITLAB_TERMINAL_STATUSES = frozenset({"success", "failed", "canceled", "skipped"})
_GITLAB_RUNNING_STATUSES = frozenset({"running"})


class GitLabPipelineBackend:
    """:class:`ForgeBackend` over ``gitlab_api``'s
    ``api.api_client_pipelines.GitLabApiPipelines`` — the fleet's internal
    GitLab-hosted repos (GitLab is this fleet's internal, lax-standards
    forge; it still gets a REAL CI-run barrier, not a stub, since an internal
    repo's publish job can fail exactly like a public one's).

    ``owner``/``repo`` are joined and percent-encoded as GitLab's own
    ``namespace%2Fproject`` path-based project id (GitLab API convention for
    any ``/`` in a project path, including nested subgroups folded into
    ``owner``), so no numeric project id needs to be resolved up front.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        url: str | None = None,
        token: str | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        if _GitLabPipelinesApi is None:
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: gitlab_api is not installed (%s) "
                "-- every GitLabPipelineBackend.latest_run_for_ref call will "
                "degrade to state='unknown'",
                _GITLAB_API_UNAVAILABLE,
            )
            self._client = None
            return
        if not url:
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: no GitLab instance url configured "
                "-- every GitLabPipelineBackend.latest_run_for_ref call will "
                "degrade to state='unknown'"
            )
            self._client = None
            return
        self._client = _GitLabPipelinesApi(url=url, token=token)

    def latest_run_for_ref(self, owner: str, repo: str, ref: str) -> RunStatus:
        if self._client is None:
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: no gitlab_api client available "
                "for %s/%s@%s -- degrading to state='unknown'",
                owner,
                repo,
                ref,
            )
            return _UNKNOWN
        project_id = quote(f"{owner}/{repo}", safe="")
        try:
            response = self._client.get_pipelines(project_id=project_id, ref=ref)
            pipelines = response.data or []
        except Exception as exc:  # noqa: BLE001 - a forge outage must never raise
            logger.warning(
                "FORGE_STATUS_UNAVAILABLE: could not list pipelines for "
                "%s/%s@%s: %s: %s -- degrading to state='unknown'",
                owner,
                repo,
                ref,
                type(exc).__name__,
                exc,
            )
            return _UNKNOWN
        if not pipelines:
            logger.info(
                "forge_status: %s/%s@%s has no pipelines yet -- state='unknown'",
                owner,
                repo,
                ref,
            )
            return _UNKNOWN

        pipeline = pipelines[0]  # GitLab returns pipelines newest-first
        status = getattr(pipeline, "status", None)
        if status in _GITLAB_TERMINAL_STATUSES:
            state: RunState = "completed"
            conclusion = status
        elif status in _GITLAB_RUNNING_STATUSES:
            state = "in_progress"
            conclusion = None
        elif status:
            # created / waiting_for_resource / preparing / pending / manual /
            # scheduled -- not yet running, but not unknown either.
            state = "queued"
            conclusion = None
        else:
            return _UNKNOWN
        return RunStatus(
            state=state,
            conclusion=conclusion,
            url=getattr(pipeline, "web_url", None),
            started_at=getattr(pipeline, "started_at", None)
            or getattr(pipeline, "created_at", None),
        )


# --------------------------------------------------------------------------- #
# No-op backend for a forge this module does not recognize -- degrade, never
# raise, so an unrecognized remote host is exactly as safe as a missing
# client (both are "this module could not obtain the signal").
# --------------------------------------------------------------------------- #


class UnknownForgeBackend:
    """:class:`ForgeBackend` for a remote host that is neither GitHub nor a
    configured GitLab instance -- always ``state='unknown'``, loudly logged
    once per call, never a silent skip."""

    def __init__(self, *, host: str = "") -> None:
        self._host = host

    def latest_run_for_ref(self, owner: str, repo: str, ref: str) -> RunStatus:
        logger.warning(
            "FORGE_STATUS_UNAVAILABLE: %r is not a recognized forge host for "
            "%s/%s@%s -- degrading to state='unknown'",
            self._host,
            owner,
            repo,
            ref,
        )
        return _UNKNOWN


# --------------------------------------------------------------------------- #
# Backend selection -- from the git remote URL host, never hardcoded.
# --------------------------------------------------------------------------- #


def owner_repo_from_remote(remote_url: str) -> tuple[str, str] | None:
    """``(owner, repo)`` parsed from a git remote URL, or ``None`` when the
    URL has fewer than two path segments to parse.

    Handles both ``https://host/owner/repo.git`` and the SSH shorthand
    ``git@host:owner/repo.git`` (``urlparse`` alone does not parse the SSH
    form, since it has no ``scheme://`` prefix). For a GitLab remote with
    nested subgroups (``https://host/group/subgroup/repo.git``), every
    segment except the last is folded into ``owner`` (joined with ``/``) --
    the same shape :class:`GitLabPipelineBackend` re-encodes as GitLab's own
    ``namespace%2Fproject`` path-based project id.
    """
    url = remote_url.strip()
    if not url:
        return None
    if "://" not in url and "@" in url and ":" in url:
        # git@host:owner/repo.git -> host/owner/repo.git
        _, _, rest = url.partition("@")
        host, _, path = rest.partition(":")
        url = f"ssh://{host}/{path}"

    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2:
        return None
    owner = "/".join(segments[:-1])
    repo = segments[-1]
    return owner, repo


def _host_from_remote(remote_url: str) -> str:
    url = remote_url.strip()
    if "://" not in url and "@" in url and ":" in url:
        _, _, rest = url.partition("@")
        host, _, _ = rest.partition(":")
        return host.lower()
    return (urlparse(url).hostname or "").lower()


def backend_for_remote(
    remote_url: str,
    *,
    github_kwargs: dict[str, Any] | None = None,
    gitlab_kwargs: dict[str, Any] | None = None,
) -> ForgeBackend:
    """Select the :class:`ForgeBackend` implementation for ``remote_url``'s
    host -- ``github.com`` gets :class:`GitHubActionsBackend`; any other
    host is treated as a (possibly self-hosted, internal-LAN) GitLab
    instance and gets :class:`GitLabPipelineBackend` pointed at that same
    host, matching the "GitHub public vs GitLab internal standards" split
    this fleet already operates under (GitHub is the ONE public forge this
    fleet uses; every other host it pushes to is one of its own GitLab
    instances). A URL this function cannot parse at all falls back to
    :class:`UnknownForgeBackend` -- degrade, never raise.
    """
    host = _host_from_remote(remote_url)
    if not host:
        return UnknownForgeBackend(host=host)
    if host == "github.com" or host.endswith(".github.com"):
        return GitHubActionsBackend(**(github_kwargs or {}))
    gitlab_kwargs = dict(gitlab_kwargs or {})
    gitlab_kwargs.setdefault("url", f"https://{host}")
    return GitLabPipelineBackend(**gitlab_kwargs)
