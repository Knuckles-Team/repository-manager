"""Dependency-readiness gate — is every declared intra-fleet constraint installable?

CONCEPT:RM-DEP-READY

The problem this closes, with a live instance. ``phased_push`` (CONCEPT:RM-PUSH)
pushes the fleet in dependency order: e.g. phase 2 publishes ``epistemic-graph``,
then a blind ``wait_minutes: 30`` sleep, then phase 3 pushes ``agent-utilities``
(whose ``pyproject.toml`` declares ``epistemic-graph[full]>=2.23.2,<3.0.0``). The
sleep is a **guess**: slow when the publish takes 4 minutes, and silently WRONG
when the publish never lands — the wave marches on and pushes every dependent
against a stale/unsatisfiable dependency. On 2026-08-12 that gap was not
hypothetical: PyPI held only ``epistemic-graph==2.23.0`` while
``agent-utilities`` required ``>=2.23.2`` — an unsatisfiable constraint nothing
in this fleet detected.

This module is the ONE artifact-availability primitive both layers of the fix
share (never two parallel implementations of "is this constraint satisfiable"):

* **Layer 1 — repo-level pre-push hook.** ``check_tree`` reads *one repo's own*
  declared dependencies (``pyproject.toml``), keeps only the ones naming another
  fleet package, and checks each declared constraint against the configured
  index. Wired into ``.pre-commit-config.yaml`` as a ``[pre-push, manual]``
  local hook, so it runs through the SAME machinery as every other heavy gate
  (:func:`repository_manager.gates.run_gate_stage` / ``_gate_before_push`` /
  the ``rm_gates`` MCP tool) — no second gate runner.
* **Layer 2 — the ``phased_push`` wave barrier.** :func:`await_gate_readiness`
  decides a phase transition by RUNNING each downstream repo's own pre-push
  gate (:func:`repository_manager.gates.run_gate_stage`, the exact call a real
  ``git push`` on that repo would make) with retry/backoff up to a
  ``wait_minutes`` ceiling — never a second, parallel "ask the index myself"
  implementation. Layer 1's hook already fails closed on an unsatisfiable
  constraint; Layer 2's only job is retry/backoff/deadline orchestration
  around calling it, so there is exactly one mechanism that decides both
  "is this phase transition ready" and "will this repo's own push succeed" —
  they are the same call. Supersedes an earlier poll-the-index-directly
  ``await_constraints`` design (removed): that duplicated Layer 1's own check
  in a second implementation, and could report "ready" while the actual gate
  a push would run still failed for a reason the poll didn't know about.

Design (CONCEPT:RM-DEP-READY pluggable-backend). ``IndexBackend`` is a two-method
protocol; :class:`PyPISimpleIndexBackend` is the default implementation over the
PEP 503/691 Simple Repository API — the SAME protocol pip/uv themselves resolve
against, so it works unmodified against pypi.org, a GitLab PyPI package-registry
proxy, devpi, or any other PEP 503-compliant private index without any
``pypi.org``-specific code. A container-registry or GitLab generic-package
backend for other artifact kinds is a second class implementing the same
protocol — nothing else here or in ``phased_push`` changes.

Fail-closed, with one loud, audited escape hatch (never a silent skip). An
index that cannot be reached is reported distinctly from a package the index
has genuinely never heard of, which is distinct again from "reachable, package
known, but no published version satisfies the constraint" — three different
problems needing three different remediations. Setting
``RM_DEPENDENCY_READINESS_OVERRIDE_REASON`` to a non-empty human reason bypasses
a failing check but never silently: the override is printed in a loud banner
and appended to an audit log (:func:`_record_override`) every time it fires, so
pushing a fix to a repo whose own dependency is temporarily unpublished is never
permanently blocked, and the bypass is never invisible.

Three further gaps, all closed in Layer 2 (:func:`await_gate_readiness`), the
owner reports as still causing later phases to fail because they install an
earlier phase's package before it is actually on PyPI:

* **Targets cross-check** (:func:`cross_check_targets`). The barrier only
  gate-checks the ``targets`` its caller computed — normally a narrowed slice
  (e.g. ``phased_push``'s ``later_phases``). If that narrowing misses a repo
  that genuinely names the just-published package, the phase "succeeds"
  because *nothing the barrier was told about* declared a constraint — not
  because nothing actually depends on it. :func:`cross_check_targets`
  independently rescans a caller-supplied ``candidate_repos`` universe (never
  reusing the caller's own narrowed list) for every repo whose OWN
  ``pyproject.toml`` names one of the published packages, and reports any
  that ``targets`` omitted. A hit there is a hard abort with a distinct
  ``TARGETS_INCOMPLETE`` reason — "something did depend on it and the barrier
  didn't know" is the real failure "nothing declared a constraint" was hiding.
* **Partial-publish detection** (:class:`AvailableVersions`'s
  ``partial_versions``, :data:`CheckStatus`'s ``"partial_publish"``). A
  publish workflow uploads a wheel and an sdist as two separate requests; a
  version whose Simple-API listing shows only one of the two is mid-publish,
  not absent and not genuinely unsatisfiable — reporting it as plain
  ``"unsatisfied"`` (today's behavior before this addition) is indistinguishable
  from "will never be satisfiable", which is exactly the confusion the
  2026-08-12 incident above turned on. ``"partial_publish"`` is reported as
  its own :data:`CheckStatus` (still a *failure* for gating purposes — a
  half-published release is not installable yet either) so the retry loop's
  own logging can tell an operator "this is still uploading" apart from "this
  was never going to work".
* **CI-run barrier** (:mod:`repository_manager.forge_status`). Before
  retrying a downstream repo's gate at all, :func:`await_gate_readiness` can
  ask the publishing repo's own forge (GitHub Actions / GitLab CI, via
  :func:`repository_manager.forge_status.backend_for_remote`) whether the
  release run for the published tag already concluded. A run that concluded
  with a non-success ``conclusion`` aborts immediately, with the run's own
  URL, instead of burning the full ``wait_minutes`` retry ceiling polling an
  index for an artifact a failed CI run already proved is never coming. An
  ``"unknown"`` forge status (no client installed, forge unreachable, ref
  never ran) degrades to today's behavior — proceed straight to index-polling
  — never blocks on a signal this module could not actually obtain.

Also: when yanked versions were excluded from a match that WOULD otherwise
have satisfied the specifier, :func:`check_constraint` reports them in
``yanked_but_would_match`` and folds them into the ``"unsatisfied"`` detail
line, so that verdict is explicable rather than a bare "nothing available".
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import requests  # transitively pinned by agent-utilities>=2.0.0 (requests>=2.34.2)
import yaml  # type: ignore[import-untyped]
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

# repository_manager.forge_status is a sibling module in this same package --
# not a second optional-import guard here (it has no hard external deps of
# its own; ITS optional forge clients are guarded inside it).
from repository_manager import forge_status

logger = logging.getLogger(__name__)

__all__ = [
    "OVERRIDE_ENV_VAR",
    "DEFAULT_INDEX_URL",
    "HOOK_ID",
    "NON_PACKAGE_SUBDIRECTORY_KEYS",
    "AvailableVersions",
    "ConstraintCheck",
    "ReadinessReport",
    "GateCheckFailure",
    "GateReadinessOutcome",
    "IndexBackend",
    "IndexQueryError",
    "IndexUnreachableError",
    "PackageUnknownError",
    "PyPISimpleIndexBackend",
    "resolve_index_urls",
    "fleet_package_names",
    "declared_fleet_constraints",
    "hook_declared",
    "check_constraint",
    "check_tree",
    "cross_check_targets",
    "await_gate_readiness",
    "dispatch",
    "main",
]

#: The PEP 503/691 Simple Repository API root every unconfigured repo resolves
#: against — pip/uv's own default. Never assumed when a repo configures its own.
DEFAULT_INDEX_URL = "https://pypi.org/simple"

_SIMPLE_JSON_ACCEPT = "application/vnd.pypi.simple.v1+json"

#: The one explicit, loud, auditable escape hatch (never a silent `--no-verify`
#: substitute). Value is the human-readable reason and must be non-empty.
OVERRIDE_ENV_VAR = "RM_DEPENDENCY_READINESS_OVERRIDE_REASON"

#: The pre-commit hook id Layer 1 registers itself under (see
#: ``.pre-commit-config.yaml`` and ``scripts/sweep_dependency_readiness_hook.py``).
#: Layer 2 (:func:`await_gate_readiness`) scopes its ``run_gate_stage`` calls to
#: exactly this id, and :func:`hook_declared` probes for it, so the hook's own
#: identity is declared in exactly one place.
HOOK_ID = "dependency-readiness"

#: Files a repo's own dependencies are declared in, checked in order.
_PYPROJECT_NAME = "pyproject.toml"


# --------------------------------------------------------------------------- #
# Errors — distinguish "index unreachable" from "index reached, package/version
# absent" so the printed message always names the right remediation.
# --------------------------------------------------------------------------- #


class IndexQueryError(Exception):
    """Base class for a failed availability query."""


class IndexUnreachableError(IndexQueryError):
    """Network/DNS/TLS/timeout/5xx/malformed-response — the index itself could
    not be queried, so nothing can be said about the package. Remediation:
    check connectivity / index configuration / retry."""


class PackageUnknownError(IndexQueryError):
    """The index was reached and has no record of this package name (HTTP
    404). Remediation: check the package name / whether it was ever published
    to *this* index."""


# --------------------------------------------------------------------------- #
# The pluggable artifact-availability predicate.
# --------------------------------------------------------------------------- #


@dataclass
class AvailableVersions:
    """What one index says is installable for one package, right now.

    ``versions``/``latest`` are FULLY published releases only — both a wheel
    AND an sdist present, non-yanked (see :func:`_version_file_summary`). Two
    further buckets distinguish the reasons a version is NOT in ``versions``,
    each needing a different remediation:

    * ``partial_versions`` — at least one non-yanked file exists, but not
      both a wheel and an sdist. A real publish workflow uploads the two as
      separate requests; a version caught mid-upload looks, to a naive
      "does the version string appear at all" check, identical to a fully
      published one — reported as ``"satisfied"`` when it may still be
      seconds away from having its other half land, or identical to
      "unsatisfied" when actually a live in-flight publish, not a failure.
      Neither is correct; :func:`check_constraint` reports this bucket as
      its own ``"partial_publish"`` :data:`CheckStatus`.
    * ``yanked_versions`` — every file for the version is yanked (PEP 592).
      Correctly excluded from ``versions``, but worth surfacing by name when
      the version would otherwise have satisfied a constraint (see
      :func:`check_constraint`'s ``yanked_but_would_match``), so an
      ``"unsatisfied"`` verdict is explicable rather than a bare "nothing
      available".
    """

    package: str
    index_url: str
    versions: list[str] = field(default_factory=list)  # sorted ascending
    latest: str | None = None
    partial_versions: list[str] = field(default_factory=list)  # sorted ascending
    yanked_versions: list[str] = field(default_factory=list)  # sorted ascending


class IndexBackend(Protocol):
    """Pluggable artifact-index backend (CONCEPT:RM-DEP-READY pluggable-backend).

    PyPI is one implementation, not the interface: a container registry or the
    GitLab generic/package registry implements the same two-method contract.
    """

    def available_versions(
        self, package: str, *, index_url: str, timeout: float
    ) -> AvailableVersions:
        """Return every version of ``package`` installable from ``index_url``.

        Raises :class:`IndexUnreachableError` when the index itself could not
        be queried, or :class:`PackageUnknownError` when it was reached but
        has no record of ``package``. Never returns an empty
        :class:`AvailableVersions` to mean either of those — an empty
        ``versions`` list means "index reached, package known, zero
        installable releases" (e.g. every release yanked).
        """
        ...


def _version_from_filename(filename: str) -> tuple[str, bool] | None:
    """``(version, is_wheel)`` parsed from one Simple-API filename, or
    ``None`` when the filename is not a wheel/sdist this parser recognizes
    (e.g. a legacy ``.egg``) — such files are skipped, never mis-attributed.
    """
    try:
        _, version = parse_sdist_filename(filename)
        return str(version), False
    except InvalidSdistFilename:
        pass
    try:
        _, version, _, _ = parse_wheel_filename(filename)
        return str(version), True
    except InvalidWheelFilename:
        return None


@dataclass
class _VersionFiles:
    """Per-version file-kind/yanked summary — see :func:`_version_file_summary`."""

    has_wheel: bool = False
    has_sdist: bool = False
    saw_file: bool = False
    saw_non_yanked: bool = False

    @property
    def fully_installable(self) -> bool:
        return self.has_wheel and self.has_sdist and self.saw_non_yanked

    @property
    def partial(self) -> bool:
        """At least one non-yanked file, but not both a wheel and an sdist —
        an in-flight (or permanently incomplete) publish, distinct from both
        "fully installable" and "nothing here"."""
        return self.saw_non_yanked and not self.fully_installable

    @property
    def all_yanked(self) -> bool:
        return self.saw_file and not self.saw_non_yanked


def _version_file_summary(data: dict[str, Any]) -> dict[str, _VersionFiles]:
    """Per-version wheel/sdist/yanked summary parsed from one Simple-API
    ``files`` list — extends the file-kind-blind, yanked-only version this
    replaced (the previous ``_installable_versions``, which asked only "does
    ANY non-yanked file exist for this version") with wheel-vs-sdist tracking
    in the SAME single pass over ``files``, so a caller can additionally tell
    "one file type published, the other still uploading" (:attr:`_VersionFiles
    .partial` — a live, in-flight publish) apart from "fully published"
    (:attr:`_VersionFiles.fully_installable`) and "nothing published, or
    every file yanked" (:attr:`_VersionFiles.all_yanked`).

    Per-file ``yanked`` status (PEP 592) still wins over anything else: a
    yanked wheel/sdist never counts toward either bucket. Returns an empty
    dict when ``files`` parses to nothing usable (e.g. a minimal PEP 691
    response with no ``files`` key) — :func:`PyPISimpleIndexBackend
    .available_versions` falls back to the top-level ``versions`` field in
    that case, exactly as the function this replaced did, so a working index
    is never reported as empty just because this parser couldn't read
    filenames.
    """
    per_version: dict[str, _VersionFiles] = {}
    for entry in data.get("files", []) or []:
        filename = entry.get("filename", "")
        parsed = _version_from_filename(filename)
        if parsed is None:
            continue
        version, is_wheel = parsed
        bucket = per_version.setdefault(version, _VersionFiles())
        bucket.saw_file = True
        if bool(entry.get("yanked")):
            continue
        bucket.saw_non_yanked = True
        if is_wheel:
            bucket.has_wheel = True
        else:
            bucket.has_sdist = True
    return per_version


def _fetch_simple_json(
    package: str, index_url: str, url: str, timeout: float
) -> dict[str, Any]:
    try:
        resp = requests.get(
            url, timeout=timeout, headers={"Accept": _SIMPLE_JSON_ACCEPT}
        )
    except requests.RequestException as exc:
        raise IndexUnreachableError(
            f"could not reach index {index_url!r} for package {package!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if resp.status_code == 404:
        raise PackageUnknownError(
            f"index {index_url!r} has no record of package {package!r} "
            f"(HTTP 404 at {url})"
        )
    if resp.status_code == 429 or resp.status_code >= 500:
        raise IndexUnreachableError(
            f"index {index_url!r} returned HTTP {resp.status_code} for "
            f"{package!r} (server/rate-limit error, not a version-absent verdict)"
        )
    if resp.status_code != 200:
        raise IndexUnreachableError(
            f"index {index_url!r} returned unexpected HTTP {resp.status_code} "
            f"for {package!r}"
        )

    try:
        return resp.json()
    except ValueError as exc:
        raise IndexUnreachableError(
            f"index {index_url!r} returned a non-JSON Simple-API response "
            f"for {package!r}: {exc}"
        ) from exc


def _available_versions_from_fallback(
    data: dict[str, Any], package: str, index_url: str
) -> AvailableVersions:
    # No `files` to parse (e.g. a minimal PEP 691 response) -- fall back to
    # the top-level `versions` field, same as the function this replaced
    # always did. Wheel/sdist completeness cannot be determined from this
    # field alone, so every listed version is treated as fully installable
    # rather than silently reported as empty -- a working index must never
    # look like a dead one.
    fallback: list[Version] = []
    for raw in data.get("versions") or []:
        try:
            fallback.append(Version(raw))
        except InvalidVersion:
            continue
    fallback.sort()
    return AvailableVersions(
        package=package,
        index_url=index_url,
        versions=[str(v) for v in fallback],
        latest=str(fallback[-1]) if fallback else None,
    )


def _available_versions_from_summary(
    summary: dict[str, _VersionFiles], package: str, index_url: str
) -> AvailableVersions:
    full: list[Version] = []
    partial: list[Version] = []
    yanked: list[Version] = []
    for raw, info in summary.items():
        try:
            parsed_version = Version(raw)
        except InvalidVersion:
            continue
        if info.fully_installable:
            full.append(parsed_version)
        elif info.partial:
            partial.append(parsed_version)
        elif info.all_yanked:
            yanked.append(parsed_version)
    full.sort()
    partial.sort()
    yanked.sort()
    return AvailableVersions(
        package=package,
        index_url=index_url,
        versions=[str(v) for v in full],
        latest=str(full[-1]) if full else None,
        partial_versions=[str(v) for v in partial],
        yanked_versions=[str(v) for v in yanked],
    )


class PyPISimpleIndexBackend:
    """Default :class:`IndexBackend` — the PEP 503/691 Simple Repository API.

    This is the protocol pip/uv themselves speak for ``--index-url`` /
    ``--extra-index-url`` resolution, so it needs no PyPI-specific handling to
    also work against a GitLab PyPI package-registry proxy, devpi, or any other
    compliant private index — only the base URL differs, and that is resolved
    by :func:`resolve_index_urls`, never hardcoded here.
    """

    def available_versions(
        self, package: str, *, index_url: str, timeout: float = 10.0
    ) -> AvailableVersions:
        name = canonicalize_name(package)
        url = f"{index_url.rstrip('/')}/{name}/"
        data = _fetch_simple_json(package, index_url, url, timeout)
        summary = _version_file_summary(data)
        if not summary:
            return _available_versions_from_fallback(data, package, index_url)
        return _available_versions_from_summary(summary, package, index_url)


# --------------------------------------------------------------------------- #
# Index resolution — honor the index the repo actually resolves against,
# never a hardcoded pypi.org.
# --------------------------------------------------------------------------- #


def _add_index_url(urls: list[str], value: str | None) -> None:
    if value and value not in urls:
        urls.append(value)


def _collect_env_index_urls(urls: list[str]) -> None:
    _add_index_url(urls, os.environ.get("UV_INDEX_URL"))
    for extra in (os.environ.get("UV_EXTRA_INDEX_URL") or "").split():
        _add_index_url(urls, extra)
    _add_index_url(urls, os.environ.get("PIP_INDEX_URL"))
    for extra in (os.environ.get("PIP_EXTRA_INDEX_URL") or "").split():
        _add_index_url(urls, extra)


def _load_pyproject_uv_config(repo_path: str | Path) -> dict[str, Any]:
    pyproject = Path(repo_path) / _PYPROJECT_NAME
    if not pyproject.exists():
        return {}
    try:
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return (data.get("tool") or {}).get("uv") or {}


def _collect_pyproject_index_urls(urls: list[str], repo_path: str | Path) -> None:
    uv_cfg = _load_pyproject_uv_config(repo_path)
    index_entries = uv_cfg.get("index") or []
    default_entries = [e for e in index_entries if e.get("default")]
    other_entries = [e for e in index_entries if not e.get("default")]
    for entry in (*default_entries, *other_entries):
        _add_index_url(urls, entry.get("url"))
    _add_index_url(urls, uv_cfg.get("index-url"))
    for extra in uv_cfg.get("extra-index-url") or []:
        _add_index_url(urls, extra)


def resolve_index_urls(repo_path: str | Path) -> list[str]:
    """The index URL(s) ``repo_path`` actually resolves dependencies against.

    Precedence (highest first), matching uv's/pip's own resolution order:

    1. ``UV_INDEX_URL`` / ``UV_EXTRA_INDEX_URL`` env vars (space-separated for
       the extra var).
    2. ``PIP_INDEX_URL`` / ``PIP_EXTRA_INDEX_URL`` env vars.
    3. ``[[tool.uv.index]]`` entries in the repo's own ``pyproject.toml`` — the
       entry marked ``default = true`` first, then declaration order.
    4. ``[tool.uv] index-url`` / ``extra-index-url`` (the older single-key form).
    5. :data:`DEFAULT_INDEX_URL` (pypi.org) — the true fallback, used only when
       nothing above configures anything, never assumed up front.

    Always returns at least one URL (the default). Order is resolution order:
    callers should stop at the first index reporting the package known.
    """
    urls: list[str] = []
    _collect_env_index_urls(urls)
    _collect_pyproject_index_urls(urls, repo_path)
    _add_index_url(urls, DEFAULT_INDEX_URL)
    return urls


# --------------------------------------------------------------------------- #
# Fleet scope — which declared dependency names are THIS fleet, derived from
# the workspace manifest, never a hardcoded au/eg pair.
# --------------------------------------------------------------------------- #


#: Top-level ``subdirectories`` keys the canonical manifest itself documents as
#: NOT PyPI-publishable Python packages — container/base-image build
#: definitions and containerized service stacks (see the workspace root
#: ``AGENTS.md`` navigation table: "``images/`` — 55 base-image build
#: definitions", "``services/`` — 141 containerized service stacks"). This is
#: the manifest's OWN structural taxonomy (package tree vs. infra trees), not
#: a hardcoded package name or pair — it is what stops an infra repo (e.g. a
#: Docker Swarm stack named identically to an unrelated third-party PyPI
#: package, such as a ``services/langfuse`` compose stack colliding with the
#: third-party ``langfuse`` SDK) from being mistaken for a fleet-published
#: package this gate should hold to a version constraint.
#:
#: Public (not a leading-underscore private) because a second call site reuses
#: it for the exact same distinction: ``Git.phased_push``'s ``bulk_push``
#: phases (CONCEPT:RM-PUSH) resolve against the WHOLE workspace manifest's
#: project map, which also contains every ``images/``/``services/`` repo —
#: this is the one shared definition of "package tree vs. infra tree" both
#: call sites reuse rather than each guessing its own.
NON_PACKAGE_SUBDIRECTORY_KEYS = frozenset({"images", "services"})


def _repo_name_from_url(url: str) -> str | None:
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[: -len(".git")]
    return name or None


def _collect_manifest_repo_names(node: dict[str, Any], names: set[str]) -> None:
    for repo in node.get("repositories", []) or []:
        url = repo.get("url") if isinstance(repo, dict) else None
        if not url:
            continue
        name = _repo_name_from_url(url)
        if name:
            names.add(canonicalize_name(name))


def _walk_manifest_repo_names(
    node: Any, names: set[str], *, skip_keys: frozenset[str] = frozenset()
) -> None:
    if not isinstance(node, dict):
        return
    _collect_manifest_repo_names(node, names)
    for key, sub in (node.get("subdirectories") or {}).items():
        if key in skip_keys:
            continue
        _walk_manifest_repo_names(sub, names, skip_keys=skip_keys)


def fleet_package_names(workspace_yml_path: str | Path) -> set[str]:
    """Every package name this fleet's canonical manifest declares.

    Only the URL's trailing path segment is used (never the ``${ENV_VAR}``
    host/path prefix), so this works unmodified inside a pre-push git hook's
    minimal environment, which need not have ``AGENT_UTILITIES_REPO_ORIGIN`` or
    similar set. Also folds in ``maintenance.phases[].project(s)`` names, which
    is how ``phased_push`` itself names what each phase publishes — those are
    always trusted regardless of which subdirectory they live under, since
    being in the phased publish plan is itself proof of Python-package-ness.

    Skips :data:`NON_PACKAGE_SUBDIRECTORY_KEYS` (``images``/``services``) when
    walking ``subdirectories`` so an infra/deploy repo never collides with an
    unrelated third-party PyPI package of the same name.
    """
    path = Path(workspace_yml_path)
    if not path.exists():
        return set()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return set()

    names: set[str] = set()
    _walk_manifest_repo_names(data, names, skip_keys=NON_PACKAGE_SUBDIRECTORY_KEYS)
    for phase in ((data.get("maintenance") or {}).get("phases")) or []:
        for proj in phase.get("projects", []) or []:
            names.add(canonicalize_name(proj))
        if phase.get("project"):
            names.add(canonicalize_name(phase["project"]))
    return names


def _iter_pyproject_requirements(pyproject: dict[str, Any]) -> Iterable[str]:
    project = pyproject.get("project") or {}
    yield from project.get("dependencies") or []
    for group in (project.get("optional-dependencies") or {}).values():
        yield from group or []


@dataclass
class DeclaredConstraint:
    """One dependency requirement declared by a repo, before it is checked."""

    package: str  # canonicalized
    raw_requirement: str
    specifier: str
    extras: tuple[str, ...]
    declared_by: str  # path to the pyproject.toml that declared it


def _resolve_fleet_packages(
    repo: Path, fleet_packages: set[str] | None, workspace_yml_path: str | Path | None
) -> set[str]:
    if fleet_packages is not None:
        return fleet_packages
    manifest = (
        Path(workspace_yml_path)
        if workspace_yml_path is not None
        else _find_workspace_manifest(repo)
    )
    return fleet_package_names(manifest) if manifest else set()


def _load_pyproject_toml(pyproject_path: Path) -> dict[str, Any] | None:
    try:
        with pyproject_path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def _constraint_from_requirement(
    raw: str,
    fleet_packages: set[str],
    own_name: str,
    pyproject_path: Path,
    seen: set[tuple[str, str]],
) -> DeclaredConstraint | None:
    try:
        req = Requirement(raw)
    except InvalidRequirement:
        return None
    name = canonicalize_name(req.name)
    if name not in fleet_packages or name == own_name:
        return None  # not a fleet package, or a repo gating on its own name
    specifier = str(req.specifier)
    # The same requirement (e.g. "agent-utilities[mcp]>=2.0.0,<3.0.0")
    # routinely repeats verbatim across `dependencies` and several
    # `optional-dependencies` groups -- dedupe on (package, specifier) so
    # one real gap is reported once, not once per group it happens to
    # appear in.
    dedup_key = (name, specifier)
    if dedup_key in seen:
        return None
    seen.add(dedup_key)
    return DeclaredConstraint(
        package=name,
        raw_requirement=raw,
        specifier=specifier,
        extras=tuple(sorted(req.extras)),
        declared_by=str(pyproject_path),
    )


def declared_fleet_constraints(
    repo_path: str | Path,
    *,
    fleet_packages: set[str] | None = None,
    workspace_yml_path: str | Path | None = None,
) -> list[DeclaredConstraint]:
    """This repo's own declared dependency constraints on OTHER fleet packages.

    ``fleet_packages`` scopes the check to intra-fleet names — never every
    third-party pin, which is out of scope for a *fleet* readiness gate and
    would make the gate noisy on ordinary PyPI releases this fleet doesn't
    control. Pass it explicitly, or a ``workspace_yml_path`` to derive it via
    :func:`fleet_package_names`; with neither, ``workspace.yml`` is looked up
    at the repo root and each of its parents (the layout every fleet repo and
    the top-level workspace share), and an empty declared-constraint list
    (never an error) results when no manifest is found — a repo with no
    reachable manifest has no known fleet scope to check, not a failure.
    """
    repo = Path(repo_path)
    fleet_packages = _resolve_fleet_packages(repo, fleet_packages, workspace_yml_path)

    pyproject_path = repo / _PYPROJECT_NAME
    if not pyproject_path.exists() or not fleet_packages:
        return []

    data = _load_pyproject_toml(pyproject_path)
    if data is None:
        return []

    own_name = canonicalize_name(repo.name)
    out: list[DeclaredConstraint] = []
    seen: set[tuple[str, str]] = set()
    for raw in _iter_pyproject_requirements(data):
        constraint = _constraint_from_requirement(
            raw, fleet_packages, own_name, pyproject_path, seen
        )
        if constraint is not None:
            out.append(constraint)
    return out


def hook_declared(repo_path: str | Path) -> bool:
    """Does ``repo_path``'s own ``.pre-commit-config.yaml`` declare the
    :data:`HOOK_ID` hook?

    A simple substring probe — the same idempotency check
    ``scripts/sweep_dependency_readiness_hook.py`` already uses to decide
    whether a repo needs the hook injected, reused here rather than a second
    YAML-parsing implementation. Fleet rollout of the hook is gradual (that
    script's own docstring), so :func:`await_gate_readiness` (Layer 2) must be
    able to tell "this repo declares a fleet constraint but can't be
    gate-verified yet" apart from "this repo's gate genuinely failed" — this
    is that check.
    """
    config_path = Path(repo_path) / ".pre-commit-config.yaml"
    if not config_path.exists():
        return False
    try:
        content = config_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return f"id: {HOOK_ID}" in content


def _find_workspace_manifest(
    start: Path, filename: str = "workspace.yml"
) -> Path | None:
    """Walk upward from ``start`` for ``workspace.yml`` — the same manifest
    named canonical in ``AGENTS.md``/``workspace.yml`` at the workspace root,
    and mirrored into ``repository_manager/workspace.yml``. Checks the repo's
    own package data copy first (always present, travels with the install),
    then each ancestor directory.
    """
    packaged = Path(__file__).parent / filename
    if packaged.exists():
        return packaged
    current = start.resolve()
    for _ in range(8):
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


# --------------------------------------------------------------------------- #
# The satisfiability check — one constraint against one or more indexes.
# --------------------------------------------------------------------------- #

CheckStatus = Literal[
    "satisfied",
    "unsatisfied",
    "partial_publish",
    "index_unreachable",
    "package_unknown",
]


@dataclass
class ConstraintCheck:
    """The verdict for one declared constraint against the configured index/es.

    ``status="partial_publish"`` is a distinct FAILURE state (see
    :attr:`satisfied`) from ``"unsatisfied"`` — a version exists that would
    satisfy the specifier, but only a wheel or an sdist has been uploaded so
    far, never both (:attr:`_VersionFiles.partial`). It still blocks a push
    (the version is not genuinely installable yet), but names a different
    remediation: wait for the other half of the publish to land, not "this
    constraint may never be satisfiable".

    ``yanked_but_would_match`` lists any version that WOULD have satisfied
    the specifier but was excluded because every one of its files is yanked
    (PEP 592) — folded into the ``"unsatisfied"`` ``detail`` line so that
    verdict is explicable rather than a bare "nothing available" when a
    matching-looking version was in fact seen and deliberately excluded.
    """

    package: str
    raw_requirement: str
    specifier: str
    declared_by: str
    status: CheckStatus
    index_urls_checked: list[str] = field(default_factory=list)
    matching_version: str | None = None
    latest_available: str | None = None
    partial_publish_version: str | None = None
    yanked_but_would_match: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _ConstraintCheckState:
    """Accumulated state across the index scan in :func:`check_constraint`."""

    checked: list[str] = field(default_factory=list)
    best_latest: str | None = None
    saw_reachable_index: bool = False
    unreachable_errors: list[str] = field(default_factory=list)
    best_partial: tuple[str, str] | None = None  # (version, index_url)
    yanked_but_would_match: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _IndexFetchOptions:
    timeout: float
    retries: int
    retry_delay_s: float


def _fetch_available_versions(
    backend: IndexBackend,
    constraint: DeclaredConstraint,
    index_url: str,
    options: _IndexFetchOptions,
) -> tuple[AvailableVersions | None, Exception | None]:
    """One index, with the publish-propagation retry.

    Returns ``(available, last_error)``. ``last_error`` is set only when
    every attempt hit an unreachable index; ``available`` is ``None`` with no
    error when the index was reached but does not know this package.
    """
    last_error: Exception | None = None
    for attempt in range(options.retries):
        try:
            return (
                backend.available_versions(
                    constraint.package, index_url=index_url, timeout=options.timeout
                ),
                None,
            )
        except PackageUnknownError:
            return None, None  # reached the index; it just doesn't know this package
        except IndexUnreachableError as exc:
            last_error = exc
            if attempt + 1 < options.retries:
                time.sleep(options.retry_delay_s)
    return None, last_error


def _record_satisfied_match(
    constraint: DeclaredConstraint,
    index_url: str,
    state: _ConstraintCheckState,
    available: AvailableVersions,
    specifier: SpecifierSet,
) -> ConstraintCheck | None:
    matches = [
        v for v in available.versions if specifier.contains(v, prereleases=False)
    ]
    if not matches:
        return None
    best_match = str(max(Version(v) for v in matches))
    return ConstraintCheck(
        package=constraint.package,
        raw_requirement=constraint.raw_requirement,
        specifier=constraint.specifier,
        declared_by=constraint.declared_by,
        status="satisfied",
        index_urls_checked=state.checked,
        matching_version=best_match,
        latest_available=available.latest,
        detail=(
            f"{constraint.package} {constraint.specifier or '(any)'}: "
            f"{best_match} satisfies it, available from {index_url}"
        ),
    )


def _accumulate_partial_and_yanked(
    state: _ConstraintCheckState,
    available: AvailableVersions,
    specifier: SpecifierSet,
    index_url: str,
) -> None:
    if state.best_partial is None:
        partial_matches = [
            v
            for v in available.partial_versions
            if specifier.contains(v, prereleases=False)
        ]
        if partial_matches:
            state.best_partial = (
                str(max(Version(v) for v in partial_matches)),
                index_url,
            )

    yanked_matches = [
        v
        for v in available.yanked_versions
        if specifier.contains(v, prereleases=False)
        and v not in state.yanked_but_would_match
    ]
    state.yanked_but_would_match.extend(yanked_matches)


def _check_index_for_constraint(
    constraint: DeclaredConstraint,
    index_url: str,
    backend: IndexBackend,
    options: _IndexFetchOptions,
    specifier: SpecifierSet,
    state: _ConstraintCheckState,
) -> ConstraintCheck | None:
    """Check one index; updates ``state`` in place.

    Returns a SATISFIED :class:`ConstraintCheck` to short-circuit the caller,
    or ``None`` to keep scanning the remaining indexes.
    """
    state.checked.append(index_url)
    available, last_error = _fetch_available_versions(
        backend, constraint, index_url, options
    )
    if available is None and last_error is not None:
        state.unreachable_errors.append(f"{index_url}: {last_error}")
        return None  # this index never answered; try the next configured one

    state.saw_reachable_index = True
    if available is None:
        return None  # reached, package unknown to this index; try the next

    if available.latest and (
        state.best_latest is None
        or Version(available.latest) > Version(state.best_latest)
    ):
        state.best_latest = available.latest

    satisfied = _record_satisfied_match(
        constraint, index_url, state, available, specifier
    )
    if satisfied is not None:
        return satisfied

    _accumulate_partial_and_yanked(state, available, specifier, index_url)
    return None


def _index_unreachable_result(
    constraint: DeclaredConstraint, state: _ConstraintCheckState
) -> ConstraintCheck:
    return ConstraintCheck(
        package=constraint.package,
        raw_requirement=constraint.raw_requirement,
        specifier=constraint.specifier,
        declared_by=constraint.declared_by,
        status="index_unreachable",
        index_urls_checked=state.checked,
        detail=(
            f"could not reach any configured index to check {constraint.package} "
            f"{constraint.specifier}: " + "; ".join(state.unreachable_errors)
        ),
    )


def _partial_publish_result(
    constraint: DeclaredConstraint, state: _ConstraintCheckState
) -> ConstraintCheck:
    assert state.best_partial is not None
    partial_version, partial_index_url = state.best_partial
    return ConstraintCheck(
        package=constraint.package,
        raw_requirement=constraint.raw_requirement,
        specifier=constraint.specifier,
        declared_by=constraint.declared_by,
        status="partial_publish",
        index_urls_checked=state.checked,
        latest_available=state.best_latest,
        partial_publish_version=partial_version,
        yanked_but_would_match=sorted(state.yanked_but_would_match, key=Version),
        detail=(
            f"{constraint.package} {constraint.specifier or '(any)'}: "
            f"{partial_version} would satisfy it, but only a wheel OR an "
            f"sdist has been uploaded to {partial_index_url} so far (not "
            "both) — this is a publish still in flight, not an "
            "unsatisfiable constraint; retry rather than treating it as "
            "a hard failure."
        ),
    )


def _unsatisfied_result(
    constraint: DeclaredConstraint, state: _ConstraintCheckState
) -> ConstraintCheck:
    yanked_note = (
        f" (note: {', '.join(sorted(state.yanked_but_would_match, key=Version))} would "
        "satisfy this constraint but every file for that release has been "
        "yanked)"
        if state.yanked_but_would_match
        else ""
    )
    return ConstraintCheck(
        package=constraint.package,
        raw_requirement=constraint.raw_requirement,
        specifier=constraint.specifier,
        declared_by=constraint.declared_by,
        status="unsatisfied",
        index_urls_checked=state.checked,
        latest_available=state.best_latest,
        yanked_but_would_match=sorted(state.yanked_but_would_match, key=Version),
        detail=(
            f"{constraint.package} declares {constraint.specifier or '(any)'} "
            f"(from {constraint.declared_by}) but the configured index "
            f"({', '.join(state.checked)}) currently offers at most "
            f"{state.best_latest or 'no published version'} — unsatisfiable right now."
            f"{yanked_note}"
        ),
    )


def check_constraint(
    constraint: DeclaredConstraint,
    *,
    index_urls: Sequence[str],
    backend: IndexBackend | None = None,
    timeout: float = 10.0,
    retries: int = 2,
    retry_delay_s: float = 3.0,
) -> ConstraintCheck:
    """Is ``constraint`` satisfiable from any of ``index_urls`` right now?

    Tries each index in order (matching resolver precedence: the first index
    with the package wins). A brief retry (``retries`` attempts,
    ``retry_delay_s`` apart) absorbs the publish-but-not-yet-CDN-visible
    window — a version that was just uploaded and has not finished propagating
    reads as transiently unreachable/absent, not as a hard failure, without
    this retry.

    Fails closed: if every configured index is unreachable, the verdict is
    ``"index_unreachable"`` (never silently treated as satisfied). If at least
    one index was reached but none had a satisfying version, the verdict is
    ``"unsatisfied"`` and ``latest_available`` names what IS published, so the
    message states the package, the declared constraint, and what's actually
    available — never just "failed".
    """
    backend = backend or PyPISimpleIndexBackend()
    specifier = (
        SpecifierSet(constraint.specifier) if constraint.specifier else SpecifierSet()
    )

    options = _IndexFetchOptions(
        timeout=timeout, retries=retries, retry_delay_s=retry_delay_s
    )
    state = _ConstraintCheckState()
    for index_url in index_urls:
        result = _check_index_for_constraint(
            constraint, index_url, backend, options, specifier, state
        )
        if result is not None:
            return result

    if not state.saw_reachable_index:
        return _index_unreachable_result(constraint, state)
    if state.best_partial is not None:
        return _partial_publish_result(constraint, state)
    return _unsatisfied_result(constraint, state)


# --------------------------------------------------------------------------- #
# The override — one explicit, loud, auditable escape hatch.
# --------------------------------------------------------------------------- #


def _record_override(
    *, reason: str, repo_path: str | Path, failures: Sequence[Any]
) -> Path:
    """Append one audit line to ``<repo>/.git/dependency-readiness-overrides.log``.

    ``.git`` is always present, always writable, never tracked, and local to
    the checkout that was overridden — no shared-resource contention with
    another lane/session (CONCEPT:RM-DEP-READY loud-override).
    """
    git_dir = Path(repo_path) / ".git"
    log_path = (
        git_dir / "dependency-readiness-overrides.log"
        if git_dir.is_dir()
        else Path(repo_path) / ".dependency-readiness-overrides.log"
    )
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "reason": reason,
        "repo_path": str(repo_path),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "failures": [f.as_dict() for f in failures],
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return log_path


def _print_override_banner(
    *, reason: str, failures: Sequence[Any], log_path: Path, stream=None
) -> None:
    # `stream` is resolved HERE, not as a default-argument value bound once at
    # import time -- a bound-at-def-time `sys.stderr` would keep pointing at
    # the original stream after pytest's `capsys` (or anything else) swaps
    # `sys.stderr` at call time, silently going unseen.
    if stream is None:
        stream = sys.stderr
    print("=" * 78, file=stream)
    print(
        "DEPENDENCY-READINESS GATE OVERRIDDEN — this push is NOT verified", file=stream
    )
    print(f"  reason ({OVERRIDE_ENV_VAR}): {reason}", file=stream)
    print(f"  audit record: {log_path}", file=stream)
    for f in failures:
        print(f"  BYPASSED: {f.detail}", file=stream)
    print("=" * 78, file=stream)


# --------------------------------------------------------------------------- #
# Layer 1 — the pre-push hook entrypoint.
# --------------------------------------------------------------------------- #


@dataclass
class ReadinessReport:
    ok: bool
    repo_path: str
    checks: list[ConstraintCheck] = field(default_factory=list)
    overridden: bool = False
    override_reason: str | None = None
    override_log: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "repo_path": self.repo_path,
            "checks": [c.as_dict() for c in self.checks],
            "overridden": self.overridden,
            "override_reason": self.override_reason,
            "override_log": self.override_log,
        }


def check_tree(
    path: str | Path | None = None,
    *,
    fleet_packages: set[str] | None = None,
    backend: IndexBackend | None = None,
    timeout: float = 10.0,
) -> ReadinessReport:
    """Layer 1: is every intra-fleet constraint THIS repo declares satisfiable?

    Mirrors :func:`repository_manager.parse_gate.check_tree`'s shape (a report
    dataclass, mutates nothing) so it slots into the same pre-commit-hook
    convention. Honors :data:`OVERRIDE_ENV_VAR` — never a silent skip: an
    override still runs every check, still reports every failure, and prints +
    audit-logs the bypass loudly before returning ``ok=True``.
    """
    repo = Path(path or Path.cwd())
    index_urls = resolve_index_urls(repo)
    constraints = declared_fleet_constraints(repo)
    checks = [
        check_constraint(c, index_urls=index_urls, backend=backend, timeout=timeout)
        for c in constraints
    ]
    failures = [c for c in checks if not c.satisfied]
    ok = not failures

    override_reason = os.environ.get(OVERRIDE_ENV_VAR, "").strip()
    overridden = False
    override_log: str | None = None
    if failures and override_reason:
        log_path = _record_override(
            reason=override_reason, repo_path=repo, failures=failures
        )
        _print_override_banner(
            reason=override_reason, failures=failures, log_path=log_path
        )
        overridden = True
        override_log = str(log_path)
        ok = True

    return ReadinessReport(
        ok=ok,
        repo_path=str(repo),
        checks=checks,
        overridden=overridden,
        override_reason=override_reason or None,
        override_log=override_log,
    )


# --------------------------------------------------------------------------- #
# Layer 2 — the phased_push wave barrier: run the real gate, retry-with-backoff
# up to a ceiling, abort loudly (never silently advance).
# --------------------------------------------------------------------------- #


#: Structured failure reasons a :class:`GateCheckFailure` can carry, beyond
#: "the hook ran and failed" (``reason=""``, the original/default shape).
#: Each is also embedded as a ``REASON:`` prefix in ``detail`` so a plain
#: log line stays self-explanatory without needing this field.
GateFailureReason = Literal["", "TARGETS_INCOMPLETE", "CI_RUN_FAILED"]


@dataclass
class GateCheckFailure:
    """One downstream repo whose pre-push gate did not confirm readiness this
    attempt — either the hook itself failed (an unsatisfiable intra-fleet
    constraint, or any other declared HEAVY-gate failure — see
    :func:`await_gate_readiness`), or the repo has not adopted the hook yet so
    its readiness cannot be gate-verified at all. Either way this phase
    transition cannot honestly call the repo ready, so it counts as a block.

    ``reason`` distinguishes the two hard-abort-before-any-retry conditions
    :func:`await_gate_readiness` can now report (:data:`GateFailureReason`)
    from the default gate-failed-this-attempt shape (``reason=""``).
    """

    repo_name: str
    repo_path: str
    detail: str
    reason: GateFailureReason = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateReadinessOutcome:
    """The verdict :meth:`Git.phased_push` acts on for one phase boundary:
    proceed immediately, or abort the wave."""

    ok: bool
    waited_s: float
    attempts: int = 0
    targets_checked: list[str] = field(default_factory=list)
    failures: list[GateCheckFailure] = field(default_factory=list)
    overridden: bool = False
    override_reason: str | None = None
    override_log: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "waited_s": self.waited_s,
            "attempts": self.attempts,
            "targets_checked": self.targets_checked,
            "failures": [f.as_dict() for f in self.failures],
            "overridden": self.overridden,
            "override_reason": self.override_reason,
            "override_log": self.override_log,
        }


def _default_run_gate(repo_path: str) -> Any:
    # Imported lazily (not at module top) purely to keep this module importable
    # standalone (e.g. from inside the pre-commit hook's own minimal
    # environment) without paying for `repository_manager.gates`' import graph
    # unless the gate-driven barrier is actually used.
    from repository_manager.gates import run_gate_stage

    return run_gate_stage(repo_path, "heavy", hook_ids=[HOOK_ID])


def _hook_failure_lines(hook_failures: list[Any]) -> str:
    lines = [
        ln.strip() for h in hook_failures for ln in h.output.splitlines() if ln.strip()
    ]
    return "; ".join(lines) if lines else f"{HOOK_ID} hook failed (no captured output)"


def _gate_failure_detail(repo_name: str, result: Any) -> str | None:
    """``None`` when ``result`` shows the hook passed; otherwise the most
    actionable one-line detail available: the failing hook's own captured
    output (which, for :data:`HOOK_ID`, is exactly Layer 1's
    ``[UNSATISFIED]``/``[INDEX_UNREACHABLE]``/... detail line — see
    :func:`main`), falling back to the run's own ``error`` (e.g. pre-commit's
    "No hook with ID `dependency-readiness`" when the repo hasn't adopted the
    hook)."""
    hook_failures = [h for h in result.hooks if h.hook_id == HOOK_ID and not h.passed]
    if not hook_failures and result.success:
        return None
    if hook_failures:
        return _hook_failure_lines(hook_failures)
    # NOTE (found during CX decomposition, not fixed — out of scope per
    # WD7-RM-03 brief): this branch is unreachable. If hook_failures is
    # empty and reached this point, the first `if` above already proved
    # `result.success` is False, so `result.success` can never be True here.
    if result.success:
        return None
    return (
        result.error
        or f"pre-push gate failed for {repo_name} (exit {result.exit_code})"
    )


def cross_check_targets(
    targets: Sequence[tuple[str, str]],
    *,
    published_packages: set[str],
    candidate_repos: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Independently recompute — from ``candidate_repos``, NEVER by reusing
    whatever narrowing the caller already applied to produce ``targets`` —
    every repo whose own ``pyproject.toml`` NAMES one of ``published_packages``
    (regardless of whether the declared specifier is satisfiable; that is
    :func:`check_constraint`'s job, not this one's), and return the
    ``(repo_name, repo_path)`` pairs present in that independently-computed
    set but ABSENT from ``targets``.

    Closes "the phase succeeded because nothing downstream declared a
    constraint": :func:`await_gate_readiness` only ever gate-checks the
    ``targets`` its caller computed (normally a narrowed slice, e.g.
    ``phased_push``'s ``later_phases``). If that narrowing missed a repo that
    genuinely depends on what was just published — outside the slice scanned,
    added to the fleet after the slice was built, whatever the reason — the
    barrier would report "ready" having never looked at it. A repo returned
    here is a hard-abort condition (CONCEPT:RM-DEP-READY targets-cross-check):
    the real failure was never "nothing declared a constraint", it was
    "something did depend on it and the barrier didn't know".

    Matched by normalized ``repo_path`` (never by name alone — two different
    checkouts of the same repo name is exactly the kind of ambiguity a
    path-based comparison avoids).
    """
    target_paths = {str(Path(path)) for _, path in targets}
    missing: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for name, path in candidate_repos:
        norm_path = str(Path(path))
        if norm_path in target_paths or norm_path in seen_paths:
            continue
        constraints = declared_fleet_constraints(
            path, fleet_packages=set(published_packages)
        )
        if constraints:
            missing.append((name, path))
            seen_paths.add(norm_path)
    return missing


@dataclass(frozen=True)
class _GateReadinessConfig:
    poll_interval_s: float
    max_interval_s: float
    sleep: Any
    now: Any


@dataclass(frozen=True)
class _ForgeRunTarget:
    backend: forge_status.ForgeBackend | None
    owner: str | None
    repo: str | None
    ref: str | None


def _cross_check_failure(
    targets: Sequence[tuple[str, str]],
    targets_checked: list[str],
    published_packages: set[str] | None,
    candidate_repos: Sequence[tuple[str, str]] | None,
) -> GateReadinessOutcome | None:
    if not (published_packages and candidate_repos is not None):
        return None
    missing = cross_check_targets(
        targets,
        published_packages=published_packages,
        candidate_repos=candidate_repos,
    )
    if not missing:
        return None
    return GateReadinessOutcome(
        ok=False,
        waited_s=0.0,
        attempts=0,
        targets_checked=targets_checked,
        failures=[
            GateCheckFailure(
                repo_name=name,
                repo_path=path,
                reason="TARGETS_INCOMPLETE",
                detail=(
                    f"TARGETS_INCOMPLETE: {name} ({path}) declares a "
                    f"constraint on {sorted(published_packages)} but was "
                    "excluded from the computed downstream target set -- "
                    "aborting rather than silently advancing past a repo "
                    "that actually depends on what was just published."
                ),
            )
            for name, path in missing
        ],
    )


def _ci_run_failed_outcome(
    forge: _ForgeRunTarget, run_status: Any, targets_checked: list[str], waited_s: float
) -> GateReadinessOutcome:
    # Guaranteed by _await_ci_run's own guard before it ever calls this
    # helper (it returns None early otherwise); asserted here so mypy can
    # narrow the Optional dataclass fields, matching the narrowing the
    # inline version got for free from the caller's own `if` in main.
    assert forge.owner and forge.repo and forge.ref
    ci_failure_detail = (
        f"CI_RUN_FAILED: the publish run for "
        f"{forge.owner}/{forge.repo}@{forge.ref} concluded "
        f"{run_status.conclusion!r}"
        + (f" ({run_status.url})" if run_status.url else "")
        + " -- aborting immediately instead of burning the "
        "retry ceiling polling an index for an artifact that "
        "is never coming."
    )
    return GateReadinessOutcome(
        ok=False,
        waited_s=waited_s,
        attempts=0,
        targets_checked=targets_checked,
        failures=[
            GateCheckFailure(
                repo_name=forge.repo,
                repo_path=f"{forge.owner}/{forge.repo}",
                reason="CI_RUN_FAILED",
                detail=ci_failure_detail,
            )
        ],
    )


def _await_ci_run(
    forge: _ForgeRunTarget,
    config: _GateReadinessConfig,
    deadline: float,
    started: float,
    targets_checked: list[str],
) -> GateReadinessOutcome | None:
    """Wait for the publish tag's CI run to conclude, sharing the caller's
    deadline/sleep/now. Returns an early-abort outcome on CI_RUN_FAILED, or
    ``None`` to fall through to index-polling (unknown status, concluded
    success, or ran out of budget waiting on CI -- the main loop below also
    sees ``remaining<=0`` and reports a normal timeout).
    """
    if forge.backend is None or not (forge.owner and forge.repo and forge.ref):
        return None
    ci_attempt = 0
    while True:
        ci_attempt += 1
        run_status = forge.backend.latest_run_for_ref(
            forge.owner, forge.repo, forge.ref
        )
        if run_status.state == "unknown":
            return None  # forge status unavailable -- degrade to index-polling
        if run_status.state == "completed":
            if run_status.conclusion not in (None, "success"):
                return _ci_run_failed_outcome(
                    forge, run_status, targets_checked, config.now() - started
                )
            return None  # concluded success -- fall through to index-polling
        remaining = deadline - config.now()
        if remaining <= 0:
            return None
        backoff = min(
            config.poll_interval_s * (2 ** (ci_attempt - 1)),
            config.max_interval_s,
            remaining,
        )
        config.sleep(backoff)


def _check_all_targets(
    targets: Sequence[tuple[str, str]], run_gate: Any
) -> list[GateCheckFailure]:
    failures: list[GateCheckFailure] = []
    for repo_name, repo_path in targets:
        if not hook_declared(repo_path):
            failures.append(
                GateCheckFailure(
                    repo_name=repo_name,
                    repo_path=repo_path,
                    detail=(
                        f"{repo_name} has not adopted the {HOOK_ID!r} pre-push "
                        "hook yet, so its readiness cannot be gate-verified "
                        "(run scripts/sweep_dependency_readiness_hook.py --apply "
                        "for it)"
                    ),
                )
            )
            continue
        result = run_gate(repo_path)
        detail = _gate_failure_detail(repo_name, result)
        if detail is not None:
            failures.append(
                GateCheckFailure(
                    repo_name=repo_name, repo_path=repo_path, detail=detail
                )
            )
    return failures


def _apply_override_if_set(
    audit_repo_path: str | Path,
    failures: list[GateCheckFailure],
    targets_checked: list[str],
    attempt: int,
    waited_s: float,
) -> GateReadinessOutcome | None:
    override_reason = os.environ.get(OVERRIDE_ENV_VAR, "").strip()
    if not override_reason:
        return None
    log_path = _record_override(
        reason=override_reason, repo_path=audit_repo_path, failures=failures
    )
    _print_override_banner(reason=override_reason, failures=failures, log_path=log_path)
    return GateReadinessOutcome(
        ok=True,
        waited_s=waited_s,
        attempts=attempt,
        targets_checked=targets_checked,
        failures=failures,
        overridden=True,
        override_reason=override_reason,
        override_log=str(log_path),
    )


def await_gate_readiness(
    targets: Sequence[tuple[str, str]],
    *,
    wait_minutes: float,
    poll_interval_s: float = 30.0,
    max_interval_s: float = 300.0,
    run_gate: Any = None,
    sleep: Any = time.sleep,
    now: Any = time.monotonic,
    audit_repo_path: str | Path = ".",
    published_packages: set[str] | None = None,
    candidate_repos: Sequence[tuple[str, str]] | None = None,
    forge_backend: forge_status.ForgeBackend | None = None,
    forge_owner: str | None = None,
    forge_repo: str | None = None,
    forge_ref: str | None = None,
) -> GateReadinessOutcome:
    """Decide a ``phased_push`` phase transition by RUNNING each ``targets``
    repo's own pre-push gate — never a second, parallel index-polling
    implementation (CONCEPT:RM-DEP-READY gate-barrier, supersedes the removed
    ``await_constraints``).

    ``targets`` is ``(repo_name, repo_path)`` for every downstream repo that
    declares a fleet constraint on what the phase just published (the caller,
    ``Git._await_phase_dependency_readiness``, computes this). Each target is
    checked via ``run_gate`` (default: :func:`repository_manager.gates.run_gate_stage`
    scoped to just :data:`HOOK_ID` — the SAME call ``Git._gate_before_push``
    makes before that repo's own real push, just narrowed to the one hook this
    barrier cares about so a retry loop reruns a fast network check, not a
    repo's entire heavy suite). A target that has not adopted the hook
    (:func:`hook_declared` is ``False``) is an unverifiable — and therefore
    blocking, never silently passing — target.

    Retries every ``poll_interval_s`` doubling up to ``max_interval_s`` (bounded
    exponential backoff — absorbs the publish-propagation window without
    hammering every target's gate every 30s) until every target's gate passes
    or ``wait_minutes`` elapses. Returns the instant every target passes
    (never waits out the rest of the budget). Never advances past an unmet
    precondition: when the deadline passes with any target still failing,
    ``ok`` is ``False`` and the caller must abort the wave — unless
    :data:`OVERRIDE_ENV_VAR` is set, in which case the bypass is loud and
    audit-logged exactly like the pre-push hook's (never a silent stand-down).
    ``run_gate``/``sleep``/``now`` are injectable so tests exercise a
    30-minute ceiling and repeated retries without a real wait or a real
    pre-commit invocation.

    Two further, optional preconditions, both backward compatible (every
    parameter below defaults to skipping the check entirely, so an existing
    caller that does not pass them observes no behavior change):

    * Pass BOTH ``published_packages`` and ``candidate_repos`` to run
      :func:`cross_check_targets` before anything else — a repo it finds that
      ``targets`` omitted is an immediate hard abort (``ok=False``,
      ``attempts=0``, ``waited_s=0.0``) with a ``reason="TARGETS_INCOMPLETE"``
      failure, never entering the retry loop at all.
    * Pass ``forge_backend`` (e.g. built via
      :func:`repository_manager.forge_status.backend_for_remote`) with
      ``forge_owner``/``forge_repo``/``forge_ref`` to wait for the published
      tag's CI run to reach a conclusion BEFORE polling any target's gate. A
      run that concludes with anything other than success aborts immediately
      with a ``reason="CI_RUN_FAILED"`` failure naming the run's own URL,
      instead of burning the retry ceiling on an index that will never see
      the artifact. This wait shares the SAME ``deadline``/``sleep``/``now``
      as the main retry loop below (never a second, separate budget) — an
      ``"unknown"`` forge status (no client, forge unreachable, ref never
      ran) breaks out of the wait immediately and falls through to today's
      index-polling behavior.
    """
    targets_checked = [name for name, _ in targets]

    cross_check_outcome = _cross_check_failure(
        targets, targets_checked, published_packages, candidate_repos
    )
    if cross_check_outcome is not None:
        return cross_check_outcome

    if not targets:
        return GateReadinessOutcome(ok=True, waited_s=0.0, attempts=0)

    run_gate = run_gate or _default_run_gate
    config = _GateReadinessConfig(
        poll_interval_s=poll_interval_s,
        max_interval_s=max_interval_s,
        sleep=sleep,
        now=now,
    )
    deadline = now() + wait_minutes * 60
    started = now()
    attempt = 0
    failures: list[GateCheckFailure] = []

    ci_outcome = _await_ci_run(
        _ForgeRunTarget(
            backend=forge_backend, owner=forge_owner, repo=forge_repo, ref=forge_ref
        ),
        config,
        deadline,
        started,
        targets_checked,
    )
    if ci_outcome is not None:
        return ci_outcome

    while True:
        attempt += 1
        failures = _check_all_targets(targets, run_gate)

        if not failures:
            return GateReadinessOutcome(
                ok=True,
                waited_s=now() - started,
                attempts=attempt,
                targets_checked=targets_checked,
            )

        remaining = deadline - now()
        if remaining <= 0:
            break
        backoff = min(poll_interval_s * (2 ** (attempt - 1)), max_interval_s, remaining)
        sleep(backoff)

    override_outcome = _apply_override_if_set(
        audit_repo_path, failures, targets_checked, attempt, now() - started
    )
    if override_outcome is not None:
        return override_outcome

    return GateReadinessOutcome(
        ok=False,
        waited_s=now() - started,
        attempts=attempt,
        targets_checked=targets_checked,
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# CLI — the pre-commit hook entrypoint.
# --------------------------------------------------------------------------- #


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core, mirroring ``parse_gate.dispatch`` — a future MCP tool
    dispatches into this instead of a second implementation."""
    if action == "check":
        report = check_tree(kwargs.get("path"))
        return report.as_dict()
    return {"ok": False, "error": f"unknown action: {action}"}


def _print_human_report(report: Any) -> None:
    if not report.checks:
        print("dependency-readiness: no intra-fleet constraints declared here")
    for c in report.checks:
        mark = "OK" if c.satisfied else c.status.upper()
        print(f"  [{mark}] {c.detail}")
    print(
        "OK"
        if report.ok and not report.overridden
        else ("OVERRIDDEN" if report.overridden else "FAILED")
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m repository_manager.dependency_readiness [path]`` — the
    ``[pre-push, manual]`` local hook entry. Exit 1 on any unsatisfiable
    intra-fleet constraint (0 if overridden — see :data:`OVERRIDE_ENV_VAR`)."""
    import argparse

    parser = argparse.ArgumentParser(prog="dependency-readiness", description=__doc__)
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    args = parser.parse_args(argv)

    report = check_tree(args.path)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        _print_human_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
