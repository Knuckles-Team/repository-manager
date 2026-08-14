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
"""

from __future__ import annotations

import json
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
    """What one index says is installable for one package, right now."""

    package: str
    index_url: str
    versions: list[str] = field(default_factory=list)  # sorted ascending
    latest: str | None = None


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
    """``(version, installable)`` parsed from one Simple-API filename, or
    ``None`` when the filename is not a wheel/sdist this parser recognizes
    (e.g. a legacy ``.egg``) — such files are skipped, never mis-attributed.
    """
    try:
        _, version = parse_sdist_filename(filename)
        return str(version), True
    except InvalidSdistFilename:
        pass
    try:
        _, version, _, _ = parse_wheel_filename(filename)
        return str(version), True
    except InvalidWheelFilename:
        return None


def _installable_versions(data: dict[str, Any]) -> list[Version]:
    """Versions with at least one non-yanked installable file.

    Prefers per-file ``yanked`` status (PEP 592) parsed from the Simple-API
    ``files`` list; a release entirely composed of yanked files is excluded,
    matching what a real resolver would accept. Falls back to the top-level
    ``versions`` field (always present under PEP 691) when ``files`` parses to
    nothing usable — e.g. a minimal index response — so a working index is
    never reported as empty just because this parser couldn't read filenames.
    """
    per_version_installable: dict[str, bool] = {}
    for entry in data.get("files", []) or []:
        filename = entry.get("filename", "")
        parsed = _version_from_filename(filename)
        if parsed is None:
            continue
        version, _ = parsed
        yanked = bool(entry.get("yanked"))
        per_version_installable[version] = per_version_installable.get(
            version, False
        ) or (not yanked)

    if not per_version_installable:
        per_version_installable = {v: True for v in (data.get("versions") or [])}

    out: list[Version] = []
    for raw, installable in per_version_installable.items():
        if not installable:
            continue
        try:
            out.append(Version(raw))
        except InvalidVersion:
            continue
    return sorted(out)


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
            data = resp.json()
        except ValueError as exc:
            raise IndexUnreachableError(
                f"index {index_url!r} returned a non-JSON Simple-API response "
                f"for {package!r}: {exc}"
            ) from exc

        versions = _installable_versions(data)
        return AvailableVersions(
            package=package,
            index_url=index_url,
            versions=[str(v) for v in versions],
            latest=str(versions[-1]) if versions else None,
        )


# --------------------------------------------------------------------------- #
# Index resolution — honor the index the repo actually resolves against,
# never a hardcoded pypi.org.
# --------------------------------------------------------------------------- #


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

    def _add(value: str | None) -> None:
        if value and value not in urls:
            urls.append(value)

    _add(os.environ.get("UV_INDEX_URL"))
    for extra in (os.environ.get("UV_EXTRA_INDEX_URL") or "").split():
        _add(extra)
    _add(os.environ.get("PIP_INDEX_URL"))
    for extra in (os.environ.get("PIP_EXTRA_INDEX_URL") or "").split():
        _add(extra)

    pyproject = Path(repo_path) / _PYPROJECT_NAME
    if pyproject.exists():
        try:
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        uv_cfg = (data.get("tool") or {}).get("uv") or {}
        index_entries = uv_cfg.get("index") or []
        default_entries = [e for e in index_entries if e.get("default")]
        other_entries = [e for e in index_entries if not e.get("default")]
        for entry in (*default_entries, *other_entries):
            _add(entry.get("url"))
        _add(uv_cfg.get("index-url"))
        for extra in uv_cfg.get("extra-index-url") or []:
            _add(extra)

    _add(DEFAULT_INDEX_URL)
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


def _walk_manifest_repo_names(
    node: Any, names: set[str], *, skip_keys: frozenset[str] = frozenset()
) -> None:
    if not isinstance(node, dict):
        return
    for repo in node.get("repositories", []) or []:
        url = repo.get("url") if isinstance(repo, dict) else None
        if not url:
            continue
        name = url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[: -len(".git")]
        if name:
            names.add(canonicalize_name(name))
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
    if fleet_packages is None:
        manifest = (
            Path(workspace_yml_path)
            if workspace_yml_path is not None
            else _find_workspace_manifest(repo)
        )
        fleet_packages = fleet_package_names(manifest) if manifest else set()

    pyproject_path = repo / _PYPROJECT_NAME
    if not pyproject_path.exists() or not fleet_packages:
        return []

    try:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return []

    own_name = canonicalize_name(repo.name)
    out: list[DeclaredConstraint] = []
    seen: set[tuple[str, str]] = set()
    for raw in _iter_pyproject_requirements(data):
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        name = canonicalize_name(req.name)
        if name not in fleet_packages:
            continue
        if name == own_name:
            continue  # a repo never gates on its own package name
        specifier = str(req.specifier)
        # The same requirement (e.g. "agent-utilities[mcp]>=2.0.0,<3.0.0")
        # routinely repeats verbatim across `dependencies` and several
        # `optional-dependencies` groups -- dedupe on (package, specifier) so
        # one real gap is reported once, not once per group it happens to
        # appear in.
        dedup_key = (name, specifier)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(
            DeclaredConstraint(
                package=name,
                raw_requirement=raw,
                specifier=specifier,
                extras=tuple(sorted(req.extras)),
                declared_by=str(pyproject_path),
            )
        )
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
    "satisfied", "unsatisfied", "index_unreachable", "package_unknown"
]


@dataclass
class ConstraintCheck:
    """The verdict for one declared constraint against the configured index/es."""

    package: str
    raw_requirement: str
    specifier: str
    declared_by: str
    status: CheckStatus
    index_urls_checked: list[str] = field(default_factory=list)
    matching_version: str | None = None
    latest_available: str | None = None
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    checked: list[str] = []
    best_latest: str | None = None
    saw_reachable_index = False
    unreachable_errors: list[str] = []

    for index_url in index_urls:
        checked.append(index_url)
        available: AvailableVersions | None = None
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                available = backend.available_versions(
                    constraint.package, index_url=index_url, timeout=timeout
                )
                break
            except PackageUnknownError:
                available = None
                last_error = None
                break  # reached the index; it just doesn't know this package
            except IndexUnreachableError as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(retry_delay_s)
        else:
            available = None

        if available is None and last_error is not None:
            unreachable_errors.append(f"{index_url}: {last_error}")
            continue  # this index never answered; try the next configured one

        saw_reachable_index = True
        if available is None:
            continue  # reached, package unknown to this index; try the next

        if available.latest and (
            best_latest is None or Version(available.latest) > Version(best_latest)
        ):
            best_latest = available.latest

        matches = [
            v for v in available.versions if specifier.contains(v, prereleases=False)
        ]
        if matches:
            best_match = str(max(Version(v) for v in matches))
            return ConstraintCheck(
                package=constraint.package,
                raw_requirement=constraint.raw_requirement,
                specifier=constraint.specifier,
                declared_by=constraint.declared_by,
                status="satisfied",
                index_urls_checked=checked,
                matching_version=best_match,
                latest_available=available.latest,
                detail=(
                    f"{constraint.package} {constraint.specifier or '(any)'}: "
                    f"{best_match} satisfies it, available from {index_url}"
                ),
            )

    if not saw_reachable_index:
        return ConstraintCheck(
            package=constraint.package,
            raw_requirement=constraint.raw_requirement,
            specifier=constraint.specifier,
            declared_by=constraint.declared_by,
            status="index_unreachable",
            index_urls_checked=checked,
            detail=(
                f"could not reach any configured index to check {constraint.package} "
                f"{constraint.specifier}: " + "; ".join(unreachable_errors)
            ),
        )

    return ConstraintCheck(
        package=constraint.package,
        raw_requirement=constraint.raw_requirement,
        specifier=constraint.specifier,
        declared_by=constraint.declared_by,
        status="unsatisfied",
        index_urls_checked=checked,
        latest_available=best_latest,
        detail=(
            f"{constraint.package} declares {constraint.specifier or '(any)'} "
            f"(from {constraint.declared_by}) but the configured index "
            f"({', '.join(checked)}) currently offers at most "
            f"{best_latest or 'no published version'} — unsatisfiable right now."
        ),
    )


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


@dataclass
class GateCheckFailure:
    """One downstream repo whose pre-push gate did not confirm readiness this
    attempt — either the hook itself failed (an unsatisfiable intra-fleet
    constraint, or any other declared HEAVY-gate failure — see
    :func:`await_gate_readiness`), or the repo has not adopted the hook yet so
    its readiness cannot be gate-verified at all. Either way this phase
    transition cannot honestly call the repo ready, so it counts as a block.
    """

    repo_name: str
    repo_path: str
    detail: str

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
        lines = [
            ln.strip()
            for h in hook_failures
            for ln in h.output.splitlines()
            if ln.strip()
        ]
        return (
            "; ".join(lines) if lines else f"{HOOK_ID} hook failed (no captured output)"
        )
    if result.success:
        return None
    return (
        result.error
        or f"pre-push gate failed for {repo_name} (exit {result.exit_code})"
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
    """
    if not targets:
        return GateReadinessOutcome(ok=True, waited_s=0.0, attempts=0)

    run_gate = run_gate or _default_run_gate
    deadline = now() + wait_minutes * 60
    started = now()
    attempt = 0
    failures: list[GateCheckFailure] = []
    targets_checked = [name for name, _ in targets]

    while True:
        attempt += 1
        failures = []
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

    override_reason = os.environ.get(OVERRIDE_ENV_VAR, "").strip()
    if override_reason:
        log_path = _record_override(
            reason=override_reason, repo_path=audit_repo_path, failures=failures
        )
        _print_override_banner(
            reason=override_reason, failures=failures, log_path=log_path
        )
        return GateReadinessOutcome(
            ok=True,
            waited_s=now() - started,
            attempts=attempt,
            targets_checked=targets_checked,
            failures=failures,
            overridden=True,
            override_reason=override_reason,
            override_log=str(log_path),
        )

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
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
