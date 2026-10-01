"""Read-only command-line pipeline for ``repository-manager-governance history-identity``.

Wires exactly the read phases this package already builds -- history
consistency admission (RM-IDENTITY-R002), ref/tag discovery (RM-IDENTITY-01),
the deterministic preview (RM-IDENTITY-02), and, when an approval file is
supplied, approval verification (RM-IDENTITY-03) -- into the one command the
spec's "the tool enumerates…"/"the tool produces a deterministic dry-run
plan…" language calls for. See :mod:`repository_manager.governance.cli` for
the argument parser and dispatch wiring.

Nothing here mutates a repository: :func:`run_preview` only calls
:func:`~repository_manager.history_identity.consistency.check_history_consistency`,
:func:`~repository_manager.history_identity.discovery.discover`, and
:func:`~repository_manager.history_identity.preview.generate_preview` --
read-only git plumbing plus the existing ``git ls-remote`` reachability probe
(queries a remote, writes nothing). There is no rewrite, ref update, fetch
that updates a ref, or push anywhere in this module or the functions it
calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from repository_manager.history_identity.approval import (
    ApprovalBinding,
    ApprovalError,
    SignedApproval,
    verify_approval,
)
from repository_manager.history_identity.consistency import (
    HistoryConsistencyError,
    admit_for_rewrite_plan,
    check_history_consistency,
)
from repository_manager.history_identity.discovery import DiscoveryError, discover
from repository_manager.history_identity.policy import (
    Identity,
    IdentityPolicy,
    IdentityPolicyError,
    build_policy,
)
from repository_manager.history_identity.preview import HistoryPreview, generate_preview


class HistoryIdentityCliError(RuntimeError):
    """A usage error (bad input file, missing key) -- exit 2, not a plan refusal."""


def _read_json_file(path: Path, *, what: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HistoryIdentityCliError(f"cannot read {what} {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HistoryIdentityCliError(
            f"{what} {path} is not valid JSON: {exc}"
        ) from exc


def _identity_from_json(raw: Any, *, where: str) -> Identity:
    if not isinstance(raw, dict) or "name" not in raw or "email" not in raw:
        raise HistoryIdentityCliError(f"{where} must have 'name' and 'email'")
    return Identity(name=raw["name"], email=raw["email"])


def _load_policy_file(path: Path) -> tuple[str, tuple[tuple[Identity, Identity], ...]]:
    """A policy file's ``{"version": ..., "aliases": [[old, canonical], ...]}``."""
    raw = _read_json_file(path, what="policy file")
    if not isinstance(raw, dict) or "version" not in raw or "aliases" not in raw:
        raise HistoryIdentityCliError(
            f"policy file {path} must have 'version' and 'aliases'"
        )
    aliases = []
    for index, pair in enumerate(raw["aliases"]):
        if not isinstance(pair, list) or len(pair) != 2:
            raise HistoryIdentityCliError(
                f"policy file {path}: aliases[{index}] must be an [old, canonical] pair"
            )
        old, canonical = pair
        aliases.append(
            (
                _identity_from_json(
                    old, where=f"policy file {path} aliases[{index}][0]"
                ),
                _identity_from_json(
                    canonical, where=f"policy file {path} aliases[{index}][1]"
                ),
            )
        )
    return raw["version"], tuple(aliases)


def _load_approval(path: Path) -> SignedApproval:
    raw = _read_json_file(path, what="approval file")
    try:
        binding_raw = raw["binding"]
        binding = ApprovalBinding(
            repository_ids=tuple(binding_raw["repository_ids"]),
            source_digest=binding_raw["source_digest"],
            policy_digest=binding_raw["policy_digest"],
            remotes=tuple(binding_raw["remotes"]),
            expires_at=binding_raw["expires_at"],
        )
        return SignedApproval(
            binding=binding, signer=raw["signer"], signature_hex=raw["signature_hex"]
        )
    except (KeyError, TypeError) as exc:
        raise HistoryIdentityCliError(
            f"approval file {path} is malformed: {exc}"
        ) from exc


def _combined_source_digest(
    repository_ids: tuple[str, ...], preview_digests: tuple[str, ...]
) -> str:
    payload = {
        "repository_ids": sorted(repository_ids),
        "preview_digests": sorted(preview_digests),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_policy_or_refusal(
    *,
    version: str,
    aliases: tuple[tuple[Identity, Identity], ...],
    allowlist_path: Path | None,
) -> IdentityPolicy:
    try:
        return build_policy(
            version=version, aliases=aliases, allowlist_path=allowlist_path
        )
    except IdentityPolicyError as exc:
        raise HistoryIdentityCliError(str(exc)) from exc


def _preview_one_repo(
    repo: str, policy: IdentityPolicy, *, remotes: tuple[str, ...], expiry: str
) -> HistoryPreview:
    """Run the consistency admission + discovery + preview pipeline for one repository.

    Raises :class:`HistoryConsistencyError` or :class:`DiscoveryError` on
    refusal; never writes to ``repo``.
    """
    repo_path = Path(repo).expanduser().resolve()
    admit_for_rewrite_plan(check_history_consistency(repo_path))
    discovery = discover(repo_path)
    return generate_preview(
        repo_path, discovery, policy, remotes=remotes, expiry=expiry
    )


def _preview_report(
    previews: list[HistoryPreview], policy: IdentityPolicy
) -> dict[str, Any]:
    repository_ids = tuple(preview.repo_id for preview in previews)
    preview_digests = tuple(preview.digest for preview in previews)
    return {
        "ok": True,
        "exit_code": 0,
        "policy_digest": policy.digest(),
        "combined_digest": _combined_source_digest(repository_ids, preview_digests),
        "previews": [
            {
                "repo_id": preview.repo_id,
                "discovery_digest": preview.discovery_digest,
                "preview_digest": preview.digest,
                "rewritten_commit_count": preview.rewritten_commit_count,
                "unmapped_commit_count": preview.unmapped_commit_count,
                "collision_warnings": list(preview.collision_warnings),
            }
            for preview in previews
        ],
    }


def _verify_approval_or_refusal(
    report: dict[str, Any],
    *,
    approval_path: str,
    approval_key_hex: str | None,
    policy: IdentityPolicy,
    remotes: tuple[str, ...],
    expiry: str,
) -> dict[str, Any]:
    if not approval_key_hex:
        raise HistoryIdentityCliError(
            "--approval was given but the configured approval-key environment variable is unset"
        )
    repository_ids = tuple(item["repo_id"] for item in report["previews"])
    expected = ApprovalBinding(
        repository_ids=repository_ids,
        source_digest=report["combined_digest"],
        policy_digest=policy.digest(),
        remotes=remotes,
        expires_at=expiry,
    )
    approval = _load_approval(Path(approval_path))
    try:
        key = bytes.fromhex(approval_key_hex)
    except ValueError as exc:
        raise HistoryIdentityCliError(f"approval key is not valid hex: {exc}") from exc
    try:
        verify_approval(approval, expected, key=key)
    except ApprovalError as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "refused": str(exc),
            "previews": report["previews"],
        }
    report["approved"] = True
    return report


def run_preview(
    *,
    repos: list[str],
    policy_path: str,
    allowlist_path: str | None,
    expiry: str,
    remotes: list[str],
    approval_path: str | None,
    approval_key_hex: str | None,
) -> dict[str, Any]:
    """The read-only discovery/preview/approval pipeline over ``repos``.

    Returns a JSON-serializable report with ``exit_code`` set the same way
    every other ``repository-manager-governance`` verb reports a refusal:
    ``0`` for a clean plan, ``1`` for any refusal (diverged history,
    incomplete discovery, an invalid or missing approval when one was
    requested). Raises :class:`HistoryIdentityCliError` for a usage error
    (bad input file, missing policy) -- the caller maps that to exit 2.
    """
    if not repos:
        raise HistoryIdentityCliError(
            "--repo is required (repeat it for more than one repository)"
        )
    version, aliases = _load_policy_file(Path(policy_path))
    policy = _build_policy_or_refusal(
        version=version,
        aliases=aliases,
        allowlist_path=Path(allowlist_path) if allowlist_path else None,
    )
    remotes_tuple = tuple(remotes)
    try:
        previews = [
            _preview_one_repo(repo, policy, remotes=remotes_tuple, expiry=expiry)
            for repo in repos
        ]
    except (HistoryConsistencyError, DiscoveryError) as exc:
        return {"ok": False, "exit_code": 1, "refused": str(exc)}

    report = _preview_report(previews, policy)
    if approval_path:
        return _verify_approval_or_refusal(
            report,
            approval_path=approval_path,
            approval_key_hex=approval_key_hex,
            policy=policy,
            remotes=remotes_tuple,
            expiry=expiry,
        )
    return report
