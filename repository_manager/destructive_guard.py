"""Refuse destructive git verbs unless a one-shot, snapshot-backed override exists.

This module is deliberately independent of MCP and CLI registration.  Callers
at an execution boundary use :func:`guard` with a fixed argv; the classifier is
also available on its own for admission checks.  A refused command is never
started.  An override is an opaque single-use token, consumed before the
snapshot/command sequence, and cannot be supplied through an environment or a
standing configuration default.

CONCEPT:RM-DESTRUCTIVE-GUARD (C-12)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shlex
import shutil
import subprocess  # nosec B404 - fixed argv execution is the guarded boundary
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repository_manager import stash_guard

logger = logging.getLogger(__name__)

__all__ = [
    "classify",
    "guard",
    "issue_override_token",
    "override_token",
]

_MAX_OUTPUT = 1024 * 1024
_C10_REFUSAL = "conflict/base moved"
_TRUSTED_GIT = shutil.which("git") or "git"
_CWD_OPTION = "-" + "C"
_CONFIG_OPTION = "-" + "c"
_OPTION_TERMINATOR = "-" * 2
_FORCE_BRANCH_OPTION = "-" + "B"
_DELETE_SHORT_OPTION = "-" + "d"
_FORCE_OPTION = "--" + "force"
_DELETE_OPTION = "--" + "delete"
_PRUNE_OPTION = "--" + "prune"
_PRUNE_TAGS_OPTION = "--" + "prune-tags"
_MIRROR_OPTION = "--" + "mirror"
_FORCE_WITH_LEASE_OPTION = "--force-with-lease"
_KNOWN_SUBCOMMANDS = frozenset(
    {
        "branch",
        "cat-file",
        "checkout",
        "clean",
        "diff",
        "for-each-ref",
        "log",
        "ls-files",
        "ls-tree",
        "push",
        "reset",
        "rev-parse",
        "restore",
        "show",
        "stash",
        "status",
    }
)
_ISSUED_TOKENS: set[str] = set()
_USED_TOKENS: set[str] = set()
_TOKEN_AUDIT: dict[str, dict[str, str]] = {}
_TOKEN_LOCK = threading.Lock()


def _clip(value: str) -> str:
    return value[:_MAX_OUTPUT] + (
        "\n[output truncated]" if len(value) > _MAX_OUTPUT else ""
    )


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in value) or "local"


def _normalize_argv(argv: Sequence[str]) -> str:
    """Canonicalize fixed argv for authorization and audit comparisons."""
    values = [str(token) for token in argv]
    if values and os.path.basename(values[0]).lower() == "git":
        values[0] = "git"
    return shlex.join(values)


def _git_executable() -> str:
    """Resolve git once without passing a partial executable path to ``Popen``."""
    return _TRUSTED_GIT


class _ArgvRejected(Exception):
    """Internal control-flow signal for :func:`_git_parts` — caught there and
    translated into its ``(-1, None, reason)`` result tuple. Never escapes
    this module.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_TAKES_VALUE = frozenset(
    {_CWD_OPTION, _CONFIG_OPTION, "--git-dir", "--work-tree", "--namespace"}
)


def _validate_git_executable_arg0(argv: Sequence[str]) -> None:
    executable = str(argv[0])
    if not (
        os.path.sep in executable or (os.path.altsep and os.path.altsep in executable)
    ):
        return
    try:
        matches = (
            Path(executable).expanduser().resolve() == Path(_git_executable()).resolve()
        )
    except OSError as exc:
        raise _ArgvRejected(
            "argv[0] cannot be resolved as the trusted git executable"
        ) from exc
    if not matches:
        raise _ArgvRejected("argv[0] is not the trusted git executable")


def _advance_past_global_option(argv: Sequence[str], index: int, token: str) -> int:
    """``token`` is a bare global option that consumes the next argv value.
    Returns the new index; raises :class:`_ArgvRejected` for a rejected one.
    """
    if index + 1 >= len(argv):
        raise _ArgvRejected(f"missing value for git global option {token}")
    if token == _CONFIG_OPTION:
        raise _ArgvRejected(
            "git configuration injection is not accepted at this boundary"
        )
    if token in {"--git-dir", "--work-tree"}:
        raise _ArgvRejected(
            "alternate git directory/work tree is not accepted at this boundary"
        )
    return index + 2


def _reject_inline_global_option(token: str) -> None:
    """``token`` is a ``--option=value`` form global option."""
    if token.startswith(_CONFIG_OPTION + "="):
        raise _ArgvRejected(
            "git configuration injection is not accepted at this boundary"
        )
    if token.startswith("--git-dir=") or token.startswith("--work-tree="):
        raise _ArgvRejected(
            "alternate git directory/work tree is not accepted at this boundary"
        )


def _reject_short_config_option(token: str) -> None:
    if token.startswith(_CONFIG_OPTION) and len(token) > len(_CONFIG_OPTION):
        raise _ArgvRejected(
            "git configuration injection is not accepted at this boundary"
        )
    if token.startswith("--config-env="):
        raise _ArgvRejected(
            "git configuration injection is not accepted at this boundary"
        )


def _scan_for_git_subcommand(argv: Sequence[str]) -> tuple[int, str | None, str | None]:
    index = 1
    while index < len(argv):
        token = str(argv[index])
        if token in _TAKES_VALUE:
            index = _advance_past_global_option(argv, index, token)
            continue
        if any(token.startswith(prefix + "=") for prefix in _TAKES_VALUE):
            _reject_inline_global_option(token)
            index += 1
            continue
        _reject_short_config_option(token)
        if token == _OPTION_TERMINATOR:
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        subcommand = token.lower()
        if subcommand not in _KNOWN_SUBCOMMANDS:
            return (
                index,
                subcommand,
                f"unsupported or unresolved git subcommand: {subcommand}",
            )
        return index, subcommand, None
    return -1, None, "git subcommand is required"


def _git_parts(argv: Sequence[str]) -> tuple[int, str | None, str | None]:
    """Find a git subcommand and reject malformed or alias-bearing argv."""
    if not argv or os.path.basename(str(argv[0])).lower() != "git":
        return -1, None, "argv[0] must resolve to git"
    try:
        _validate_git_executable_arg0(argv)
        return _scan_for_git_subcommand(argv)
    except _ArgvRejected as exc:
        return -1, None, exc.reason


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "dangerous": False,
        "supported": False,
        "pattern": None,
        "safer_alternative": None,
        "reason": "unsupported-git-argv",
        "error": reason,
    }


def _safe() -> dict[str, Any]:
    return {
        "dangerous": False,
        "supported": True,
        "pattern": None,
        "safer_alternative": None,
    }


def _rule(pattern: str, alternative: str, alternative_code: str) -> dict[str, Any]:
    return {
        "supported": True,
        "pattern": pattern,
        "safer_alternative": alternative,
        "alternative_code": alternative_code,
        "refusal_code": _C10_REFUSAL,
    }


def classify(argv: Sequence[str]) -> dict[str, Any]:
    """Classify a fixed argv without executing it.

    Unknown/non-git argv is reported as unsupported.  Only a bounded set of
    known git subcommands reaches the normal fixed-argv executor; a dangerous
    match always requires an explicit override and a recovery snapshot.
    """
    values = [str(token) for token in argv]
    sub_index, subcommand, parse_error = _git_parts(values)
    if parse_error is not None or sub_index < 0 or subcommand is None:
        return _unsupported(parse_error or "malformed git argv")
    args = values[sub_index + 1 :]

    classifier = _SUBCOMMAND_CLASSIFIERS.get(subcommand)
    if classifier is not None:
        result = classifier(args)
        if result is not None:
            return result

    return _safe()


def _classify_reset(args: list[str]) -> dict[str, Any] | None:
    # NOTE: this first special case and the general reset handling directly
    # below it always produce the identical dict when "--hard" is present
    # (both use pattern "git reset --hard" / alternative_code "mixed-reset")
    # -- preserved verbatim, redundant-but-harmless, from the original
    # unrefactored function; not a behavior change to fix here.
    if "--hard" in args:
        return {
            "dangerous": True,
            **_rule(
                "git reset --hard",
                "use `git reset --mixed <target>` and inspect the resulting status",
                "mixed-reset",
            ),
        }

    pattern = "git reset --hard" if "--hard" in args else "git reset (index mutation)"
    alternative = (
        "use `git reset --mixed <target>` and inspect the resulting status"
        if "--hard" in args
        else "inspect the index and use a reviewed path-specific operation"
    )
    return {
        "dangerous": True,
        **_rule(pattern, alternative, "mixed-reset"),
    }


def _checkout_is_forced_or_targets_everything(args: list[str]) -> bool:
    pathspecs = [token for token in args if not token.startswith("-")]
    return (
        _checkout_is_forced(args)
        or "." in pathspecs
        or ("--" in args and "." in args[args.index("--") + 1 :])
    )


def _checkout_is_forced(args: list[str]) -> bool:
    return any(
        token == _FORCE_OPTION
        or token == _FORCE_BRANCH_OPTION
        or (token.startswith("-") and not token.startswith("--") and "f" in token[1:])
        for token in args
    )


def _classify_checkout(args: list[str]) -> dict[str, Any] | None:
    if _checkout_is_forced_or_targets_everything(args):
        return {
            "dangerous": True,
            **_rule(
                "git checkout -f / git checkout . / git checkout -- .",
                "inspect the diff and use a reviewed, path-specific operation",
                "reviewed-path-operation",
            ),
        }
    return {
        "dangerous": True,
        **_rule(
            "git checkout (working-tree mutation)",
            "inspect the diff and use a reviewed, path-specific operation",
            "reviewed-path-operation",
        ),
    }


def _classify_restore(args: list[str]) -> dict[str, Any] | None:
    return {
        "dangerous": True,
        **_rule(
            "git restore (working-tree/index mutation)",
            "inspect the diff and use a reviewed, path-specific operation",
            "reviewed-path-operation",
        ),
    }


def _clean_targets_ignored_scope(args: list[str]) -> bool:
    return any(
        token.startswith("-")
        and not token.startswith("--")
        and ("x" in token or "X" in token)
        for token in args
    )


def _clean_is_forced(args: list[str]) -> bool:
    return any(
        token == _FORCE_OPTION
        or (token.startswith("-") and not token.startswith("--") and "f" in token)
        for token in args
    )


def _classify_clean(args: list[str]) -> dict[str, Any] | None:
    ignored_scope = _clean_targets_ignored_scope(args)
    if _clean_is_forced(args) or ignored_scope:
        return {
            "dangerous": True,
            **_rule(
                "git clean -f*",
                "review untracked paths first; preserve them with `park`/`unpark`",
                "park-unpark",
            ),
            "ignored_scope": ignored_scope,
        }
    return None


def _classify_stash(args: list[str]) -> dict[str, Any] | None:
    action = next((token for token in args if not token.startswith("-")), "")
    private_ref = any(
        token.startswith(("refs/lane/", "refs/lane-backup/")) for token in args
    )
    # Only inspection of an explicitly named private commit is safe.
    # Applying it directly would mutate the worktree outside the
    # park/unpark protocol and outside the guard's mutation lease; callers
    # must route restoration through ``stash_guard.unpark`` instead.  A
    # private-looking token on ``push`` must not make Git write refs/stash.
    if private_ref and action == "show":
        return _safe()
    return {
        "dangerous": True,
        **_rule(
            "git stash (shared refs/stash)",
            "use `stash_guard.park` and `stash_guard.unpark` with a private ref",
            "park-unpark",
        ),
        "stash_action": action or "stash",
    }


def _classify_branch(args: list[str]) -> dict[str, Any] | None:
    has_delete, has_force = _branch_delete_and_force_flags(args)
    forced_delete = has_delete and has_force
    if forced_delete:
        return {
            "dangerous": True,
            **_rule(
                "git branch -D",
                "use the guarded prune path and git's `branch -d` reachability check",
                "guarded-prune",
            ),
        }
    if has_force:
        return {
            "dangerous": True,
            **_rule(
                "git branch --force",
                "review the target ref before a non-force branch update",
                "guarded-prune",
            ),
        }
    return None


def _branch_delete_and_force_flags(args: list[str]) -> tuple[bool, bool]:
    has_delete = False
    has_force = False
    for token in args:
        if token == _DELETE_OPTION or token.startswith(_DELETE_OPTION + "="):
            has_delete = True
        elif token == _FORCE_OPTION:
            has_force = True
        elif token.startswith("-") and not token.startswith("--"):
            delete_bit, force_bit = _branch_short_option_flags(token)
            has_delete = has_delete or delete_bit
            has_force = has_force or force_bit
    return has_delete, has_force


def _branch_short_option_flags(token: str) -> tuple[bool, bool]:
    short = token[1:]
    return (
        "d" in short or "D" in short,
        "f" in short or "D" in short or "M" in short,
    )


def _classify_push(args: list[str]) -> dict[str, Any] | None:
    if _push_is_forced(args) or _push_requests_remote_delete(args):
        return {
            "dangerous": True,
            **_rule(
                "git push --force",
                "use a regular fast-forward push after a reviewed reconciliation",
                "reviewed-fast-forward",
            ),
        }
    return None


def _push_requests_remote_delete(args: list[str]) -> bool:
    return any(_push_token_requests_remote_delete(token) for token in args)


def _push_token_requests_remote_delete(token: str) -> bool:
    return (
        token == _DELETE_OPTION
        or token == _DELETE_SHORT_OPTION
        or token in {_PRUNE_OPTION, _PRUNE_TAGS_OPTION}
        or token.startswith(_PRUNE_OPTION + "=")
        or token.startswith(_PRUNE_TAGS_OPTION + "=")
        or (token.startswith(":") and len(token) > 1)
        or (token.startswith("-") and not token.startswith("--") and "d" in token[1:])
    )


def _push_is_forced(args: list[str]) -> bool:
    return any(
        token in {"-f", _FORCE_OPTION, _FORCE_WITH_LEASE_OPTION}
        or token == _MIRROR_OPTION
        or token.startswith("--force=")
        or token.startswith(_FORCE_WITH_LEASE_OPTION + "=")
        or (token.startswith("-") and not token.startswith("--") and "f" in token[1:])
        or (token.startswith("+") and len(token) > 1)
        for token in args
    )


_SUBCOMMAND_CLASSIFIERS: dict[str, Callable[[list[str]], dict[str, Any] | None]] = {
    "reset": _classify_reset,
    "checkout": _classify_checkout,
    "restore": _classify_restore,
    "clean": _classify_clean,
    "stash": _classify_stash,
    "branch": _classify_branch,
    "push": _classify_push,
}


def _verify_override_authorization(authorization: Callable[[], bool] | None) -> None:
    if authorization is None or not callable(authorization):
        raise PermissionError("explicit override authorization callback required")
    try:
        authorized = bool(authorization())
    except Exception as exc:  # pragma: no cover - external authorization seam
        raise PermissionError("override authorization callback failed") from exc
    if not authorized:
        raise PermissionError("explicit override authorization callback required")


def _validate_override_audit_context(audit_context: Mapping[str, str] | None) -> None:
    required = {"actor", "lane", "operation", "repository", "argv"}
    if audit_context is None or not required.issubset(audit_context):
        raise PermissionError(
            "override audit context must name actor, lane, repository, argv, and operation"
        )
    if any(not str(audit_context[key]).strip() for key in required):
        raise PermissionError("override audit context fields must be non-empty")


def _validate_override_audit_argv(audit_context: Mapping[str, str]) -> None:
    try:
        normalized_audit_argv = _normalize_argv(shlex.split(str(audit_context["argv"])))
    except ValueError as exc:
        raise PermissionError("override audit argv is not valid fixed argv") from exc
    if str(audit_context["argv"]) != normalized_audit_argv:
        raise PermissionError("override audit argv must be normalized fixed argv")


def _resolve_override_repository(audit_context: Mapping[str, str]) -> str:
    try:
        return str(Path(str(audit_context["repository"])).expanduser().resolve())
    except (OSError, RuntimeError) as exc:
        raise PermissionError("override repository identity is not resolvable") from exc


def issue_override_token(
    *,
    authorization: Callable[[], bool] | None = None,
    audit_context: Mapping[str, str] | None = None,
) -> str:
    """Mint an opaque token for one authorized, auditable invocation.

    Token minting intentionally has no environment/configuration fallback.  A
    caller must supply a live authorization callback and a context naming the
    actor, lane, repository, normalized argv, and operation; this keeps a
    standing ``RM_*`` switch from turning every future refusal into an
    unreviewed destructive action.
    """
    _verify_override_authorization(authorization)
    _validate_override_audit_context(audit_context)
    assert audit_context is not None
    _validate_override_audit_argv(audit_context)
    repository = _resolve_override_repository(audit_context)
    token = f"rm-override-{uuid.uuid4().hex}"
    with _TOKEN_LOCK:
        _ISSUED_TOKENS.add(token)
        _TOKEN_AUDIT[token] = {
            str(key): str(value) for key, value in audit_context.items()
        }
        _TOKEN_AUDIT[token]["repository"] = repository
    logger.warning(
        "destructive override token minted: actor=%s lane=%s operation=%s",
        audit_context["actor"],
        audit_context["lane"],
        audit_context["operation"],
    )
    return token


override_token = issue_override_token


def _consume_override(
    token: object,
    *,
    lane: str,
    operation: str,
    argv: Sequence[str],
    repository: Path,
) -> tuple[bool, str]:
    if not isinstance(token, str) or not token.strip():
        return False, "override token is not a token string"
    with _TOKEN_LOCK:
        if token not in _ISSUED_TOKENS:
            return False, "override token was not minted by the authorization seam"
        if token in _USED_TOKENS:
            return False, "override token is already consumed"
        audit = _TOKEN_AUDIT[token]
        if audit.get("lane") != lane:
            return False, "override token is bound to a different lane"
        normalized_argv = _normalize_argv(argv)
        if audit.get("argv") != normalized_argv:
            return False, "override token is bound to different argv"
        raw_operation = normalized_argv
        if audit.get("operation") not in {operation, raw_operation}:
            return False, "override token is bound to a different operation"
        expected_repo = audit.get("repository")
        if not expected_repo:
            return False, "override token has no repository binding"
        if Path(expected_repo).expanduser().resolve() != repository:
            return False, "override token is bound to a different repository"
        _USED_TOKENS.add(token)
    return True, "consumed"


class _GitAdapter:
    """The narrow ``stash_guard`` protocol backed by fixed argv subprocesses."""

    def git_action(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict[str, str] | None = None,
        timeout: int = 1800,
        raw_output: bool = False,
    ) -> Any:
        del quiet, raw_output
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            return _AdapterResult(False, "", str(exc))
        if not argv or os.path.basename(argv[0]).lower() != "git":
            return _AdapterResult(False, "", "adapter accepts only git commands")
        argv[0] = _git_executable()
        try:
            result = subprocess.run(
                argv,
                cwd=path,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return _AdapterResult(False, "", f"{type(exc).__name__}: {exc}")
        output = (result.stdout or "") + (result.stderr or "")
        return _AdapterResult(result.returncode == 0, output.strip(), output.strip())


class _AdapterResult:
    def __init__(self, ok: bool, data: str, error: str):
        self.status = "success" if ok else "error"
        self.data = data
        self.error = type("Error", (), {"message": error})() if not ok else None


def _path_from_argv(argv: Sequence[str], path: Path) -> Path:
    """Honor ``git -C`` when present, without accepting shell syntax."""
    values = [str(token) for token in argv]
    sub_index, _subcommand, _error = _git_parts(values)
    prefix = values[:sub_index] if sub_index >= 0 else values
    for index, token in enumerate(prefix[:-1]):
        if token == _CWD_OPTION:
            target = Path(values[index + 1]).expanduser()
            return (target if target.is_absolute() else path / target).resolve()
        if token.startswith("-C") and len(token) > 2:
            target = Path(token[2:]).expanduser()
            return (target if target.is_absolute() else path / target).resolve()
    return path


def _snapshot_head_sha(
    path: Path, operation: str, created_at: str, timeout: int
) -> tuple[str, dict[str, Any] | None]:
    head = subprocess.run(
        [_git_executable(), "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    sha = head.stdout.strip() if head.returncode == 0 else ""
    if sha:
        return sha, None
    return "", {
        "ok": False,
        "snapshot_created_at": created_at,
        "triggering_operation": operation,
        "error": "cannot create a recovery point without HEAD",
    }


def _write_snapshot_ref(
    path: Path, ref: str, sha: str, operation: str, created_at: str, timeout: int
) -> dict[str, Any] | None:
    stored = subprocess.run(
        [_git_executable(), "update-ref", ref, sha],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if stored.returncode == 0:
        return None
    return {
        "ok": False,
        "snapshot_ref": ref,
        "snapshot_sha": sha,
        "snapshot_created_at": created_at,
        "triggering_operation": operation,
        "error": (stored.stderr or stored.stdout or "git update-ref failed").strip(),
    }


def _verify_snapshot_tree_status(
    path: Path, ref: str, sha: str, operation: str, created_at: str, timeout: int
) -> dict[str, Any] | None:
    status = subprocess.run(
        [_git_executable(), "status", "--porcelain"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if status.returncode == 0:
        return None
    return {
        "ok": False,
        "snapshot_ref": ref,
        "snapshot_sha": sha,
        "snapshot_created_at": created_at,
        "triggering_operation": operation,
        "error": (status.stderr or status.stdout or "git status failed").strip(),
    }


def _park_snapshot_wip(
    path: Path, lane: str, ref: str, sha: str, operation: str, created_at: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(parked, error)`` — ``error`` is set (and ``parked`` is ``None``) when
    the WIP park itself failed.
    """
    parked = stash_guard.park(
        _GitAdapter(),
        str(path),
        lane=lane,
        message=f"pre-destructive {operation}",
        _lease=False,
    )
    if parked.get("ok"):
        return parked, None
    return None, {
        "ok": False,
        "snapshot_ref": ref,
        "snapshot_sha": sha,
        "snapshot_created_at": created_at,
        "triggering_operation": operation,
        "park_ref": parked.get("ref"),
        "error": f"snapshot WIP park failed: {parked.get('error', 'unknown error')}",
    }


def _snapshot(
    path: Path,
    lane: str,
    operation: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    """Create the commit ref and, when needed, park dirty WIP before execution."""
    created_at = dt.datetime.now(dt.UTC).isoformat()
    sha, error = _snapshot_head_sha(path, operation, created_at, timeout)
    if error is not None:
        return error
    ref = f"refs/lane-backup/pre-destructive/{_slug(lane)}-{uuid.uuid4().hex}"
    error = _write_snapshot_ref(path, ref, sha, operation, created_at, timeout)
    if error is not None:
        return error
    error = _verify_snapshot_tree_status(path, ref, sha, operation, created_at, timeout)
    if error is not None:
        return error
    # Always use the temp-index/private-tree path.  Porcelain can be empty
    # for assume-unchanged or skip-worktree entries even when their bytes have
    # changed, so status is only an error check and never a completeness gate.
    parked, error = _park_snapshot_wip(path, lane, ref, sha, operation, created_at)
    if error is not None:
        return error
    assert parked is not None
    park_ref = parked.get("ref") if parked.get("parked") else None
    logger.warning(
        "destructive override snapshot created: operation=%s lane=%s ref=%s sha=%s park_ref=%s",
        operation,
        lane,
        ref,
        sha,
        park_ref,
    )
    return {
        "ok": True,
        "snapshot_ref": ref,
        "snapshot_sha": sha,
        "snapshot_created_at": created_at,
        "triggering_operation": operation,
        "park_ref": park_ref,
    }


@dataclass
class _DestructiveRunContext:
    """Bundled parameters + accumulating result for one guarded destructive
    run — every phase reads/writes ``base`` in place, exactly as the
    original single function's locals did.
    """

    values: list[str]
    decision: dict[str, Any]
    tree: Path
    effective_lane: str
    override: str | None
    timeout: int
    execute: bool
    base: dict[str, Any]


def _refuse_ignored_clean_scope(ctx: _DestructiveRunContext) -> dict[str, Any] | None:
    if not ctx.decision.get("ignored_scope"):
        return None
    ctx.base.update(
        {
            "reason": "ignored-clean-forbidden",
            "error": (
                "refused git clean -x/-X: ignored content is not captured by "
                "the bounded private park; remove it only through an explicitly "
                "reviewed, separate recovery workflow"
            ),
        }
    )
    logger.warning("destructive git verb refused: ignored clean scope")
    return ctx.base


def _refuse_without_override(ctx: _DestructiveRunContext) -> dict[str, Any] | None:
    if isinstance(ctx.override, str) and ctx.override.strip():
        return None
    ctx.base.update(
        {
            "reason": "destructive git verb refused by default",
            "error": f"refused {ctx.decision['pattern']}: {ctx.decision['safer_alternative']}",
        }
    )
    logger.warning("destructive git verb refused: %s", ctx.decision["pattern"])
    return ctx.base


def _consume_override_or_refuse(ctx: _DestructiveRunContext) -> dict[str, Any] | None:
    consumed, consume_reason = _consume_override(
        ctx.override,
        lane=ctx.effective_lane,
        operation=ctx.decision["pattern"],
        argv=ctx.values,
        repository=ctx.tree,
    )
    if consumed:
        ctx.base["override_consumed"] = True
        return None
    ctx.base.update(
        {
            "reason": "override-single-use"
            if "consumed" in consume_reason
            else "override-scope-mismatch",
            "error": consume_reason,
        }
    )
    return ctx.base


def _snapshot_or_refuse(ctx: _DestructiveRunContext) -> dict[str, Any] | None:
    ctx.base["snapshot_required"] = True
    snapshot = _snapshot(
        ctx.tree, ctx.effective_lane, _normalize_argv(ctx.values), timeout=ctx.timeout
    )
    ctx.base.update(
        {
            key: snapshot.get(key)
            for key in (
                "snapshot_ref",
                "snapshot_sha",
                "snapshot_created_at",
                "triggering_operation",
                "park_ref",
            )
        }
    )
    if snapshot.get("ok"):
        return None
    ctx.base.update(
        {
            "reason": "snapshot-required",
            "error": snapshot.get("error", "recovery snapshot failed"),
            "override_consumed": True,
        }
    )
    logger.error("destructive operation refused: recovery snapshot failed")
    return ctx.base


def _preflight_tree_check(ctx: _DestructiveRunContext) -> dict[str, Any] | None:
    # The snapshot and execution share one cooperative lease.  A second RM
    # actor cannot introduce WIP between park and the destructive command.
    preflight = subprocess.run(
        [_git_executable(), "status", "--porcelain"],
        cwd=str(ctx.tree),
        capture_output=True,
        text=True,
        check=False,
        timeout=ctx.timeout,
    )
    if preflight.returncode != 0:
        return {
            **ctx.base,
            "reason": "tree-invariant-failed",
            "error": (preflight.stderr or preflight.stdout or "status failed").strip(),
        }
    if preflight.stdout.strip():
        return {
            **ctx.base,
            "reason": "tree-changed-after-snapshot",
            "error": "working tree changed after recovery snapshot; operation refused",
        }
    return None


def _run_destructive_git_command(
    ctx: _DestructiveRunContext,
) -> tuple[subprocess.CompletedProcess[str] | None, dict[str, Any] | None]:
    """``(result, error)`` — ``error`` is the base-derived failure dict to
    return immediately when the subprocess itself could not be run.
    """
    try:
        result = subprocess.run(
            [_git_executable(), *ctx.values[1:]],
            cwd=str(ctx.tree),
            capture_output=True,
            text=True,
            check=False,
            timeout=ctx.timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, {
            **ctx.base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "snapshot_required": True,
        }
    return result, None


def _build_execution_result(
    ctx: _DestructiveRunContext,
    result: subprocess.CompletedProcess[str],
    final_status: str,
    postcondition_ok: bool,
) -> dict[str, Any]:
    if result.returncode == 0 and not postcondition_ok:
        return {
            **ctx.base,
            "ok": False,
            "status": "postcondition-failed",
            "executed": True,
            "returncode": result.returncode,
            "stdout": _clip(result.stdout or ""),
            "stderr": _clip(result.stderr or ""),
            "final_status": final_status,
            "final_tree_clean": False,
            "snapshot_required": True,
            "override_used": True,
            "error": "working tree is not clean after destructive operation",
        }
    return {
        **ctx.base,
        "ok": result.returncode == 0,
        "status": "success" if result.returncode == 0 else "error",
        "executed": True,
        "returncode": result.returncode,
        "stdout": _clip(result.stdout or ""),
        "stderr": _clip(result.stderr or ""),
        "final_status": final_status,
        "final_tree_clean": postcondition_ok,
        "snapshot_required": True,
        "override_used": True,
    }


def _postcondition_result(
    ctx: _DestructiveRunContext, result: subprocess.CompletedProcess[str]
) -> dict[str, Any]:
    final = subprocess.run(
        [_git_executable(), "status", "--porcelain"],
        cwd=str(ctx.tree),
        capture_output=True,
        text=True,
        check=False,
        timeout=ctx.timeout,
    )
    final_status = (final.stdout or final.stderr or "").strip()
    postcondition_ok = final.returncode == 0 and not final.stdout.strip()
    logger.warning(
        "destructive override used once: pattern=%s lane=%s snapshot_ref=%s",
        ctx.decision["pattern"],
        ctx.effective_lane,
        ctx.base["snapshot_ref"],
    )
    return _build_execution_result(ctx, result, final_status, postcondition_ok)


def _execute_and_verify(ctx: _DestructiveRunContext) -> dict[str, Any]:
    if not ctx.execute:
        return {
            **ctx.base,
            "ok": True,
            "status": "authorized",
            "reason": "snapshot-backed override authorized; execution suppressed",
        }
    result, error = _run_destructive_git_command(ctx)
    if error is not None:
        return error
    assert result is not None
    return _postcondition_result(ctx, result)


def _run_destructive(
    values: list[str],
    decision: dict[str, Any],
    tree: Path,
    effective_lane: str,
    override: str | None,
    timeout: int,
    execute: bool,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Run the override path while the caller holds the worktree lease."""
    base.update(decision)
    ctx = _DestructiveRunContext(
        values=values,
        decision=decision,
        tree=tree,
        effective_lane=effective_lane,
        override=override,
        timeout=timeout,
        execute=execute,
        base=base,
    )
    for phase in (
        _refuse_ignored_clean_scope,
        _refuse_without_override,
        _consume_override_or_refuse,
        _snapshot_or_refuse,
        _preflight_tree_check,
    ):
        refusal = phase(ctx)
        if refusal is not None:
            return refusal
    return _execute_and_verify(ctx)


def _build_guard_base_result(
    values: list[str], tree: Path, effective_lane: str
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "refused",
        "executed": False,
        "argv": values,
        "repository": str(tree),
        "lane": effective_lane,
        "snapshot_ref": None,
        "snapshot_sha": None,
        "snapshot_created_at": None,
        "triggering_operation": None,
        "park_ref": None,
        "override_consumed": False,
        "override_used": False,
        "snapshot_required": False,
        "stdout": "",
        "stderr": "",
    }


def _refuse_unsupported_argv(
    decision: dict[str, Any], base: dict[str, Any]
) -> dict[str, Any] | None:
    if decision.get("supported", False):
        return None
    base.update(decision)
    base.update(
        {
            "status": "refused",
            "reason": "unsupported-git-argv",
            "error": decision.get("error", "unsupported or malformed git argv"),
        }
    )
    logger.warning("git operation refused at guard boundary: %s", base["error"])
    return base


def _run_safe_argv(
    values: list[str], tree: Path, timeout: int, execute: bool, base: dict[str, Any]
) -> dict[str, Any]:
    if not execute:
        return {**base, "ok": True, "status": "allowed", "reason": "safe argv"}
    try:
        result = subprocess.run(
            [_git_executable(), *values[1:]],
            cwd=str(tree),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        **base,
        "ok": result.returncode == 0,
        "status": "success" if result.returncode == 0 else "error",
        "executed": True,
        "returncode": result.returncode,
        "stdout": _clip(result.stdout or ""),
        "stderr": _clip(result.stderr or ""),
    }


def _run_dangerous_argv(ctx: _DestructiveRunContext) -> dict[str, Any]:
    try:
        with stash_guard.hold_tree_mutation_lease(
            str(ctx.tree), note=f"destructive {ctx.decision['pattern']}"
        ):
            return _run_destructive(
                ctx.values,
                ctx.decision,
                ctx.tree,
                ctx.effective_lane,
                ctx.override,
                ctx.timeout,
                ctx.execute,
                ctx.base,
            )
    except stash_guard.BlockedByLease as exc:
        return {
            **ctx.base,
            "reason": "tree-mutation-busy",
            "error": str(exc),
        }
    except (OSError, RuntimeError) as exc:
        return {
            **ctx.base,
            "status": "error",
            "reason": "tree-mutation-lease-unavailable",
            "error": str(exc),
        }


def guard(
    argv: Sequence[str],
    *,
    path: Path | str | None = None,
    lane: str | None = None,
    override: str | None = None,
    timeout: int = 1800,
    execute: bool = True,
) -> dict[str, Any]:
    """Classify and, when permitted, execute one fixed-argv operation.

    Destructive matches refuse before spawning a process.  Supplying a
    single-use token from :func:`issue_override_token` authorizes one execution
    only; that execution still requires a successful commit ref and (for a
    dirty tree) a private WIP park.  ``execute=False`` is useful for callers
    that only need the admission result.
    """
    values = [str(token) for token in argv]
    decision = classify(values)
    tree = _path_from_argv(values, Path(path or os.getcwd()).expanduser().resolve())
    effective_lane = lane or tree.name or "local"
    base = _build_guard_base_result(values, tree, effective_lane)

    unsupported = _refuse_unsupported_argv(decision, base)
    if unsupported is not None:
        return unsupported
    if not decision.get("dangerous"):
        return _run_safe_argv(values, tree, timeout, execute, base)

    ctx = _DestructiveRunContext(
        values=values,
        decision=decision,
        tree=tree,
        effective_lane=effective_lane,
        override=override,
        timeout=timeout,
        execute=execute,
        base=base,
    )
    return _run_dangerous_argv(ctx)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m repository_manager.destructive_guard",
        description="Refuse destructive git argv unless a snapshot-backed override is supplied.",
    )
    parser.add_argument("--path", default=None, help="working tree for the command")
    parser.add_argument(
        "--lane", default=None, help="lane identity for the snapshot ref"
    )
    parser.add_argument("--override", default=None, help="single-use operator token")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a fixed command is required after --")
    result = guard(
        command,
        path=args.path,
        lane=args.lane,
        override=args.override,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
