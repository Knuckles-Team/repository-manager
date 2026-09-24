"""``repository-manager-governance`` — the lane and concept-ID command surface.

Moved from ``agent-utilities lane …`` / ``agent-utilities concept …`` with the
modules it drives (OQ-3, AUD-29): lane arbitration
(:mod:`repository_manager.governance.lanes`) and same-host concept-ID
reservation (:mod:`repository_manager.governance.concept_allocator`). The verbs,
flags, JSON output and exit codes are unchanged — only the program name moved —
so every documented ``lane lease … -- <command>`` recipe keeps its contract:
an unavailable lease is exit 75, a refusal exit 1, a usage error exit 2.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from repository_manager.governance.lanes import LaneArbitrationError


def build_parser() -> argparse.ArgumentParser:
    """The ``lane`` and ``concept`` sub-command tree."""
    p = argparse.ArgumentParser(
        prog="repository-manager-governance",
        description="Concurrent-lane arbitration and concept-ID reservation.",
    )
    p.add_argument("--json", action="store_true", help="compact JSON output")
    sub = p.add_subparsers(dest="command", required=True)

    # CONCEPT:AU-OS.governance.atomic-concept-id-reservation — atomic concept-ID reservation (offline/worktree entry point).
    cp = sub.add_parser("concept", help="reserve/list/release/reconcile concept ids")
    cp.add_argument(
        "concept_action",
        choices=["reserve", "release", "list", "reconcile", "resolve"],
    )
    cp.add_argument(
        "--session", default="", help="claiming session id (default host:pid)"
    )
    cp.add_argument("--design-doc", default="", help="design-doc path to record")
    cp.add_argument(
        "--id",
        dest="concept_id",
        default="",
        help="canonical OKF-CIS concept id for reserve, release, or resolve",
    )
    cp.add_argument(
        "--status", default="", help="filter for list (reserved/landed/expired)"
    )
    cp.add_argument("--ttl", type=int, default=86_400, help="reservation TTL seconds")
    cp.add_argument(
        "--repo", default="", help="repo root (default: the current working tree)"
    )

    # CONCEPT:AU-OS.governance.lane-arbitration-classes — concurrent-lane arbitration entry point.
    # Nested subparsers (not one positional choice) so `lease` can take a
    # REMAINDER command without swallowing the other actions' own flags.
    lp = sub.add_parser(
        "lane", help="concurrent-lane arbitration: isolation, leases, guards"
    )
    lane_sub = lp.add_subparsers(dest="lane_action", required=True)

    def _lane_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        parser = lane_sub.add_parser(name, help=help_text)
        parser.add_argument("--path", default="", help="working tree (default: cwd)")
        return parser

    _lane_parser("status", "this lane's isolation, partitions, and live leases")
    _lane_parser("env", "shell exports that give this lane its own build/test state")
    _lane_parser(
        "classify", "the shared-resource -> arbitration-class table"
    ).add_argument(
        "--resource", default="", help="one resource instead of the whole table"
    )
    guard_p = _lane_parser("guard", "refuse a mutation that would destroy other work")
    guard_p.add_argument("--operation", default="edit", help="what is being attempted")
    guard_p.add_argument(
        "--reset", default="", help="path a global actor wants to reset or discard"
    )
    guard_p.add_argument(
        "--owner", default="", help="the lane that owns the reset target"
    )
    guard_p.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="`-- <command>` to run under the guard (lease held across the mutation)",
    )
    _lane_parser(
        "park", "clean this tree for a moment WITHOUT touching the shared refs/stash"
    )
    _lane_parser("unpark", "restore what `lane park` set aside")
    bind_cargo_p = _lane_parser(
        "bind-cargo",
        "write .cargo/config.toml so cargo PARTITION binds with no export needed",
    )
    bind_cargo_p.add_argument(
        "--force",
        action="store_true",
        help="append the partition block even if .cargo/config.toml already exists",
    )
    lease_p = _lane_parser("lease", "hold a LEASE-class resource, or report its holder")
    lease_p.add_argument("--resource", default="", help="LEASE-class resource name")
    lease_p.add_argument("--operation", default="", help="why the lease is being taken")
    lease_p.add_argument(
        "--ttl",
        dest="lease_ttl",
        type=int,
        default=1_800,
        help="positive lease TTL seconds (maximum 86400)",
    )
    lease_p.add_argument(
        "--host-id",
        default="",
        help="host identity for host-scoped admission (blank uses the local hostname)",
    )
    lease_p.add_argument(
        "--host-health",
        default="",
        help="host health evidence; GPU admission requires exactly healthy",
    )
    lease_p.add_argument(
        "--reboot-verified",
        action="store_true",
        help="assert the required post-reboot host verification",
    )
    lease_p.add_argument(
        "--nvml-verified",
        action="store_true",
        help="assert that NVML has verified the GPU",
    )
    lease_p.add_argument(
        "--output-file",
        "--output",
        dest="output_file",
        default="",
        help="required file-backed stdout/stderr sink for heavy resources",
    )
    lease_p.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="`-- <command>` to run while holding the lease (omit to just report)",
    )

    return p


def _concept_resolve(args: argparse.Namespace) -> dict[str, Any]:
    """CONCEPT:AU-OS.governance.concept-id-canonicalization — validate and project a canonical OKF-CIS id."""
    from repository_manager.governance import concept_hierarchy as ch

    if not args.concept_id:
        return {"error": "resolve requires --id"}
    try:
        parsed = ch.parse_okf_id(args.concept_id)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "raw": parsed.raw,
        "canonical": parsed.canonical,
        "slug": parsed.slug,
        "pillar": parsed.pillar,
        "domain": parsed.domain,
        "concept": parsed.concept,
        "facets": list(parsed.facets),
        "path": parsed.path,
        "iri": parsed.iri,
    }


def _concept_release(
    args: argparse.Namespace, ca: Any, repo_root: Path | None
) -> dict[str, Any]:
    if not args.concept_id:
        return {"error": "release requires --id"}
    return {"released": ca.release_concept_id(args.concept_id, repo_root=repo_root)}


def _concept_reserve(
    args: argparse.Namespace, ca: Any, repo_root: Path | None
) -> dict[str, Any]:
    import uuid

    if not args.concept_id:
        return {"error": "reserve requires --id"}
    sid = args.session or f"session-{uuid.uuid4().hex}"
    return ca.reserve_concept_id(
        args.concept_id,
        session_id=sid,
        design_doc=args.design_doc or None,
        ttl_seconds=int(args.ttl),
        repo_root=repo_root,
    )


def _concept(args: argparse.Namespace) -> dict[str, Any]:
    """Same-host compatibility concept reservation against the file ledger.

    Separate-host callers must use graph-os' native concept authority; this
    legacy CLI path is intentionally not advertised as globally atomic.
    """
    from repository_manager.governance import concept_allocator as ca

    repo_root = Path(args.repo).expanduser().resolve() if args.repo else None
    action = args.concept_action
    if action == "list":
        return {
            "reservations": ca.list_reservations(
                repo_root=repo_root, status=args.status or None
            )
        }
    if action == "reconcile":
        return ca.reconcile(repo_root=repo_root)
    if action == "resolve":
        return _concept_resolve(args)
    if action == "release":
        return _concept_release(args, ca, repo_root)
    # reserve
    return _concept_reserve(args, ca, repo_root)


def _lane_env(lanes: Any, path: str | None) -> dict[str, Any]:
    parts = lanes.partitioned_paths(path)
    orphaned = lanes.orphaned_precommit_patches(path)
    return {
        "exports": {
            "CARGO_TARGET_DIR": str(parts.cargo_target_dir),
            "PYTEST_ADDOPTS": f"--basetemp={parts.pytest_basetemp}",
            "TMPDIR": str(parts.scratch_dir),
            "PRE_COMMIT_HOME": str(parts.precommit_home),
        },
        "stash_ref": parts.stash_ref,
        "note": (
            "never `git stash` — refs/stash is one ref shared by every "
            "worktree. To READ a pristine file while yours is dirty use "
            "`git show HEAD:<path>` (mutates nothing). To PARK work use a "
            f"scratch commit on your branch, or `lane park` -> {parts.stash_ref}"
        ),
        "precommit_home_note": (
            "PRE_COMMIT_HOME is also per-lane: a shared pre-commit store "
            "means a killed/OOMed/power-lost pre-commit orphans another "
            "lane's uncommitted work as an unreplayed patch file (D-OB-12), "
            "and the store's shared SQLite db.db raises `OperationalError: "
            "database is locked` under concurrent lanes. See D-ORC-37."
        ),
        "orphaned_precommit_patches": [
            p for p in orphaned if p["state"] in ("ORPHANED", "unknown")
        ],
    }


def _lane_bind_cargo(lanes: Any, path: str | None, force: bool) -> dict[str, Any]:
    try:
        return lanes.write_cargo_partition_config(path, force=force)
    except lanes.LaneArbitrationError as exc:
        return {"written": False, "refused": str(exc), "exit_code": 1}


def _lane_rule_summary(rule: Any, *, include_evidence: bool) -> dict[str, Any]:
    summary = {
        "name": rule.name,
        "class": rule.arbitration.value,
        "scope": getattr(rule, "scope", "repo"),
        "capacity": getattr(rule, "capacity", 1),
        "host_scoped": getattr(rule, "host_scoped", False),
        "requires_output_file": getattr(rule, "requires_output_file", False),
        "requires_healthy_host": getattr(rule, "requires_healthy_host", False),
        "output_max_bytes": getattr(rule, "output_max_bytes", None),
        "output_truncate": getattr(rule, "output_truncate", None),
    }
    if include_evidence:
        summary["mechanism"] = rule.mechanism
        summary["evidence"] = rule.evidence
    return summary


def _lane_classify_one(lanes: Any, rules: list[Any], resource: str) -> dict[str, Any]:
    rule = next((item for item in rules if item.name == resource), None)
    if rule is None:
        # Preserve the hard-error vocabulary from the governance module.
        lanes.resource_class(resource)
        raise AssertionError("resource_class unexpectedly returned")
    summary = _lane_rule_summary(rule, include_evidence=False)
    summary["resource"] = summary.pop("name")
    return summary


def _lane_classify(lanes: Any, resource: str | None) -> dict[str, Any]:
    rules = lanes.resource_rules()
    if resource:
        return _lane_classify_one(lanes, rules, resource)
    return {"resources": [_lane_rule_summary(r, include_evidence=True) for r in rules]}


def _lane_guard(
    lanes: Any, args: argparse.Namespace, path: str | None
) -> dict[str, Any]:
    command = [a for a in getattr(args, "command_args", []) if a != "--"]
    try:
        if args.reset:
            owner = args.owner or "unknown"
            if not command:
                lanes.require_resettable_tree(
                    args.reset, operation=args.operation, owner=owner
                )
                return {"allowed": True, "target": args.reset}
            # The mutation itself runs INSIDE the guard, so the tree cannot go
            # dirty between the check and the command — the whole point of the
            # single choke point.
            with lanes.guarded_tree_mutation(
                args.reset, operation=args.operation, owner=owner
            ) as scope:
                completed: subprocess.CompletedProcess[Any] = subprocess.run(
                    command, check=False
                )
            return {
                "allowed": True,
                "target": str(scope.tree),
                "command": command,
                "exit_code": completed.returncode,
            }
        scope = lanes.require_mutable_tree(path, operation=args.operation)
        return {"allowed": True, "lane": scope.lane, "tree": str(scope.tree)}
    except lanes.LaneArbitrationError as exc:
        return {"allowed": False, "refused": str(exc), "exit_code": 1}


def _run_bounded_command(command: list[str], sink: Any) -> int:
    """Stream combined child output through the bounded writer while it runs."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    output = process.stdout
    try:
        if output is None:  # pragma: no cover - Popen(PIPE) always supplies it
            if process.poll() is None:
                process.kill()
            process.wait()
            raise RuntimeError("bounded command has no stdout pipe")
        try:
            for chunk in iter(lambda: output.read(64 * 1024), b""):
                sink.write(chunk)
        except (LaneArbitrationError, OSError):
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        return process.wait()
    finally:
        if output is not None:
            output.close()


def _lane_command(args: argparse.Namespace) -> list[str]:
    return [a for a in getattr(args, "command_args", []) if a != "--"]


def _lease_output_settings(lanes: Any, held: dict[str, Any]) -> tuple[int, bool]:
    configured_max: object = held.get("output_max_bytes")
    max_bytes: object = (
        configured_max
        if configured_max is not None
        else getattr(lanes, "RESOURCE_OUTPUT_MAX_BYTES", 16 * 1024 * 1024)
    )
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise LaneArbitrationError("invalid bounded output byte cap")
    truncate: object = held.get("output_truncate")
    if not isinstance(truncate, bool):
        truncate = getattr(lanes, "RESOURCE_OUTPUT_TRUNCATE", True)
    if not isinstance(truncate, bool):
        raise LaneArbitrationError("invalid bounded output truncate policy")
    return max_bytes, truncate


def _run_lease_process(
    lanes: Any,
    command: list[str],
    output_file: str | None,
    held: dict[str, Any],
) -> subprocess.CompletedProcess[Any]:
    completed: subprocess.CompletedProcess[Any]
    if output_file:
        max_bytes, truncate = _lease_output_settings(lanes, held)
        with lanes.open_resource_output(
            output_file, max_bytes=max_bytes, truncate=truncate
        ) as sink:
            exit_code = _run_bounded_command(command, sink)
            completed = subprocess.CompletedProcess(command, exit_code)
    else:
        completed = subprocess.run(command, check=False)
    return completed


def _hold_lease_command(
    lanes: Any,
    args: argparse.Namespace,
    path: str | None,
    command: list[str],
    output_file: str | None,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[Any]]:
    with lanes.hold_lease(
        args.resource,
        operation=args.operation,
        ttl_seconds=args.lease_ttl,
        path=path,
        host_id=getattr(args, "host_id", "") or None,
        host_health=getattr(args, "host_health", "") or None,
        reboot_verified=getattr(args, "reboot_verified", False),
        nvml_verified=getattr(args, "nvml_verified", False),
        output_path=output_file,
    ) as held:
        completed = _run_lease_process(lanes, command, output_file, held)
    return held, completed


def _lane_lease_validation(
    lanes: Any, args: argparse.Namespace
) -> dict[str, Any] | None:
    if not args.resource:
        return {"exit_code": 2, "error": "lease requires --resource"}
    if lanes.resource_class(args.resource) is not lanes.ArbitrationClass.LEASE:
        return {
            "exit_code": 2,
            "error": f"{args.resource} is not a LEASE-class resource",
        }
    return None


def _lane_lease_status(
    lanes: Any,
    args: argparse.Namespace,
    path: str | None,
    command: list[str],
) -> dict[str, Any] | None:
    if not command:
        return {
            "resource": args.resource,
            "holder": lanes.lease_status(args.resource, path),
        }
    return None


def _run_lane_lease(
    lanes: Any,
    args: argparse.Namespace,
    path: str | None,
    command: list[str],
) -> dict[str, Any]:
    output_file = getattr(args, "output_file", "") or None
    try:
        held, completed = _hold_lease_command(lanes, args, path, command, output_file)
        return {
            "resource": args.resource,
            "held_by": held["lane"],
            "command": command,
            "output_file": str(output_file) if output_file else None,
            "exit_code": completed.returncode,
        }
    except lanes.LeaseUnavailable as exc:
        return {"deferred": True, "holder": exc.holder, "exit_code": 75}
    except lanes.LaneArbitrationError as exc:
        return {"deferred": False, "refused": str(exc), "exit_code": 1}


def _lane_lease(
    lanes: Any, args: argparse.Namespace, path: str | None
) -> dict[str, Any]:
    validation = _lane_lease_validation(lanes, args)
    if validation is not None:
        return validation
    command = _lane_command(args)
    status = _lane_lease_status(lanes, args, path, command)
    if status is not None:
        return status
    return _run_lane_lease(lanes, args, path, command)


def _lane(args: argparse.Namespace) -> dict[str, Any]:
    """Lane arbitration — the operator/agent surface over :mod:`~repository_manager.governance.lanes`.

    Every action here exists so the *safe* path is the convenient one: you get
    your isolated paths from ``env``, you run a contended operation through
    ``lease``, and ``guard`` refuses the mutation that would eat someone's work.
    """
    from repository_manager.governance import lanes

    path = args.path or None
    action = args.lane_action
    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "status": lambda: lanes.lane_report(path),
        "env": lambda: _lane_env(lanes, path),
        "park": lambda: lanes.park_worktree(path),
        "unpark": lambda: lanes.unpark_worktree(path),
        "bind-cargo": lambda: _lane_bind_cargo(lanes, path, args.force),
        "classify": lambda: _lane_classify(lanes, args.resource),
        "guard": lambda: _lane_guard(lanes, args, path),
        "lease": lambda: _lane_lease(lanes, args, path),
    }
    handler = handlers.get(action)
    if handler is None:
        return {"exit_code": 2, "error": f"unknown lane action: {action}"}
    return handler()


_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "concept": _concept,
    "lane": _lane,
}


def main(argv: list[str] | None = None) -> int:
    """Run one verb and print its JSON result; the exit code is the verdict.

    A refusal or a deferral must be actionable by a shell or hook, not just
    readable — the guard is worthless if ``&&`` still proceeds after it.
    """
    args = build_parser().parse_args(argv)
    out = _COMMAND_HANDLERS[args.command](args)
    print(json.dumps(out, indent=None if args.json else 2, default=str))
    return int(out.get("exit_code", 0))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
