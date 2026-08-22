"""``fail_fast_audit`` — a STATIC scanner, proven against known-bad and
known-good ``.pre-commit-config.yaml`` fixtures. Per this workspace's own
lesson (gates report more coverage than they have until proven against a
known-bad input), every violation case below has a matching "and the
compliant version of the same hook produces nothing" case, so a gate that
only ever fires (or only ever stays quiet) would be caught here.
"""

from __future__ import annotations

from pathlib import Path

from repository_manager.fail_fast_audit import check_fleet, check_repo, dispatch

# The real fleet boilerplate (agent-packages/agents/{ansible-tower-mcp,
# archivebox-api,arr-mcp}/.pre-commit-config.yaml, byte-identical across all
# three, captured 2026-08-21) -- used as the compliant control case: it must
# never trip this scanner, since it deliberately carries no fail-fast flag.
_FLEET_BOILERPLATE_PYTEST_ENTRY = (
    "bash -c 'repo=$(dirname \"$(git rev-parse --path-format=absolute "
    '--git-common-dir)"); if [ -n "$AGENT_UTILITIES_ROOT" ]; then '
    'root="$AGENT_UTILITIES_ROOT"; else au_d="$repo"; root=""; while '
    '[ "$au_d" != "/" ]; do if [ -d "$au_d/agent-utilities/scripts" ]; then '
    'root="$au_d/agent-utilities"; break; fi; au_d=$(dirname "$au_d"); done; '
    'fi; if [ -d "$root" ] && [ "$repo" != "$root" ]; then mkdir -p '
    ".uv-workspace-siblings && ln -sfn \"$root\" "
    '.uv-workspace-siblings/agent-utilities; fi; test_target="tests"; for d '
    'in tests/unit test/unit tests test; do if [ -d "$d" ]; then '
    'test_target="$d"; break; fi; done; if [ -f uv.lock ]; then uv run '
    '--all-extras pytest "$test_target" -q --tb=short -m "not slow" '
    '--timeout=60; else pytest "$test_target" -q --tb=short -m "not slow" '
    "--timeout=60; fi'"
)


def _write_config(repo_path: Path, hooks_entries: list[tuple[str, str]]) -> None:
    """Build a minimal ``.pre-commit-config.yaml`` with one local hook per
    ``(hook_id, entry)`` pair, using an explicit YAML block scalar so no
    hand-escaping of the entry text is needed in the fixture itself."""

    repo_path.mkdir(parents=True, exist_ok=True)
    lines = ["repos:", "- repo: local", "  hooks:"]
    for hook_id, entry in hooks_entries:
        lines.append(f"  - id: {hook_id}")
        lines.append(f"    name: {hook_id}")
        lines.append("    entry: |-")
        for entry_line in entry.splitlines() or [""]:
            lines.append(f"      {entry_line}")
        lines.append("    language: system")
        lines.append("    pass_filenames: false")
    (repo_path / ".pre-commit-config.yaml").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# pytest
# --------------------------------------------------------------------------- #


def test_pytest_bare_dash_x_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", 'pytest "tests" -x -q --tb=short')])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "-x" in violation.flags
    assert violation.hook_id == "pytest"


def test_pytest_exitfirst_long_form_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", "pytest tests --exitfirst")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "--exitfirst" in violation.flags


def test_pytest_maxfail_nonzero_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", "pytest tests --maxfail=3")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "--maxfail=3" in violation.flags


def test_pytest_maxfail_two_token_form_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", "pytest tests --maxfail 3")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "--maxfail 3" in violation.flags


def test_pytest_maxfail_zero_is_not_flagged(tmp_path: Path) -> None:
    """``--maxfail=0`` means "no limit" to pytest -- not a truncation."""

    _write_config(tmp_path, [("pytest", "pytest tests --maxfail=0 -q")])
    assert check_repo(tmp_path) == []


def test_pytest_bundled_short_carrying_x_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", "pytest tests -xvs")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "-xvs" in violation.flags


def test_pytest_plain_run_with_no_fail_fast_flag_is_clean(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", 'pytest "tests" -q --tb=short -m "not slow"')])
    assert check_repo(tmp_path) == []


def test_pytest_behind_uv_run_launcher_is_still_recognized(tmp_path: Path) -> None:
    _write_config(
        tmp_path, [("pytest", 'uv run --all-extras pytest "tests" -x -q --tb=short')]
    )
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "-x" in violation.flags


# --------------------------------------------------------------------------- #
# cargo
# --------------------------------------------------------------------------- #


def test_cargo_test_missing_no_fail_fast_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("cargo-test", "cargo test --workspace")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "cargo"
    assert violation.flags == ("--no-fail-fast",)


def test_cargo_nextest_run_missing_no_fail_fast_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("cargo-test", "cargo nextest run --workspace")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "cargo"


def test_cargo_test_with_no_fail_fast_is_clean(tmp_path: Path) -> None:
    _write_config(tmp_path, [("cargo-test", "cargo test --workspace --no-fail-fast")])
    assert check_repo(tmp_path) == []


def test_cargo_build_and_check_are_never_flagged(tmp_path: Path) -> None:
    """Proves the cargo recogniser is scoped to test/nextest run -- a build
    or check invocation carrying no test flags at all must never trip this."""

    _write_config(
        tmp_path,
        [
            ("cargo-build", "cargo build --release"),
            ("cargo-check", "cargo check --workspace"),
        ],
    )
    assert check_repo(tmp_path) == []


# --------------------------------------------------------------------------- #
# go
# --------------------------------------------------------------------------- #


def test_go_test_failfast_single_dash_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("go-test", "go test -failfast ./...")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "go"
    assert "-failfast" in violation.flags


def test_go_test_failfast_double_dash_is_flagged(tmp_path: Path) -> None:
    _write_config(tmp_path, [("go-test", "go test --failfast ./...")])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "go"


def test_go_test_without_failfast_is_clean(tmp_path: Path) -> None:
    _write_config(tmp_path, [("go-test", "go test ./...")])
    assert check_repo(tmp_path) == []


# --------------------------------------------------------------------------- #
# Statement segmentation inside a wrapping bash -c script.
# --------------------------------------------------------------------------- #


def test_flag_hidden_inside_a_conditional_bash_c_branch_is_found(
    tmp_path: Path,
) -> None:
    entry = (
        "bash -c 'if [ -f uv.lock ]; then uv run --all-extras pytest tests -x "
        "-q; else pytest tests -q; fi'"
    )
    _write_config(tmp_path, [("pytest", entry)])
    [violation] = check_repo(tmp_path)
    assert violation.tool == "pytest"
    assert "-x" in violation.flags


def test_the_real_fleet_boilerplate_entry_is_clean(tmp_path: Path) -> None:
    """The actual byte-identical fleet entry (no fail-fast flag anywhere in
    either branch) must produce zero violations -- the compliant control."""

    _write_config(tmp_path, [("pytest", _FLEET_BOILERPLATE_PYTEST_ENTRY)])
    assert check_repo(tmp_path) == []


def test_one_violation_per_statement_even_with_multiple_hooks(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        [
            ("pytest", "pytest tests -x"),
            ("cargo-test", "cargo test"),
            ("lint", "ruff check ."),
        ],
    )
    violations = check_repo(tmp_path)
    tools = sorted(v.tool for v in violations)
    assert tools == ["cargo", "pytest"]


# --------------------------------------------------------------------------- #
# Missing / unreadable config.
# --------------------------------------------------------------------------- #


def test_missing_config_file_reports_zero_violations(tmp_path: Path) -> None:
    assert check_repo(tmp_path) == []


def test_unparseable_yaml_reports_zero_violations_not_a_crash(tmp_path: Path) -> None:
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: [this is: not: valid")
    assert check_repo(tmp_path) == []


# --------------------------------------------------------------------------- #
# check_fleet / dispatch
# --------------------------------------------------------------------------- #


def test_check_fleet_aggregates_by_repo(tmp_path: Path) -> None:
    clean_repo = tmp_path / "clean"
    dirty_repo = tmp_path / "dirty"
    _write_config(clean_repo, [("pytest", "pytest tests -q")])
    _write_config(dirty_repo, [("pytest", "pytest tests -x")])

    results = check_fleet([clean_repo, dirty_repo])
    assert results[str(clean_repo)] == []
    assert len(results[str(dirty_repo)]) == 1


def test_dispatch_check_ok_true_when_clean(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", "pytest tests -q")])
    result = dispatch("check", repo_path=str(tmp_path))
    assert result == {"ok": True, "repo_path": str(tmp_path), "violations": []}


def test_dispatch_check_ok_false_when_violations_found(tmp_path: Path) -> None:
    _write_config(tmp_path, [("pytest", "pytest tests -x")])
    result = dispatch("check", repo_path=str(tmp_path))
    assert result["ok"] is False
    assert len(result["violations"]) == 1
    assert result["violations"][0]["tool"] == "pytest"


def test_dispatch_check_requires_repo_path() -> None:
    result = dispatch("check")
    assert result == {"ok": False, "error": "check requires repo_path"}


def test_dispatch_check_fleet_requires_repo_paths() -> None:
    result = dispatch("check_fleet")
    assert result == {"ok": False, "error": "check_fleet requires repo_paths"}


def test_dispatch_check_fleet_reports_totals(tmp_path: Path) -> None:
    clean_repo = tmp_path / "clean"
    dirty_repo = tmp_path / "dirty"
    _write_config(clean_repo, [("pytest", "pytest tests -q")])
    _write_config(dirty_repo, [("pytest", "pytest tests -x")])

    result = dispatch("check_fleet", repo_paths=[str(clean_repo), str(dirty_repo)])
    assert result["ok"] is False
    assert result["repos_checked"] == 2
    assert result["repos_with_violations"] == 1
    assert len(result["violations"]) == 1


def test_dispatch_unknown_action() -> None:
    result = dispatch("bogus")
    assert result == {"ok": False, "error": "unknown action: bogus"}
