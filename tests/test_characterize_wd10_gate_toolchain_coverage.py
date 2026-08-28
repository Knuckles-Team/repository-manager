"""Characterization tests for check_gate_toolchain_coverage.py (wD10-C-MISC).

This is a security/gate scanner (SPECIAL CASE in the wave's PREAMBLE): a
`language: system` pre-commit hook or mergequeue gate that invokes a binary
outside both TOOLCHAIN_BINARIES and BASE_PROVIDED must FAIL LOUD
(UNCLASSIFIED) rather than silently pass. These tests plant that known-bad
input (an unrecognised command), confirm it FAILs, then a well-formed
equivalent (recognised toolchain binaries only) and confirm it PASSes -- both
before and after the extract-method refactor of main/_command_tokens/
_iter_precommit_system_entries/_iter_mergequeue_entries. There is no
pre-existing test module for this script; this file is the primary
characterization baseline.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_gate_toolchain_coverage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gate_toolchain_char", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gtc = _load_module()


# ---------------------------------------------------------------------------
# _command_tokens
# ---------------------------------------------------------------------------


def test_command_tokens_simple_command():
    assert gtc._command_tokens("node script.js") == ["node"]


def test_command_tokens_recurses_into_bash_dash_c():
    tokens = gtc._command_tokens("bash -c 'node scripts/no_fabrication_gate.mjs'")
    assert "bash" in tokens
    assert "node" in tokens


def test_command_tokens_separator_puts_next_token_in_command_position():
    tokens = gtc._command_tokens("cargo build && pnpm install")
    assert "cargo" in tokens
    assert "pnpm" in tokens


def test_command_tokens_assignment_prefix_is_skipped_not_a_command():
    tokens = gtc._command_tokens("FOO=bar node script.js")
    assert tokens == ["node"]


def test_command_tokens_escaped_parens_do_not_trigger_command_position():
    tokens = gtc._command_tokens(
        r'find . -type f \( -name "mcp_server.py" -o -name "agent_server.py" \)'
    )
    # "-name" must never appear as a command-position token here.
    assert "find" in tokens
    assert "-name" not in tokens


def test_command_tokens_real_paren_subshell_puts_next_in_command_position():
    tokens = gtc._command_tokens("(cd sub && node build.js)")
    assert "node" in tokens


def test_command_tokens_unbalanced_quotes_reports_tokenize_failure():
    tokens = gtc._command_tokens("node 'unterminated")
    assert len(tokens) == 1
    assert tokens[0].startswith("<<TOKENIZE-FAILED:")


# ---------------------------------------------------------------------------
# _iter_precommit_system_entries
# ---------------------------------------------------------------------------


def test_iter_precommit_system_entries_only_language_system(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text(
        textwrap.dedent(
            """
            repos:
              - repo: local
                hooks:
                  - id: system-hook
                    language: system
                    entry: node script.js
                  - id: python-managed-hook
                    language: python
                    entry: some-tool
            """
        ).strip(),
        encoding="utf-8",
    )
    entries = gtc._iter_precommit_system_entries(tmp_path / ".pre-commit-config.yaml")
    assert entries == [("system-hook", "node script.js")]


def test_iter_precommit_system_entries_malformed_yaml_warns_and_returns_empty(
    tmp_path, capsys
):
    bad = tmp_path / ".pre-commit-config.yaml"
    bad.write_text("repos: [", encoding="utf-8")
    entries = gtc._iter_precommit_system_entries(bad)
    assert entries == []
    assert "WARNING" in capsys.readouterr().err


def test_iter_precommit_system_entries_non_mapping_document_returns_empty(tmp_path):
    doc = tmp_path / ".pre-commit-config.yaml"
    doc.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert gtc._iter_precommit_system_entries(doc) == []


# ---------------------------------------------------------------------------
# _iter_mergequeue_entries
# ---------------------------------------------------------------------------


def test_iter_mergequeue_entries_unwraps_bash_dash_c(tmp_path):
    doc = tmp_path / ".mergequeue.yaml"
    doc.write_text(
        textwrap.dedent(
            """
            gates:
              - name: fmt
                command: ["bash", "-c", "cargo fmt --check"]
            """
        ).strip(),
        encoding="utf-8",
    )
    entries = gtc._iter_mergequeue_entries(doc)
    assert entries == [("fmt", "cargo fmt --check")]


def test_iter_mergequeue_entries_argv_list_form_is_joined(tmp_path):
    doc = tmp_path / ".mergequeue.yaml"
    doc.write_text(
        textwrap.dedent(
            """
            gates:
              - name: build
                command: ["cargo", "build", "--release"]
            """
        ).strip(),
        encoding="utf-8",
    )
    entries = gtc._iter_mergequeue_entries(doc)
    assert entries == [("build", "cargo build --release")]


def test_iter_mergequeue_entries_skips_non_list_command(tmp_path):
    doc = tmp_path / ".mergequeue.yaml"
    doc.write_text(
        textwrap.dedent(
            """
            gates:
              - name: bad
                command: "not-a-list"
            """
        ).strip(),
        encoding="utf-8",
    )
    assert gtc._iter_mergequeue_entries(doc) == []


# ---------------------------------------------------------------------------
# main (end-to-end CLI, --dockerfile static mode)
# ---------------------------------------------------------------------------


def _build_fleet(tmp_path: Path, hook_entry: str) -> Path:
    fleet_root = tmp_path / "fleet"
    (fleet_root / "agent-utilities").mkdir(parents=True)
    (fleet_root / "agents" / "repo-one").mkdir(parents=True)
    (fleet_root / "agents" / "repo-one" / ".pre-commit-config.yaml").write_text(
        textwrap.dedent(
            f"""
            repos:
              - repo: local
                hooks:
                  - id: a-hook
                    language: system
                    entry: {hook_entry}
            """
        ).strip(),
        encoding="utf-8",
    )
    return fleet_root


def _run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_main_fails_on_unclassified_command(tmp_path):
    fleet_root = _build_fleet(tmp_path, "totally-unknown-binary --flag")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")

    result = _run_gate(
        "--fleet-root", str(fleet_root), "--dockerfile", str(dockerfile)
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "UNCLASSIFIED" in result.stdout
    assert "totally-unknown-binary" in result.stdout


def test_main_passes_when_only_base_provided_binaries_are_used(tmp_path):
    fleet_root = _build_fleet(tmp_path, "echo hello")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")

    result = _run_gate(
        "--fleet-root", str(fleet_root), "--dockerfile", str(dockerfile)
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_main_reports_missing_toolchain_binary(tmp_path):
    fleet_root = _build_fleet(tmp_path, "node script.js")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")

    result = _run_gate(
        "--fleet-root", str(fleet_root), "--dockerfile", str(dockerfile)
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "MISSING" in result.stdout
    assert "node" in result.stdout


def test_main_reports_ok_when_dockerfile_provisions_toolchain(tmp_path):
    fleet_root = _build_fleet(tmp_path, "node script.js")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.12-slim\nENV NODE_HOME=/opt/node\n", encoding="utf-8"
    )

    result = _run_gate(
        "--fleet-root", str(fleet_root), "--dockerfile", str(dockerfile)
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[OK" in result.stdout


def test_main_rejects_mutually_exclusive_image_and_dockerfile(tmp_path):
    fleet_root = _build_fleet(tmp_path, "echo hello")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")

    result = _run_gate(
        "--fleet-root",
        str(fleet_root),
        "--dockerfile",
        str(dockerfile),
        "--image",
        "some:tag",
    )

    assert result.returncode != 0
    assert "mutually exclusive" in (result.stdout + result.stderr)


def test_main_rejects_nonexistent_fleet_root(tmp_path):
    result = _run_gate(
        "--fleet-root",
        str(tmp_path / "does-not-exist"),
        "--dockerfile",
        str(tmp_path / "Dockerfile"),
    )
    assert result.returncode != 0
