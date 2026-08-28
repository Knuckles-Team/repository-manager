"""Characterization tests for check_workspace_docs.py::check_repository.

check_workspace_docs.py is a documentation/privacy gate (SPECIAL CASE in the
wave's PREAMBLE): it must catch a missing required doc file, a README missing
the governed-capability contract marker, an mkdocs nav gap, a broken local
Markdown link, and any of the UNSAFE_PATTERNS (a private .arpa hostname, a
host-specific home path, a TLS-verification bypass, a credential-like
literal). These tests plant each known-bad input, confirm it is reported,
then a fully well-formed repository fixture and confirm zero errors -- run
unmodified before and after the extract-method refactor of check_repository.
There is no pre-existing test module for this script; this file is the
primary characterization baseline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "check_workspace_docs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_workspace_docs_char", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cwd_mod = _load_module()

GOVERNED_MARKER = "<!-- GOVERNED-CAPABILITY:START -->"

REQUIRED_NAV = """
nav:
  - Home: index.md
  - Install: installation.md
  - Configure: configuration.md
  - Deploy: deployment.md
  - Usage: usage.md
"""


def _make_well_formed_repository(root: Path) -> Path:
    repo = root / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (repo / "README.md").write_text(f"# Repo\n\n{GOVERNED_MARKER}\ndone\n", encoding="utf-8")
    (repo / "mkdocs.yml").write_text(REQUIRED_NAV, encoding="utf-8")
    for name in ("index", "installation", "configuration", "deployment", "usage"):
        (docs / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
    return repo


def test_well_formed_repository_has_no_errors(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    assert cwd_mod.check_repository(repo) == []


def test_missing_required_file_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").unlink()
    errors = cwd_mod.check_repository(repo)
    assert any("missing required documentation: docs/usage.md" in e for e in errors)


def test_readme_missing_governed_capability_marker_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "README.md").write_text("# Repo\nno marker here\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("governed capability contract" in e for e in errors)


def test_mkdocs_nav_target_missing_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").unlink()
    (repo / "mkdocs.yml").write_text(REQUIRED_NAV, encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("nav target does not exist: usage.md" in e for e in errors)


def test_mkdocs_nav_omitting_required_page_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "mkdocs.yml").write_text(
        "nav:\n  - Home: index.md\n  - Install: installation.md\n"
        "  - Configure: configuration.md\n  - Deploy: deployment.md\n",
        encoding="utf-8",
    )
    errors = cwd_mod.check_repository(repo)
    assert any("mkdocs nav omits required page: usage.md" in e for e in errors)


def test_arpa_hostname_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text("connect to host.arpa now\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("private .arpa hostname" in e for e in errors)


def test_host_specific_home_path_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text("path is /home/apps/foo\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("host-specific home path" in e for e in errors)


def test_tls_verification_bypass_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text("curl --insecure https://x\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("TLS verification bypass" in e for e in errors)


def test_credential_like_literal_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text("token: ghp_abcdefgh12345678\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("credential-like literal" in e for e in errors)


def test_broken_local_markdown_link_is_reported(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text("[link](missing.md)\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert any("missing local link: missing.md" in e for e in errors)


def test_external_and_anchor_links_are_not_flagged(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text(
        "[ext](https://example.com) [anchor](#section) [mail](mailto:a@b.com)\n",
        encoding="utf-8",
    )
    errors = cwd_mod.check_repository(repo)
    assert errors == []


def test_valid_local_markdown_link_is_not_flagged(tmp_path):
    repo = _make_well_formed_repository(tmp_path)
    (repo / "docs" / "usage.md").write_text("[index](index.md)\n", encoding="utf-8")
    errors = cwd_mod.check_repository(repo)
    assert errors == []


def test_readme_missing_entirely_reports_both_missing_file_and_no_marker_crash_free(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    errors = cwd_mod.check_repository(repo)
    assert any("missing required documentation: README.md" in e for e in errors)
    # README.md does not exist -> readme.is_file() is False -> the governed
    # capability check must not also fire (no crash reading a missing file).
    assert not any("governed capability contract" in e for e in errors)
