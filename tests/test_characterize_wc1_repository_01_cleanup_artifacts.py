"""Characterization tests for ``Git.cleanup_artifacts`` (WC1-REPOSITORY-01).

The only pre-existing reference to ``cleanup_artifacts``
(``tests/test_repository_manager.py:690``) monkeypatches it to a no-op --
zero real coverage of its filesystem-walk logic. These tests exercise the
real function against a real temp directory tree and pin its observable
behavior (which files/dirs survive vs. get removed) before this lane's
extract-method refactor, then are re-run unmodified after to require an
identical result.
"""

from repository_manager.repository_manager import Git


def _git(tmp_path):
    return Git(path=str(tmp_path))


def test_cleanup_artifacts_missing_dir_is_a_noop(tmp_path):
    missing = tmp_path / "does-not-exist"
    # Must not raise.
    assert _git(tmp_path).cleanup_artifacts(str(missing)) is None


def test_cleanup_artifacts_removes_file_patterns_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    targets = [
        tmp_path / "coverage.xml",
        tmp_path / ".coverage",
        tmp_path / "sub" / "run.log",
        tmp_path / "sub" / "knowledge_graph.db-wal",
        tmp_path / "sub" / "failed_tests.txt",
    ]
    for f in targets:
        f.write_text("x")
    keep = tmp_path / "sub" / "keep.py"
    keep.write_text("x")

    _git(tmp_path).cleanup_artifacts(str(tmp_path))

    for f in targets:
        assert not f.exists(), f"{f} should have been removed"
    assert keep.exists()


def test_cleanup_artifacts_removes_directory_patterns(tmp_path):
    pytest_cache = tmp_path / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "README.md").write_text("x")
    htmlcov = tmp_path / "sub" / "htmlcov"
    htmlcov.mkdir(parents=True)
    (htmlcov / "index.html").write_text("x")
    agent_data = tmp_path / "agent_data"
    agent_data.mkdir()
    (agent_data / "state.db").write_text("x")

    _git(tmp_path).cleanup_artifacts(str(tmp_path))

    assert not pytest_cache.exists()
    assert not htmlcov.exists()
    assert not agent_data.exists()


def test_cleanup_artifacts_ignores_venv_node_modules_git(tmp_path):
    for ignored in (".venv", "node_modules", ".git"):
        d = tmp_path / ignored
        d.mkdir()
        (d / "coverage.xml").write_text("x")  # would match a removal pattern

    _git(tmp_path).cleanup_artifacts(str(tmp_path))

    for ignored in (".venv", "node_modules", ".git"):
        assert (tmp_path / ignored / "coverage.xml").exists()


def test_cleanup_artifacts_root_transient_scripts_removed_only_at_root(tmp_path):
    root_script = tmp_path / "debug_thing.py"
    root_script.write_text("x")
    (tmp_path / "sub").mkdir()
    nested_script = tmp_path / "sub" / "debug_thing.py"
    nested_script.write_text("x")

    _git(tmp_path).cleanup_artifacts(str(tmp_path))

    assert not root_script.exists()
    assert nested_script.exists()  # only root-level scripts are targeted


def test_cleanup_artifacts_root_nonstandard_txt_removed_but_requirements_kept(tmp_path):
    stray = tmp_path / "notes.txt"
    stray.write_text("x")
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("x")
    reqs_dev = tmp_path / "requirements-dev.txt"
    reqs_dev.write_text("x")

    _git(tmp_path).cleanup_artifacts(str(tmp_path))

    assert not stray.exists()
    assert reqs.exists()
    assert reqs_dev.exists()


def test_cleanup_artifacts_unrelated_files_survive(tmp_path):
    keep = tmp_path / "README.md"
    keep.write_text("x")
    keep_py = tmp_path / "module.py"
    keep_py.write_text("x")

    _git(tmp_path).cleanup_artifacts(str(tmp_path))

    assert keep.exists()
    assert keep_py.exists()
