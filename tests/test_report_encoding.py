"""Reports have a UTF-8 format independent of the host locale."""

import builtins
from pathlib import Path

import pytest

from repository_manager import models


def test_report_writer_uses_utf8_on_a_legacy_codepage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def legacy_open(path, mode, *, encoding=None):
        return builtins.open(path, mode, encoding=encoding or "cp1252")

    monkeypatch.setattr(models, "open", legacy_open, raising=False)
    report = tmp_path / "report.md"
    lines = ["# EPİSTEMIC-GRAPH", "Success ✅ | Failure ❌"]
    assert models._write_report_text_file(
        str(report), lines, error_message="synthetic report failure: %s"
    )
    assert report.read_bytes() == "\n".join(lines).encode("utf-8")
