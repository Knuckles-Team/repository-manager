"""Validate tracked public owner specifications without network or workspace inputs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

REQUIRED = ("spec.md", "plan.md", "test-spec.md", "tasks.md")
FORBIDDEN = (
    re.compile(r"plans/", re.IGNORECASE),
    re.compile(r"\bgitlab\b", re.IGNORECASE),
    re.compile(r"\bhomelab\b", re.IGNORECASE),
    re.compile(r"(?:file://|/(?:home|Users|tmp|workspace)/)", re.IGNORECASE),
    re.compile(r"\b(?:EH-\d{1,3}|RF-\d{3})\b"),
)
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
LOCAL_HOST = re.compile(
    r"(?:^localhost$|\.local$|\.internal$|\.lan$|^127\.|^10\.|^192\.168\.)",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(
    r"\[(?:Describe|Exact|Record|ID|Feature name|repository name|Observable behavior|Measurable)",
    re.IGNORECASE,
)
CONTENT_MARKERS = {
    "spec.md": re.compile(r"requirement|acceptance|\bFR-\d+\b", re.IGNORECASE),
    "plan.md": re.compile(
        r"(?m)^# .*plan|architecture|design|interface|sequence", re.IGNORECASE
    ),
    "test-spec.md": re.compile(r"test|expected|proof", re.IGNORECASE),
    "tasks.md": re.compile(
        r"(?m)(^# .*tasks|^- \[[ xX]\]|^\|[^\n]*\bTask\b)", re.IGNORECASE
    ),
}


def tracked_specs(root: Path) -> list[Path]:
    return [
        path.relative_to(root) for path in (root / "specs").rglob("*") if path.is_file()
    ]


def _document_errors(path: Path, content: str) -> list[str]:
    errors = []
    if len(content.strip()) < 250:
        errors.append(f"{path}: add substantive design or proof")
    if PLACEHOLDER.search(content):
        errors.append(f"{path}: unresolved template placeholder")
    if not CONTENT_MARKERS[path.name].search(content):
        errors.append(f"{path}: missing required behavior, design, test, or tasks")
    return errors


def _spec_names(paths: set[Path]) -> list[str]:
    return sorted(
        {
            path.parts[1]
            for path in paths
            if len(path.parts) > 2 and path.parts[1] != "_template"
        }
    )


def _file_errors(root: Path, paths: set[Path], name: str) -> list[str]:
    errors = []
    for filename in REQUIRED:
        path = Path("specs") / name / filename
        if path not in paths:
            errors.append(f"{path}: missing required owner contract")
        else:
            errors.extend(
                _document_errors(path, (root / path).read_text(encoding="utf-8"))
            )
    return errors


def _contract_errors(root: Path, paths: set[Path]) -> list[str]:
    errors = [e for name in _spec_names(paths) for e in _file_errors(root, paths, name)]
    errors.extend(
        f"{path}: status.json is generated; it must not be tracked"
        for path in sorted(paths)
        if len(path.parts) == 3 and path.name == "status.json"
    )
    return errors


def _link_error(root: Path, path: Path, target: str) -> str | None:
    target = target.strip().split(" ", 1)[0].strip("<>")
    if target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme:
        public = parsed.scheme in ("http", "https") and parsed.hostname
        if not public or LOCAL_HOST.search(parsed.hostname or ""):
            return f"{path}: non-public link {target}"
        return None
    if target.startswith("/"):
        return f"{path}: absolute local link {target}"
    candidate = (root / path.parent / unquote(parsed.path)).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.exists():
        return f"{path}: broken relative link {target}"
    return None


def _reference_errors(root: Path, path: Path) -> list[str]:
    content = (root / path).read_text(encoding="utf-8")
    errors: list[str] = []
    if "_template" not in path.parts:
        errors.extend(
            f"{path}: private or local reference ({pattern.pattern})"
            for pattern in FORBIDDEN
            if pattern.search(content)
        )
    errors.extend(
        issue
        for target in LINK.findall(content)
        if (issue := _link_error(root, path, target))
    )
    return errors


def problems(root: Path) -> list[str]:
    paths = set(tracked_specs(root))
    errors = _contract_errors(root, paths)
    for path in sorted(paths):
        if path.suffix == ".md":
            errors.extend(_reference_errors(root, path))
    return errors


def main() -> int:
    errors = problems(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Public specs are self-contained and locally linked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
