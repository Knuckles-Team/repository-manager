#!/usr/bin/env python3
"""Validate documentation contracts for the manifest-listed agent fleet.

The check is intentionally dependency-free so it can run before MkDocs or the
Python packages are installed.  It validates repository discovery from the
packaged ``workspace.yml``, required operator pages, MkDocs navigation, local
Markdown links, and a small set of privacy/TLS anti-patterns.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


AGENT_SECTION = re.compile(r"^      agents:\s*$")
TOP_LEVEL_SECTION = re.compile(r"^  [a-zA-Z0-9_-]+:\s*$")
GITHUB_REPOSITORY = re.compile(r"github\.com/[^/]+/([^/]+?)(?:\.git)?[\"']?$")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
NAV_PAGE = re.compile(r"^\s*-\s+[^:]+:\s*[\"']?([^\"']+\.md)[\"']?\s*$")

REQUIRED_FILES = (
    "README.md",
    "mkdocs.yml",
    "docs/index.md",
    "docs/installation.md",
    "docs/configuration.md",
    "docs/deployment.md",
    "docs/usage.md",
)

UNSAFE_PATTERNS = (
    ("private .arpa hostname", re.compile(r"(?i)\b[a-z0-9.-]+\.arpa\b")),
    (
        "host-specific home path",
        re.compile(
            r"(?i)(?:[a-z]:[\\/]users[\\/](?:user|username|[^<%${}\\/]+)[\\/]"
            r"|/mnt/[a-z]/users/|/home/(?:user|username|apps)/)"
        ),
    ),
    (
        "TLS verification bypass",
        re.compile(
            r"(?i)(?:--insecure\b|verify(?:_ssl|_tls)?\s*[=:]\s*[\"']?false\b|"
            r"[A-Z0-9_]*(?:SSL|TLS|VERIFY)[A-Z0-9_]*[\"']?\s*[:=]\s*[\"']?false\b)"
        ),
    ),
    (
        "credential-like literal",
        re.compile(r"(?i)\b(?:sk-lf|pk-lf|glpat|ghp)_[a-z0-9_-]{8,}\b"),
    ),
)


def manifest_repositories(manifest: Path) -> list[str]:
    """Return GitHub repository names from the agents section only."""

    repositories: list[str] = []
    in_agents = False
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if AGENT_SECTION.match(raw_line):
            in_agents = True
            continue
        if in_agents and TOP_LEVEL_SECTION.match(raw_line):
            break
        if not in_agents or "url:" not in raw_line:
            continue
        match = GITHUB_REPOSITORY.search(raw_line.strip())
        if match:
            repositories.append(match.group(1))
    return repositories


def markdown_files(repository: Path) -> list[Path]:
    files = [repository / "README.md"]
    docs = repository / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.rglob("*.md")))
    return [path for path in files if path.is_file()]


def check_repository(repository: Path) -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        if not (repository / relative).is_file():
            errors.append(f"missing required documentation: {relative}")

    readme = repository / "README.md"
    if readme.is_file() and "<!-- GOVERNED-CAPABILITY:START -->" not in readme.read_text(
        encoding="utf-8"
    ):
        errors.append("README is missing the governed capability contract")

    mkdocs = repository / "mkdocs.yml"
    if mkdocs.is_file():
        nav_pages: set[str] = set()
        for line_number, line in enumerate(
            mkdocs.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = NAV_PAGE.match(line)
            if not match:
                continue
            page = match.group(1)
            nav_pages.add(page)
            if not (repository / "docs" / page).is_file():
                errors.append(f"mkdocs.yml:{line_number}: nav target does not exist: {page}")
        for required_page in (
            "index.md",
            "installation.md",
            "configuration.md",
            "deployment.md",
            "usage.md",
        ):
            if required_page not in nav_pages:
                errors.append(f"mkdocs nav omits required page: {required_page}")

    for document in markdown_files(repository):
        relative_document = document.relative_to(repository)
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for label, pattern in UNSAFE_PATTERNS:
                if pattern.search(line):
                    errors.append(f"{relative_document}:{line_number}: {label}")

            for raw_target in MARKDOWN_LINK.findall(line):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(
                    ("http://", "https://", "mailto:", "#", "data:", "/")
                ):
                    continue
                local_target = target.split("#", 1)[0].split("?", 1)[0]
                if local_target and not (document.parent / local_target).exists():
                    errors.append(
                        f"{relative_document}:{line_number}: missing local link: {target}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    default_agents_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--agents-root", type=Path, default=default_agents_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_agents_root / "repository-manager" / "repository_manager" / "workspace.yml",
    )
    args = parser.parse_args()

    repositories = manifest_repositories(args.manifest)
    if not repositories:
        print("documentation contract failed: no agent repositories in manifest")
        return 1

    failures: dict[str, list[str]] = {}
    for name in repositories:
        repository = args.agents_root / name
        if not repository.is_dir():
            failures[name] = ["manifest-listed repository is not present"]
            continue
        errors = check_repository(repository)
        if errors:
            failures[name] = errors

    if failures:
        print(f"documentation contract failed for {len(failures)} repository(s)")
        for name, errors in failures.items():
            print(f"[{name}]")
            for error in errors:
                print(f"  - {error}")
        return 1

    print(f"documentation contract passed for {len(repositories)} repository(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
