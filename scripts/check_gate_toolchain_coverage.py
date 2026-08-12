#!/usr/bin/env python3
"""Fail the build if the repository-manager image cannot run a fleet gate.

D-MQ-8 (see ``docker/Dockerfile``) documented one instance of this defect
class: ``rm__merge_queue``'s gate subprocesses run *inside* this container,
so any repo whose ``.pre-commit-config.yaml``/``.mergequeue.yaml`` declares a
toolchain the image lacks makes every candidate for that repo auto-reject in
seconds with a ``PermissionError``/``FileNotFoundError`` that reads exactly
like a real gate failure. That was fixed once for ``cargo`` and never
mirrored for ``node``/``pnpm`` (the concrete defect this script's own
Dockerfile change closes) -- a hand-maintained package list will drift again
the next time a repo adopts a new toolchain, so this script makes the
requirement an INVARIANT instead: it derives what the image must provide
from the fleet's own gate declarations and fails if the image cannot
provide it, rather than asking someone to remember to update a list.

What it does
------------
1. Scans every ``.pre-commit-config.yaml`` (hooks with ``language: system``
   only -- any other ``language:`` is a pre-commit-managed environment that
   pre-commit builds for itself) and every ``.mergequeue.yaml`` (every gate
   ``command:`` -- there is no ``language`` concept there; every gate is a
   raw subprocess) under ``--fleet-root``.
2. Extracts, per entry, which of a known ALLOWLIST of toolchain binaries
   (``TOOLCHAIN_BINARIES`` below) it invokes in *command position* -- the
   first word of the entry, or the first word after a shell command
   separator (``;``, ``&&``, ``||``, ``|``, ``do``, ``then``, ``else``,
   ``(``). This is deliberately NOT a shell parser: it is a conservative
   heuristic (see ``_command_tokens``) that only asks "does a known
   toolchain name appear where a command belongs", which is enough to
   catch the ``node``/``pnpm``/``cargo``/``docker`` shape of hook this
   fleet actually writes.
3. Classifies every command-position token as one of:
     - a member of ``TOOLCHAIN_BINARIES``  -> REQUIRED, checked against the
       image/Dockerfile.
     - a member of ``BASE_PROVIDED``       -> always fine (guaranteed by
       this Dockerfile's own base layers -- python, git, uv, coreutils).
     - an environment-variable assignment (``FOO=bar cmd``) or a shell
       keyword -> not a command, skipped.
     - anything else                       -> UNCLASSIFIED, and the gate
       FAILS. Under-detection (silently assuming an unrecognised command is
       fine) is exactly the failure mode that let the node/pnpm gap ship
       for as long as it did -- this script never guesses "probably fine"
       on an unrecognised command, it stops and asks a human to either add
       it to ``TOOLCHAIN_BINARIES``/``BASE_PROVIDED`` or explain why it
       does not need the image.
4. Checks whether each REQUIRED binary is actually available, in one of two
   modes:
     - ``--image TAG``      (CI mode)    -- ``docker run --rm TAG sh -c
       'command -v X'`` for each required X, against the real built image.
     - ``--dockerfile PATH`` (local mode) -- static pattern match over the
       Dockerfile text for a known provisioning signature per binary
       (``DOCKERFILE_PROVISION_SIGNATURES``). Cheaper, runs with no Docker
       daemon, but is itself a heuristic over what the Dockerfile author
       intended to install -- the ``--image`` mode against a real build is
       the authoritative check.

Explicit scope boundary: this only sees command TEXT in these two file
types. A toolchain pulled in transitively (e.g. ``cmake`` via a Rust
crate's ``build.rs``, never typed as a literal token in any hook/gate
entry) is invisible to it by construction -- that class of requirement has
to be verified by actually running the gate against the image, which is
what ``docker/Dockerfile``'s own D-MQ-8 comment and this script's README
wiring both call for.

Usage::

    # local mode -- no Docker daemon needed, run from repository-manager/
    python3 scripts/check_gate_toolchain_coverage.py

    # CI mode -- against a real built image
    python3 scripts/check_gate_toolchain_coverage.py --image repository-manager:local

    # explicit fleet root / Dockerfile (defaults resolve relative to this file)
    python3 scripts/check_gate_toolchain_coverage.py \\
        --fleet-root ../../.. --dockerfile docker/Dockerfile
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - repository-manager always ships pyyaml
    print("ERROR: PyYAML is required (pip install pyyaml).", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Allowlists -- the invariant. Extending fleet toolchain coverage means
# adding one line here (and a provisioning signature / Dockerfile install
# below), not hunting for every place a package list was hand-maintained.
# ---------------------------------------------------------------------------

# Binaries a `language: system` hook / mergequeue gate is known to invoke
# that this image must independently provide. NOT exhaustive by design --
# an entry that invokes something outside this set and outside
# BASE_PROVIDED is UNCLASSIFIED (hard fail), which is what forces this list
# to be extended rather than silently drifting out of date.
TOOLCHAIN_BINARIES: dict[str, str] = {
    "node": "JavaScript/TypeScript runtime",
    "npm": "Node package manager (ships with node)",
    "npx": "Node package runner (ships with node)",
    "pnpm": "JS package manager (agent-webui/agent-terminal-ui/geniusbot gates)",
    "yarn": "JS package manager (alternate)",
    "cargo": "Rust build tool (epistemic-graph gates)",
    "rustc": "Rust compiler",
    "rustup": "Rust toolchain manager",
    "go": "Go toolchain",
    "docker": "Container build/compose CLI",
    "docker-compose": "Legacy standalone compose CLI",
    "mvn": "Java/Maven build tool",
    "gradle": "Java/Kotlin build tool",
    "dotnet": ".NET SDK/CLI",
    "ruby": "Ruby interpreter",
    "gem": "RubyGems package manager",
}

# Binaries/interpreters guaranteed by this Dockerfile's own base layers
# (the python:*-slim base image + its own unconditional `apt-get install
# git`/uv COPY) or standard Debian coreutils/POSIX utilities that ship in
# any Debian-slim image. Recognised so ordinary hooks (e.g. a `bash -c
# 'for f in ...; do ...; done'` loop) do not trip UNCLASSIFIED just for
# using `find`/`sed`/`test`/etc.
BASE_PROVIDED: frozenset[str] = frozenset(
    {
        "python3",
        "python",
        "pip",
        "pip3",
        "uv",
        "uvx",
        "git",
        "bash",
        "sh",
        "bump2version",
        "pre-commit",
        "sed",
        "grep",
        "egrep",
        "fgrep",
        "awk",
        "find",
        "xargs",
        "curl",
        "wget",
        "mkdir",
        "rm",
        "rmdir",
        "cp",
        "mv",
        "ls",
        "echo",
        "printf",
        "test",
        "[",
        "dirname",
        "basename",
        "cat",
        "head",
        "tail",
        "tar",
        "cut",
        "tr",
        "wc",
        "date",
        "sort",
        "uniq",
        "chmod",
        "chown",
        "cd",
        "source",
        "export",
        "set",
        "true",
        "false",
        "exit",
        "sha256sum",
        "md5sum",
        "which",
        "env",
        "sleep",
        "touch",
        "diff",
        "tee",
        "gzip",
        "gunzip",
        "ln",
        "pwd",
        "readlink",
        "realpath",
    }
)

# Shell control-flow keywords: never a command themselves, and (other than
# do/then/else/() they do not put the NEXT token in command position either
# -- e.g. the loop variable/list after `for`/`in` is not a command.
_STRUCTURAL_NO_TRIGGER = {"for", "in", "while", "until", "if", "elif", "case", "esac", "}", "!"}
_STRUCTURAL_TRIGGER = {"do", "then", "else"}
_SEPARATOR_TRIGGER = {";", "&&", "||", "|", "("}
_NON_TRIGGER_PUNCTUATION = {")", "}", ">", ">>", "<", "&"}
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Static local-mode signatures: a regex that, if found in the Dockerfile
# text, is evidence that binary is installed. Kept 1:1 with
# TOOLCHAIN_BINARIES; a name with no entry here can only be checked via
# --image.
DOCKERFILE_PROVISION_SIGNATURES: dict[str, re.Pattern[str]] = {
    "node": re.compile(r"nodejs\.org/dist|NODE_HOME|node-v[\d.]+-linux"),
    "npm": re.compile(r"nodejs\.org/dist|NODE_HOME|node-v[\d.]+-linux"),
    "npx": re.compile(r"nodejs\.org/dist|NODE_HOME|node-v[\d.]+-linux"),
    "pnpm": re.compile(r"install\s+-g\s+pnpm@|corepack prepare pnpm@"),
    "yarn": re.compile(r"install\s+-g\s+yarn@|corepack prepare yarn@"),
    "cargo": re.compile(r"rustup\.rs|RUSTUP_HOME|CARGO_HOME"),
    "rustc": re.compile(r"rustup\.rs|RUSTUP_HOME|CARGO_HOME"),
    "rustup": re.compile(r"rustup\.rs|RUSTUP_HOME|CARGO_HOME"),
    "go": re.compile(r"go\.dev/dl|GOROOT\b"),
    "docker": re.compile(r"docker-ce|docker\.io\b|download\.docker\.com"),
    "docker-compose": re.compile(r"docker-compose-plugin|docker/compose/releases"),
    "mvn": re.compile(r"maven\.apache\.org|apache-maven-"),
    "gradle": re.compile(r"gradle\.org/next-steps|services\.gradle\.org"),
    "dotnet": re.compile(r"dot\.net/install|dotnet-install"),
    "ruby": re.compile(r"apt-get install[^\n]*\bruby\b|rvm\.io"),
    "gem": re.compile(r"apt-get install[^\n]*\bruby\b|rvm\.io"),
}


@dataclass
class RequiredUse:
    """One (file, hook/gate) site that requires a binary."""

    binary: str
    source_file: str
    site_id: str
    snippet: str


@dataclass
class Unclassified:
    """A command-position token that matched no known allowlist."""

    token: str
    source_file: str
    site_id: str
    snippet: str


@dataclass
class ScanResult:
    required: dict[str, list[RequiredUse]] = field(default_factory=dict)
    unclassified: list[Unclassified] = field(default_factory=list)

    def add_required(self, binary: str, use: RequiredUse) -> None:
        self.required.setdefault(binary, []).append(use)


def _command_tokens(script: str) -> list[str]:
    """Return the command-position tokens of a shell fragment (heuristic).

    Deliberately not a full shell parser (see module docstring): walks a
    shlex token stream with `punctuation_chars` enabled (so `;`, `&&`,
    `||`, `|`, `(`, `)`, `>`, `<` become their own tokens) and tracks
    whether the next token is in "command position" -- the start of the
    script, or immediately after a separator/`do`/`then`/`else`/`(`.
    """
    try:
        lexer = shlex.shlex(script, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        # Unbalanced quotes or similar -- cannot safely tokenize; surface
        # the whole fragment as unclassified rather than guess.
        return [f"<<TOKENIZE-FAILED: {script!r}>>"]

    commands: list[str] = []
    expect_command = True
    for tok in tokens:
        if tok in _SEPARATOR_TRIGGER or tok in _STRUCTURAL_TRIGGER:
            expect_command = True
            continue
        if tok in _NON_TRIGGER_PUNCTUATION:
            expect_command = False
            continue
        if tok in _STRUCTURAL_NO_TRIGGER:
            expect_command = False
            continue
        if not expect_command:
            continue
        if _ASSIGNMENT_RE.match(tok):
            # `FOO=bar cmd` -- still waiting for the real command.
            continue
        commands.append(tok)
        expect_command = False
    return commands


def _classify(token: str) -> tuple[str, str | None]:
    """Classify one command-position token.

    Returns (kind, binary) where kind is "toolchain", "base", or
    "unclassified". Path-like tokens (`.venv/bin/python`,
    `./scripts/foo.sh`) are matched on their basename, since the venv/
    script wrapper itself is not the dependency -- its own interpreter is,
    and that's either already covered by BASE_PROVIDED (a `.venv` python)
    or invisible to static text scanning (a script's own shebang), which
    is the documented scope boundary in the module docstring.
    """
    name = token.rsplit("/", 1)[-1]
    if name in TOOLCHAIN_BINARIES:
        return "toolchain", name
    if name in BASE_PROVIDED:
        return "base", name
    # A bare script/path invocation (`./scripts/x.sh`, `scripts/x.py`) --
    # its own shebang interpreter is out of this scanner's reach; treat as
    # base-provided-by-assumption ONLY when it clearly names its own
    # interpreter file extension, so this does not silently swallow a
    # genuine unknown command.
    if re.search(r"\.(py|sh|mjs|cjs)$", name):
        return "base", name
    return "unclassified", None


def _iter_precommit_system_entries(path: Path) -> list[tuple[str, str]]:
    """Return [(hook_id, entry_text), ...] for language: system hooks."""
    try:
        doc = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError) as exc:
        print(f"WARNING: could not parse {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(doc, dict):
        return []
    out: list[tuple[str, str]] = []
    for repo in doc.get("repos") or []:
        if not isinstance(repo, dict):
            continue
        for hook in repo.get("hooks") or []:
            if not isinstance(hook, dict):
                continue
            if hook.get("language") != "system":
                continue
            entry = hook.get("entry")
            hook_id = hook.get("id", "<unnamed>")
            if isinstance(entry, str):
                out.append((hook_id, entry))
    return out


def _iter_mergequeue_entries(path: Path) -> list[tuple[str, str]]:
    """Return [(gate_name, command_text), ...] for every mergequeue gate.

    There is no `language` field in .mergequeue.yaml -- every gate is a
    raw subprocess, so every gate's `command:` is in scope.
    """
    try:
        doc = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError) as exc:
        print(f"WARNING: could not parse {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(doc, dict):
        return []
    out: list[tuple[str, str]] = []
    for gate in doc.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        command = gate.get("command")
        name = gate.get("name", "<unnamed>")
        if not isinstance(command, list) or not command:
            continue
        # ["bash", "-c", "<script>", ...] -- the script is what to scan;
        # anything else is already a literal argv list (command[0] is the
        # binary, unambiguous, no shell text to parse).
        if command[0] in ("bash", "sh") and len(command) >= 3 and command[1] == "-c":
            out.append((name, command[2]))
        else:
            out.append((name, shlex.join(str(c) for c in command)))
    return out


# Directories never worth descending into: VCS internals, dependency
# closures, and build/cache output. Pruning these is what keeps the scan in
# the pre-push HEAVY tier's seconds rather than crawling every node_modules/
# .venv/target across 75 repos (measured: unpruned ~9s, several times any
# single repo's own pre-commit budget).
_PRUNE_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        "target-isolated",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
    }
)


def _find_fleet_configs(fleet_root: Path, filename: str) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(fleet_root):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        if filename in filenames:
            found.append(Path(dirpath) / filename)
    return sorted(found)


def scan_fleet(fleet_root: Path) -> ScanResult:
    result = ScanResult()
    precommit_files = _find_fleet_configs(fleet_root, ".pre-commit-config.yaml")
    mergequeue_files = _find_fleet_configs(fleet_root, ".mergequeue.yaml")

    def _handle(rel: str, site_id: str, text: str) -> None:
        for tok in _command_tokens(text):
            kind, binary = _classify(tok)
            snippet = text.strip().replace("\n", " ")[:160]
            if kind == "toolchain":
                assert binary is not None
                result.add_required(
                    binary, RequiredUse(binary, rel, site_id, snippet)
                )
            elif kind == "unclassified":
                result.unclassified.append(Unclassified(tok, rel, site_id, snippet))

    for f in precommit_files:
        rel = str(f.relative_to(fleet_root))
        for hook_id, entry in _iter_precommit_system_entries(f):
            _handle(rel, hook_id, entry)

    for f in mergequeue_files:
        rel = str(f.relative_to(fleet_root))
        for gate_name, command_text in _iter_mergequeue_entries(f):
            _handle(rel, gate_name, command_text)

    return result


def _probe_image(image: str, binaries: list[str]) -> dict[str, bool | None]:
    if not binaries:
        return {}
    check = " ".join(f"command -v {b} >/dev/null 2>&1 && echo {b}=OK || echo {b}=MISSING;" for b in binaries)
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", image, "sh", "-c", check],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: could not run image probe against {image}: {exc}", file=sys.stderr)
        sys.exit(2)
    if proc.returncode != 0 and not proc.stdout:
        print(f"ERROR: `docker run` against {image} failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(2)
    verdicts: dict[str, bool | None] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        name, _, verdict = line.partition("=")
        verdicts[name] = verdict.strip() == "OK"
    return verdicts


def _probe_dockerfile(dockerfile: Path, binaries: list[str]) -> dict[str, bool | None]:
    text = dockerfile.read_text()
    verdicts: dict[str, bool | None] = {}
    for b in binaries:
        sig = DOCKERFILE_PROVISION_SIGNATURES.get(b)
        verdicts[b] = bool(sig.search(text)) if sig else None
    return verdicts


def _find_fleet_root(start: Path) -> Path | None:
    """Walk upward for the `agent-packages` root (has both `agents/` and
    `agent-utilities/`). NOT a fixed `parents[N]` index: this script's own
    directory depth from the fleet root is only `scripts/repository-manager/
    agents/agent-packages` in the CANONICAL checkout. A git worktree (this
    repo's own recommended isolation pattern, `git worktree add
    /home/apps/worktrees/repository-manager/<branch>`) lives at a different
    depth entirely -- a fixed-index default silently resolved to the wrong
    directory here in testing (scanned an unrelated tree one level short of
    real `agent-packages`) rather than failing loudly, which is exactly the
    under-detection failure mode this whole script exists to avoid. Mirrors
    the same upward-search idiom `agent-webui`'s own system hooks already
    use to locate `agent-utilities/scripts` from an arbitrary checkout depth.
    """
    d = start
    while d != d.parent:
        if (d / "agent-utilities").is_dir() and (d / "agents").is_dir():
            return d
        d = d.parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve()
    default_dockerfile = here.parents[1] / "docker" / "Dockerfile"  # repository-manager/docker/Dockerfile
    default_fleet_root = _find_fleet_root(here)
    parser.add_argument(
        "--fleet-root",
        type=Path,
        default=default_fleet_root,
        required=default_fleet_root is None,
        help="agent-packages root. Auto-detected by walking up from this script "
        "for a directory containing both agents/ and agent-utilities/; pass "
        "explicitly if that search fails (e.g. a partial/sparse checkout).",
    )
    parser.add_argument("--image", default=None, help="Probe a built image via `docker run` (CI mode).")
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=None,
        help=f"Statically scan a Dockerfile's declared installs (local mode). Default: {default_dockerfile}",
    )
    args = parser.parse_args()

    if args.image and args.dockerfile:
        parser.error("--image and --dockerfile are mutually exclusive")
    if not args.image and not args.dockerfile:
        args.dockerfile = default_dockerfile

    fleet_root = args.fleet_root.resolve()
    if not fleet_root.is_dir():
        parser.error(f"--fleet-root {fleet_root} is not a directory")

    result = scan_fleet(fleet_root)
    required_binaries = sorted(result.required)

    verdicts: dict[str, bool | None]
    if args.image:
        mode = f"image probe ({args.image})"
        verdicts = _probe_image(args.image, required_binaries)
    else:
        dockerfile = args.dockerfile.resolve()
        if not dockerfile.is_file():
            parser.error(f"--dockerfile {dockerfile} does not exist")
        mode = f"static Dockerfile scan ({dockerfile})"
        verdicts = _probe_dockerfile(dockerfile, required_binaries)

    print(f"# rm_gates toolchain coverage — fleet root: {fleet_root}")
    print(f"# mode: {mode}")
    print(f"# {len(required_binaries)} distinct toolchain(s) declared by language: system "
          f"hooks / mergequeue gates fleet-wide\n")

    ok = True
    for binary in required_binaries:
        uses = result.required[binary]
        verdict = verdicts.get(binary)
        if verdict is True:
            status = "OK"
        elif verdict is False:
            status = "MISSING"
            ok = False
        else:
            status = "UNKNOWN (no static signature for this binary — rerun with --image)"
            ok = False
        sites = ", ".join(f"{u.source_file}::{u.site_id}" for u in uses[:3])
        more = f" (+{len(uses) - 3} more)" if len(uses) > 3 else ""
        print(f"[{status:7}] {binary:16} — {TOOLCHAIN_BINARIES[binary]}")
        print(f"          required by: {sites}{more}")

    if result.unclassified:
        ok = False
        print(f"\n{len(result.unclassified)} UNCLASSIFIED command(s) — a language: system hook or "
              f"mergequeue gate invokes something outside both TOOLCHAIN_BINARIES and BASE_PROVIDED. "
              f"This fails LOUD rather than silently assuming the command is harmless (that silent "
              f"assumption is the exact failure mode that let the node/pnpm gap ship). Add the "
              f"binary to one of the two allowlists in this script, with a justification, to clear it.")
        seen: set[tuple[str, str]] = set()
        for u in result.unclassified:
            key = (u.token, u.source_file)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {u.token!r} in {u.source_file}::{u.site_id}: {u.snippet}")

    if ok:
        print("\nPASS — every declared toolchain is covered.")
        return 0
    print("\nFAIL — see MISSING/UNKNOWN/UNCLASSIFIED above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
