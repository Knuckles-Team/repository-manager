#!/usr/bin/env bash
# One-command contributor setup from a fresh clone (local, Claude Code on the
# web, and the hosted CI workflow all run this same script).
#
#   scripts/bootstrap.sh            uv, Python, pinned siblings, .venv, git hooks
#   scripts/bootstrap.sh --native   also build the epistemic-graph kernel (slow;
#                                   only `pytest -m native` needs it)
#
# Idempotent and non-interactive. Afterwards:
#   uvx pre-commit run --all-files
#   uv run --frozen --no-sync pytest tests -m 'not slow and not integration and not native'
set -euo pipefail

cd "$(dirname "$0")/.."

native=0
for arg in "$@"; do
  case "$arg" in
    --native) native=1 ;;
    -h | --help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

log() { printf 'bootstrap: %s\n' "$*"; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1. uv + Python ------------------------------------------------------------
# uv older than 0.9 cannot download current CPython patch releases.
uv_minor() { uv --version 2>/dev/null | awk '{split($2, v, "."); print v[1] * 1000 + v[2]}'; }
if ! command -v uv >/dev/null 2>&1 || [[ "$(uv_minor)" -lt 9 ]]; then
  log "installing a current uv"
  python3 -m pip install --quiet --user --upgrade "uv>=0.9" \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
python_version="$(tr -d '[:space:]' < .python-version)"
log "Python ${python_version} (uv $(uv --version | awk '{print $2}'))"
uv python install "$python_version"

# 2. Sibling sources --------------------------------------------------------
# uv.lock installs agent-utilities (and, through it, the SDK and the
# epistemic-graph engine) editable from ignored .uv-workspace-siblings paths.
# Each is cloned at exactly the commit in scripts/siblings.lock, the only place
# those pins live. A developer symlink at a sibling path is respected as-is.
sync_sibling() {
  local name="$1" url="$2" sha="$3" path="$4"
  if [ -L "$path" ]; then
    log "sibling ${name}: using developer symlink ${path} -> $(readlink "$path")"
    return
  fi
  if [ -d "$path/.git" ] && [ "$(git -C "$path" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
    log "sibling ${name}: at ${sha:0:12}"
    return
  fi
  if [ -e "$path" ] && [ ! -d "$path/.git" ]; then
    echo "bootstrap: ${path} exists but is not a git checkout; remove it and re-run" >&2
    exit 1
  fi
  log "sibling ${name}: fetching ${sha:0:12}"
  mkdir -p "$path"
  [ -d "$path/.git" ] || git -C "$path" init -q
  git -C "$path" remote remove origin 2>/dev/null || true
  git -C "$path" remote add origin "$url"
  local attempt
  for attempt in 1 2 3 4; do
    git -C "$path" fetch -q --depth 1 origin "$sha" && break
    [ "$attempt" = 4 ] && { echo "bootstrap: cannot fetch ${name}@${sha}" >&2; exit 1; }
    sleep $((2 ** attempt))
  done
  git -C "$path" -c advice.detachedHead=false checkout -q --force FETCH_HEAD
}

while read -r name url sha path; do
  case "$name" in '' | '#'*) continue ;; esac
  sync_sibling "$name" "$url" "$sha" "$path"
done < scripts/siblings.lock

# 3. Locked environment -----------------------------------------------------
# epistemic-graph is a native (Rust) engine; building it would dominate setup
# time and only the `native`-marked tests need it.
sync_args=(--frozen --python "$python_version" --extra test --extra agent)
if [[ "$native" == 0 ]]; then
  sync_args+=(--no-install-package epistemic-graph)
fi
log "syncing .venv (uv sync ${sync_args[*]})"
uv sync "${sync_args[@]}"

# 4. Git hooks --------------------------------------------------------------
log "installing pre-commit and pre-push hooks"
uvx pre-commit install --hook-type pre-commit --hook-type pre-push >/dev/null

log "done"
