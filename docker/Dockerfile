# Two right-sized images from ONE Dockerfile, selected by build --target. Both are
# slim multi-stage (build in a builder, ship only /usr/local — no dev-tools/source):
#
#   --target agent  (DEFAULT)  installs repository-manager[agent] = the FULL runtime
#                              (agent-utilities[agent-runtime]: pydantic-ai + skills;
#                              the full epistemic-graph engine is a base dependency).
#                              The provider's [agent] also includes [mcp], so this image
#                              can run EITHER the agent OR the mcp server. Push as the
#                              default tags: latest / main / <version>.
#
#   --target mcp               installs repository-manager[mcp]   = MCP-server tooling ONLY
#                              (agent-utilities[mcp]: FastMCP/FastAPI plus the mandatory
#                              full epistemic-graph base runtime). It omits the optional
#                              provider agent/frontends. Push as the :mcp / :mcp-<version> tag.
#
#   docker build --target agent -t repository-manager:local .
#   docker build --target mcp   -t repository-manager:mcp    .
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=3600
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Slim MCP-only dependency closure.
FROM builder-base AS builder-mcp
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages repository-manager[mcp]==3.3.0

# Full agent runtime dependency closure.
FROM builder-base AS builder-agent
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages repository-manager[agent]==3.3.0

FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6 AS runtime-base
# repository-manager shells out to `git` at RUNTIME — the validate/release flow runs
# pre-commit (which requires git) and the git_tools/rm_git actions call git directly.
# A git tool must ship git in its runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
# uv is needed at RUNTIME too: the validate/release flow's pytest pre-commit hooks
# shell out to `uv run`. A dev/validator worker must ship the toolchain it drives.
COPY --from=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a /uv /uvx /bin/

# D-MQ-8 — `rm__merge_queue`'s gate subprocesses run INSIDE this container, and a
# target repo's own `.mergequeue.yaml` can declare ANY toolchain (e.g.
# epistemic-graph's cargo-check/clippy gates). Without one here, every such gate
# auto-rejects in seconds with `[Errno 13] Permission denied: 'cargo'` — no
# `cargo` on PATH at all — which reads exactly like a real gate failure and
# silently strands every candidate for that repo. This is the SAME defect class
# already fixed once for the host-native `merge-queue-runner.service` (its
# PATH there is explicitly widened to include `~/.cargo/bin`) — that fix was
# never mirrored into this container; it is mirrored here now the same way:
# install the toolchain and put it on PATH.
#
# Pinned to the exact channel `epistemic-graph/rust-toolchain.toml` declares
# (keep these in sync) so the queue's `environment_signature` fingerprint and
# every gate's actual compiler match what that file's own `rustup` would
# resolve.
#
# CARGO_HOME/RUSTUP_HOME are fixed, HOME-INDEPENDENT paths under /usr/local —
# NOT rustup's default `$HOME/.cargo`/`$HOME/.rustup`. This container's runtime
# HOME is overridden per-deployment (the k8s Deployment sets `HOME=/home/rm`,
# not this Dockerfile's own `ENV HOME` below), so a HOME-relative install would
# resolve under root's BUILD-time HOME and be invisible at RUNTIME under the
# deployed HOME — the container's own version of the non-interactive-shell PATH
# trap the host-native fix had to account for. A fixed path plus adding that
# fixed path to PATH (below) sidesteps HOME entirely.
#
# `cmake` (+ build-essential's C toolchain) is required for epistemic-graph's
# `--all-features` build: the `ros2-rmw` feature's `cyclonedds-rust-sys` vendors
# CycloneDDS' C sources and builds them with `cmake` at compile time — without
# it the cargo-check/clippy gates fail on that feature alone, unrelated to the
# code under review. (A census found this gate had likely never successfully
# built anywhere until cmake was installed.)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain none \
    && "$CARGO_HOME/bin/rustup" toolchain install 1.96.0 \
        --profile minimal --component clippy --component rustfmt \
    && "$CARGO_HOME/bin/rustup" default 1.96.0 \
    # World-writable: the runtime user is a fixed non-root uid (below), and cargo
    # itself writes to its registry cache/index under CARGO_HOME on a non-`
    # --offline` gate (epistemic-graph's cargo-check/clippy are not `--offline`).
    && chmod -R a+rwX "$RUSTUP_HOME" "$CARGO_HOME" \
    && rm -rf /var/lib/apt/lists/*

# node + pnpm — the SAME defect class as D-MQ-8 above, never mirrored from
# cargo to node until now. `agent-webui/.pre-commit-config.yaml` declares
# three `language: system` hooks that shell out to `node`/`pnpm` directly
# (no-fabrication-gate, pnpm-build, typecheck), and those gate subprocesses
# run INSIDE this container the same way the cargo ones do. Without a
# node/pnpm on PATH, every candidate for agent-webui (and any other repo
# whose `.pre-commit-config.yaml`/`.mergequeue.yaml` declares a `language:
# system` node hook) auto-rejects in seconds with `[Errno 2] No such file
# or directory: 'node'` — reads exactly like a real gate failure, silently
# stranding the candidate. `rm_gates_toolchain_coverage.py` (run in CI
# against the built image, see its own header) is what stops this
# particular gap from being reintroduced by the next repo that adopts a
# new toolchain — this block is only what today's fleet needs.
#
# Installed as the official prebuilt linux-x64 tarball into a FIXED,
# HOME-INDEPENDENT path (/usr/local/node) — the same requirement as
# CARGO_HOME/RUSTUP_HOME above, for the same reason (this stage's
# build-time HOME is root's; the runtime HOME below is not). pnpm is
# installed with a plain `npm install -g` rather than `corepack prepare
# --activate`: corepack's own package cache defaults to $COREPACK_HOME,
# which resolves under HOME the same way rustup's default `$HOME/.cargo`
# does — the exact HOME-relative-install trap already named above. `npm
# install -g` instead drops the pnpm binary under node's own install
# prefix ($NODE_HOME), which carries no HOME dependency at all.
#
# Versions pinned to what agent-webui's own toolchain declares, so gate
# output here matches what that repo's own CI/local runs produce: Node 22
# (agent-webui/docker/Dockerfile's `FROM node:22-slim`) and pnpm 10.29.3
# (agent-webui/package.json's `packageManager` field — keep these in
# sync, same discipline as the rust-toolchain.toml pin above). sha256
# verified against nodejs.org's own published SHASUMS256.txt for this
# release.
ENV NODE_HOME=/usr/local/node
RUN curl --proto '=https' --tlsv1.2 -sSfL \
        https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.xz \
        -o /tmp/node.tar.xz \
    && echo "d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307  /tmp/node.tar.xz" \
        | sha256sum -c - \
    && mkdir -p "$NODE_HOME" \
    && tar -xJf /tmp/node.tar.xz -C "$NODE_HOME" --strip-components=1 \
    && rm -f /tmp/node.tar.xz \
    # npm's own shebang execs `node` via PATH lookup (not a hardcoded path),
    # so PATH must carry $NODE_HOME/bin for THIS invocation even though the
    # image-wide PATH ENV is only set below.
    && PATH="$NODE_HOME/bin:$PATH" "$NODE_HOME/bin/npm" install -g pnpm@10.29.3 \
    && PATH="$NODE_HOME/bin:$PATH" "$NODE_HOME/bin/npm" cache clean --force \
    # Read+execute only (unlike CARGO_HOME/RUSTUP_HOME): pnpm's own runtime
    # store/cache is HOME-relative by design and created under the
    # deployed runtime HOME on first use, not under $NODE_HOME — nothing
    # here needs a runtime write.
    && chmod -R a+rX "$NODE_HOME"
ARG HOST=127.0.0.1
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/cargo/bin:/usr/local/node/bin:/usr/local/bin:${PATH}" \
    UV_HTTP_TIMEOUT=3600 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Slim image: MCP server only.
# Fixed identity for least-privilege runtime and stable volume ownership.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin app
ENV HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_CACHE_HOME=/tmp/.cache

FROM runtime-base AS mcp
USER 10001:10001
COPY --from=builder-mcp /usr/local /usr/local
CMD ["repository-manager-mcp"]

# Full image (DEFAULT target): agent runtime; can also run the mcp server.
FROM runtime-base AS agent
USER 10001:10001
COPY --from=builder-agent /usr/local /usr/local
CMD ["repository-manager-agent"]
