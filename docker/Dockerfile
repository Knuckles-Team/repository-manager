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
ARG HOST=127.0.0.1
ARG PORT=8000
ARG TRANSPORT="stdio"
ARG AUTH_TYPE="none"
ENV HOST=${HOST} \
    PORT=${PORT} \
    TRANSPORT=${TRANSPORT} \
    AUTH_TYPE=${AUTH_TYPE} \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/cargo/bin:/usr/local/bin:${PATH}" \
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
