FROM flwr/superexec:latest

USER root

# Copy UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /uvx /bin/

# Install minimal build tools
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy workspace config & lockfile from ROOT
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml ./shared/
COPY server_k8s/pyproject.toml ./server_k8s/

# Pre-fetch & install dependencies ONLY for server (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package cloudshield-server --no-dev --no-install-workspace

# Copy source code for shared and server
COPY shared ./shared
COPY server_k8s ./server_k8s

# Complete sync (installs workspace packages)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package cloudshield-server --no-dev

# Fix ownership so non-root 'app' user can execute and write to /app
RUN chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/server_k8s/app:/app/shared/src:$PYTHONPATH"

# Switch to default 'app' user and install dependencies
USER app

ENTRYPOINT [ "flower-superexec" ]