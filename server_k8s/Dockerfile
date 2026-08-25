FROM flwr/superexec:latest

USER root

# Copy UV binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /uvx /bin/

# Install minimal build tools
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure virtualenv binaries are prioritized and configure UV behavior
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet

# Copy project manifests to leverage layer caching for dependencies
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml ./shared/
COPY server_k8s/pyproject.toml ./server_k8s/
COPY client_edge/pyproject.toml ./client_edge/

# Copy the entire data directory into /app/server_k8s/data/
COPY server_k8s/data/ ./server_k8s/data/

# Copy dependency files with correct ownership for user 'app'
RUN chown -R app:app /app

# Switch to default 'app' user and install dependencies
USER app

# Install dependencies from uv.lock without installing the local project package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --project server_k8s --no-install-workspace

ENTRYPOINT [ "flower-superexec" ]