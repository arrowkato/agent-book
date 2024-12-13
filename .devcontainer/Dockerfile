FROM debian:bookworm-slim AS builder


WORKDIR /opt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates gnupg curl sqlite3


SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]

# Download the latest installer
ADD https://astral.sh/uv/install.sh uv-installer.sh

# Run the installer then remove it
RUN sh uv-installer.sh


FROM mcr.microsoft.com/vscode/devcontainers/base:bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    graphviz libgraphviz-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME="/opt/.cargo/bin"
ENV PATH="$CARGO_HOME/:$PATH"

ENV PYTHONUNBUFFERED=True
ENV UV_LINK_MODE=copy

WORKDIR /opt

COPY --from=builder /root/.cargo/bin/uv $CARGO_HOME/uv
COPY .python-version ./

RUN uv python pin "$(cat .python-version)"

RUN chown -R vscode $CARGO_HOME
