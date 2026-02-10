#FROM python:3.15.0a1-trixie
FROM pytorch/manylinux-cpu
# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

# Environment configuration
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=manual \
    UV_PYTHON=python3.13 

# Set working directory
WORKDIR /build

# Copy only necessary files to install the package
COPY . . 

# Install the package into the image
RUN --mount=type=cache,target=/root/.cache/uv \ 
    uv python install 3.13 && uv python pin 3.13  \
    && uv run pip install . --system --compile-bytecode

# Switch to a clean workspace for users
WORKDIR /workspace
COPY examples/ /workspace

RUN apt-get clean && \
    rm -rf /root/.cache /var/lib/apt/lists/* /tmp/* /build

# Set uv run as default entrypoint for running user scripts
ENTRYPOINT ["uv", "run"]
