FROM python:3.13-bookworm

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

# Environment configuration
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13 

# Set working directory
WORKDIR /build

# Copy only necessary files to install the package
COPY . .

# Install the package into the image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install . --system --compile-bytecode && \ 
    cd / && rm -rf /build 


# Switch to a clean workspace for users
WORKDIR /workspace
COPY examples/ /workspace

# Set uv run as default entrypoint for running user scripts
ENTRYPOINT ["uv", "run"]
