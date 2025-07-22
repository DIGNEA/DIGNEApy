FROM python:3.13-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13 


COPY . /digneapy
WORKDIR /digneapy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

WORKDIR /digneapy/examples
