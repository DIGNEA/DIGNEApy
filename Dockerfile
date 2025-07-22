FROM python:3.13-rc-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13 

WORKDIR /digneapy
COPY . .

CMD ["uv", "run", "digneapy/examples/solvers/knapsack_evolutionary_solver.py"]