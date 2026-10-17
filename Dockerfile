ARG PYTHON_IMAGE=python:3.14-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1

FROM ${UV_IMAGE} AS uv


FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /src

COPY pyproject.toml uv.lock CMakeLists.txt README.md LICENSE ./
COPY cmake ./cmake
COPY core ./core
COPY bindings ./bindings
COPY python ./python

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /wheels


FROM builder AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --only-group test --no-hashes --no-emit-project --output-file /tmp/test-requirements.txt \
    && uv pip install --system -r /tmp/test-requirements.txt /wheels/*.whl

COPY tests ./tests

CMD ["pytest", "-q", "tests/python"]


FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 app

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*.whl

USER app
WORKDIR /home/app

ENTRYPOINT ["hopfilter"]
CMD ["--help"]
