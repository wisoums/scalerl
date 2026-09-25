# syntax=docker/dockerfile:1
#
# ScaleRL runtime image: Scenario Lab UI and trainer (evaluation, tuning, and
# later DQN/PPO training). Build from the repository root:
#
#   docker build -t scalerl .
#
# Dependencies are installed from hash-pinned Linux requirements
# (docker/scalerl/requirements.txt, regenerated with
# scripts/compile-docker-requirements.sh). PyTorch is the CPU build, so the
# image runs on ordinary Docker (amd64 and arm64) without a GPU. No datasets,
# credentials, or local experiment state enter the build context
# (see .dockerignore); data/raw and outputs/ are mounted at runtime.

ARG PYTHON_IMAGE=python:3.12.14-slim-trixie

FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Third-party dependencies first, so source edits do not reinstall them.
COPY docker/scalerl/requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt

# Then the real ScaleRL package (a regular, non-editable install). The wheel is
# built with the hash-pinned hatchling from a separate build venv and without
# build isolation, so no unpinned build backend is ever downloaded, and the
# build tooling never reaches the runtime venv.
COPY docker/scalerl/build-requirements.txt /tmp/build-requirements.txt
RUN python -m venv /opt/build \
    && /opt/build/bin/pip install --require-hashes -r /tmp/build-requirements.txt
COPY pyproject.toml README.md LICENSE /build/
COPY src /build/src
RUN /opt/build/bin/pip wheel --no-deps --no-build-isolation --wheel-dir /dist /build \
    && pip install --no-deps /dist/*.whl


FROM ${PYTHON_IMAGE} AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN useradd --create-home --uid 10001 scalerl
COPY --from=builder /opt/venv /opt/venv

# Relative defaults such as data/raw/<Azure trace> and outputs/ resolve here.
WORKDIR /app
RUN mkdir -p data/raw outputs && chown -R scalerl:scalerl /app
USER scalerl

EXPOSE 8501
CMD ["python", "-m", "scalerl.dashboard", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
