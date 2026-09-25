# syntax=docker/dockerfile:1
#
# The native Optuna Dashboard with the PostgreSQL driver, pinned to the same
# Optuna 5.x release as ScaleRL tuning. Installed at build time, never on start.

FROM python:3.12.14-slim-trixie

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY docker/optuna-dashboard/requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN useradd --create-home --uid 10001 optuna
USER optuna
WORKDIR /home/optuna

EXPOSE 8080
