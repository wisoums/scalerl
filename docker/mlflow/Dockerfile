# syntax=docker/dockerfile:1
#
# MLflow tracking server for the Compose stack: PostgreSQL metadata (psycopg)
# and proxied artifacts on the S3-compatible store (boto3). Infrastructure-only
# dependencies live here, not in ScaleRL's own dependency list.

FROM python:3.12.14-slim-trixie

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

COPY docker/mlflow/requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN useradd --create-home --uid 10001 mlflow
USER mlflow
WORKDIR /home/mlflow

EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]
