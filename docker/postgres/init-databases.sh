#!/bin/sh
# Create the separate MLflow and Optuna databases on first initialization.
#
# Runs once, from /docker-entrypoint-initdb.d, when the PostgreSQL volume is
# empty. Database names are fixed and non-secret; the owner and password come
# from POSTGRES_USER / POSTGRES_PASSWORD in the environment.
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<SQL
CREATE DATABASE mlflow OWNER "$POSTGRES_USER";
CREATE DATABASE optuna OWNER "$POSTGRES_USER";
SQL
