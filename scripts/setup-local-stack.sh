#!/usr/bin/env bash
# Prepare a checkout for the Docker Compose stack. Safe to re-run.
#
#   scripts/setup-local-stack.sh
#   docker compose up --build
#
# - creates ./outputs and ./data/raw as YOUR directories. Compose binds them with
#   create_host_path: false, so Docker never creates them as root;
# - creates .env from .env.example (local development values) with SCALERL_UID /
#   SCALERL_GID set to your user, so the trainer can write ./outputs on Linux;
# - never overwrites an existing .env; it only warns when its UID/GID differ.
set -euo pipefail

cd "$(dirname "$0")/.."

uid=$(id -u)
gid=$(id -g)

if [[ "$uid" == "0" ]]; then
  echo "Run this as your normal user, not root: the directories must belong to you." >&2
  exit 1
fi

for dir in outputs data/raw; do
  mkdir -p "$dir"
  if [[ ! -w "$dir" ]]; then
    echo "$dir is not writable by $(id -un) (likely created by Docker as root)." >&2
    echo "Fix it with: sudo chown -R $uid:$gid $dir" >&2
    exit 1
  fi
done
echo "ok   outputs/ and data/raw/ exist and are yours"

if [[ ! -f .env ]]; then
  sed -e "s/^SCALERL_UID=.*/SCALERL_UID=$uid/" \
      -e "s/^SCALERL_GID=.*/SCALERL_GID=$gid/" \
      .env.example > .env
  echo "ok   created .env from .env.example (local development values; UID/GID $uid:$gid)"
else
  current_uid=$(sed -n 's/^SCALERL_UID=//p' .env)
  current_gid=$(sed -n 's/^SCALERL_GID=//p' .env)
  if [[ "$current_uid" != "$uid" || "$current_gid" != "$gid" ]]; then
    echo "warn .env has SCALERL_UID/GID=${current_uid:-unset}:${current_gid:-unset}, you are $uid:$gid."
    echo "     On Linux, set them to $uid and $gid so the trainer can write ./outputs."
  else
    echo "ok   .env already exists (UID/GID $uid:$gid)"
  fi
fi
