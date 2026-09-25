"""One-shot, idempotent Garage setup for the MLflow artifact bucket.

Run by the ``garage-init`` Compose service (standard library only). Through
Garage's Admin API v2 it:

1. waits for the node;
2. assigns the single node a storage role and applies the layout (first run);
3. imports the S3 access key from the environment (if absent);
4. creates the artifact bucket (if absent);
5. grants the key read/write/owner on the bucket.

Every step checks the current state first, so re-running it after
``docker compose up`` or a restart is safe. No credentials are printed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

ADMIN_URL = os.environ.get("GARAGE_ADMIN_URL", "http://garage:3903").rstrip("/")
ADMIN_TOKEN = os.environ["GARAGE_ADMIN_TOKEN"]
ACCESS_KEY_ID = os.environ["GARAGE_ACCESS_KEY_ID"]
SECRET_ACCESS_KEY = os.environ["GARAGE_SECRET_ACCESS_KEY"]
BUCKET = os.environ.get("MLFLOW_BUCKET", "mlflow")
KEY_NAME = "scalerl-mlflow"
ZONE = "local"
CAPACITY_BYTES = 10 * 1024**3  # layout weight for the only node, not a quota
WAIT_SECONDS = 60


class NotFound(Exception):
    pass


def call(endpoint: str, body: dict[str, Any] | None = None, **query: str) -> Any:
    url = f"{ADMIN_URL}/v2/{endpoint}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method="GET" if body is None else "POST",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise NotFound(endpoint) from error
        detail = error.read().decode(errors="replace")[:300]
        raise SystemExit(
            f"garage-init: {endpoint} failed with HTTP {error.code}: {detail}"
        ) from error
    return json.loads(payload) if payload else None


def wait_for_node() -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        try:
            status = call("GetClusterStatus")
            if status["nodes"]:
                return status
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"garage-init: Garage admin API not reachable at {ADMIN_URL}")
        time.sleep(1)


def ensure_layout(status: dict[str, Any]) -> None:
    layout = call("GetClusterLayout")
    if layout["roles"]:
        print(f"garage-init: layout v{layout['version']} already applied")
        return
    node_id = status["nodes"][0]["id"]
    call(
        "UpdateClusterLayout",
        {"roles": [{"id": node_id, "zone": ZONE, "capacity": CAPACITY_BYTES, "tags": []}]},
    )
    version = layout["version"] + 1
    call("ApplyClusterLayout", {"version": version})
    print(f"garage-init: applied layout v{version} for the single node")


def ensure_key() -> None:
    try:
        call("GetKeyInfo", id=ACCESS_KEY_ID)
        print("garage-init: access key already present")
    except NotFound:
        call(
            "ImportKey",
            {"accessKeyId": ACCESS_KEY_ID, "secretAccessKey": SECRET_ACCESS_KEY, "name": KEY_NAME},
        )
        print("garage-init: imported access key")


def ensure_bucket() -> str:
    try:
        bucket = call("GetBucketInfo", globalAlias=BUCKET)
        print(f"garage-init: bucket {BUCKET!r} already exists")
    except NotFound:
        bucket = call("CreateBucket", {"globalAlias": BUCKET})
        print(f"garage-init: created bucket {BUCKET!r}")
    return str(bucket["id"])


def main() -> int:
    ensure_layout(wait_for_node())
    ensure_key()
    bucket_id = ensure_bucket()
    call(
        "AllowBucketKey",
        {
            "bucketId": bucket_id,
            "accessKeyId": ACCESS_KEY_ID,
            "permissions": {"read": True, "write": True, "owner": True},
        },
    )
    print(f"garage-init: key has read/write/owner on {BUCKET!r}; done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
