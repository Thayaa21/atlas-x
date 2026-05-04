import json
import os
from typing import Any, Optional

import redis


def get_redis_client() -> Optional[redis.Redis]:
    dsn = os.getenv("REDIS_DSN", "")
    if not dsn:
        # Allow plain host:port style.
        host = os.getenv("REDIS_HOST", "")
        port = int(os.getenv("REDIS_PORT", "6379"))
        if not host:
            return None
        dsn = f"redis://{host}:{port}"

    try:
        return redis.from_url(dsn, decode_responses=True)
    except Exception:
        return None


def redis_get_json(client: Optional[redis.Redis], key: str) -> Optional[Any]:
    if client is None:
        return None
    val = client.get(key)
    if val is None:
        return None
    return json.loads(val)


def redis_set_json(client: Optional[redis.Redis], key: str, value: Any, *, ttl_seconds: int = 300) -> None:
    if client is None:
        return
    payload = json.dumps(value)
    client.setex(key, ttl_seconds, payload)

