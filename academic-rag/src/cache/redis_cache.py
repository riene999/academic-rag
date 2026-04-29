import hashlib
import json
import os
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from loguru import logger

from src.utils.cache import CacheStats, TTLCache

try:
    import redis
except ImportError:  # pragma: no cover - exercised when redis is not installed.
    redis = None


K = TypeVar("K")
V = TypeVar("V")


def redis_enabled() -> bool:
    return os.environ.get("USE_REDIS", "").strip().lower() == "true"


def md5_key(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


class RedisCache(Generic[K, V]):
    """Redis-backed cache with the same get/set/clear shape as TTLCache."""

    def __init__(
        self,
        max_size: int = 4096,
        ttl_seconds: int = 1800,
        redis_config: Optional[Any] = None,
        value_codec: str = "json",
    ):
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.value_codec = value_codec
        self.stats = CacheStats()
        self._fallback = TTLCache[K, V](max_size=max_size, ttl_seconds=ttl_seconds)
        self._redis = None

        if not redis_enabled():
            return

        if redis is None:
            logger.warning("USE_REDIS=true but redis package is unavailable; falling back to TTLCache")
            return

        try:
            cfg = self._normalize_config(redis_config)
            self._redis = redis.Redis(**cfg)
            self._redis.ping()
        except Exception as exc:
            self._redis = None
            logger.warning("Redis unavailable, falling back to TTLCache: {}", exc)

    @property
    def using_redis(self) -> bool:
        return self._redis is not None

    def get(self, key: K) -> Optional[V]:
        if self._redis is None:
            return self._fallback.get(key)

        try:
            raw = self._redis.get(str(key))
            if raw is None:
                self.stats.miss += 1
                return None
            self.stats.hit += 1
            return self._decode(raw)
        except Exception as exc:
            logger.warning("Redis cache get failed, falling back to TTLCache: {}", exc)
            self._redis = None
            return self._fallback.get(key)

    def set(self, key: K, value: V) -> None:
        if self._redis is None:
            self._fallback.set(key, value)
            return

        try:
            self._redis.setex(str(key), self.ttl_seconds, self._encode(value))
        except Exception as exc:
            logger.warning("Redis cache set failed, falling back to TTLCache: {}", exc)
            self._redis = None
            self._fallback.set(key, value)

    def delete(self, key: K) -> None:
        if self._redis is None:
            if hasattr(self._fallback, "_data"):
                self._fallback._data.pop(key, None)
            return

        try:
            self._redis.delete(str(key))
        except Exception as exc:
            logger.warning("Redis cache delete failed, falling back to TTLCache: {}", exc)
            self._redis = None
            if hasattr(self._fallback, "_data"):
                self._fallback._data.pop(key, None)

    def clear(self, pattern: str = "*") -> None:
        if self._redis is None:
            self._fallback.clear()
            return

        try:
            keys = list(self._redis.scan_iter(match=pattern))
            if keys:
                self._redis.delete(*keys)
        except Exception as exc:
            logger.warning("Redis cache clear failed, falling back to TTLCache: {}", exc)
            self._redis = None
            self._fallback.clear()

    def get_counter(self, key: str, default: int = 0) -> int:
        if self._redis is None:
            return int(default)

        try:
            value = self._redis.get(key)
            if value is None:
                self._redis.setnx(key, int(default))
                return int(default)
            return int(value)
        except Exception as exc:
            logger.warning("Redis counter get failed, falling back to local counter: {}", exc)
            self._redis = None
            return int(default)

    def incr_counter(self, key: str) -> int:
        if self._redis is None:
            return 0

        try:
            return int(self._redis.incr(key))
        except Exception as exc:
            logger.warning("Redis counter incr failed, falling back to local counter: {}", exc)
            self._redis = None
            return 0

    def size(self, pattern: str = "*") -> int:
        if self._redis is None:
            return len(getattr(self._fallback, "_data", {}))

        try:
            return sum(1 for _ in self._redis.scan_iter(match=pattern))
        except Exception as exc:
            logger.warning("Redis cache size failed, falling back to TTLCache: {}", exc)
            self._redis = None
            return len(getattr(self._fallback, "_data", {}))

    def _encode(self, value: V) -> bytes:
        if self.value_codec == "ndarray":
            array = np.asarray(value)
            meta = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            return json.dumps(meta).encode("utf-8") + b"\n" + array.tobytes()

        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def _decode(self, raw: bytes) -> V:
        if self.value_codec == "ndarray":
            header, payload = raw.split(b"\n", 1)
            meta = json.loads(header.decode("utf-8"))
            array = np.frombuffer(payload, dtype=np.dtype(meta["dtype"]))
            return array.reshape(meta["shape"]).copy()

        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _normalize_config(redis_config: Optional[Any]) -> dict:
        if redis_config is None:
            return {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "socket_timeout": 1.0,
                "socket_connect_timeout": 1.0,
                "decode_responses": False,
            }

        cfg = {
            "host": getattr(redis_config, "host", "localhost"),
            "port": getattr(redis_config, "port", 6379),
            "db": getattr(redis_config, "db", 0),
            "password": getattr(redis_config, "password", None),
            "socket_timeout": getattr(redis_config, "socket_timeout", 1.0),
            "socket_connect_timeout": getattr(redis_config, "socket_connect_timeout", 1.0),
            "decode_responses": False,
        }
        return {key: value for key, value in cfg.items() if value is not None}
