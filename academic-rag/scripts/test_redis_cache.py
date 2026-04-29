import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.agent import ConversationMemory
from src.cache.redis_cache import RedisCache, md5_key


@dataclass
class RedisTestConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    socket_timeout: float = 0.2
    socket_connect_timeout: float = 0.2


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_embedding_cache() -> None:
    cache = RedisCache[str, np.ndarray](
        max_size=16,
        ttl_seconds=60,
        redis_config=RedisTestConfig(),
        value_codec="ndarray",
    )
    query_text = "Represent this sentence for searching relevant passages: local sgd"
    key = f"vec:{md5_key(query_text)}"
    value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    cache.set(key, value)
    cached = cache.get(key)

    assert_true(cached is not None, "embedding cache miss")
    assert_true(np.allclose(cached, value), "embedding cache value mismatch")
    print("OK embedding cache hit")


def test_retrieval_cache() -> None:
    cache = RedisCache[str, list[tuple[int, float, int]]](
        max_size=16,
        ttl_seconds=60,
        redis_config=RedisTestConfig(),
        value_codec="json",
    )
    query = "local sgd convergence"
    key = f"ret:1:{md5_key(query)}:5:0.5"
    value = [(3, 0.91, 1), (7, 0.82, 2)]

    cache.set(key, value)
    cached = cache.get(key)
    normalized = [tuple(item) for item in cached or []]

    assert_true(normalized == value, "retrieval cache value mismatch")
    print("OK retrieval cache hit")


def test_session_memory() -> None:
    memory = ConversationMemory(max_turns=2, redis_config=RedisTestConfig())
    memory.clear("test-session")

    memory.add_turn("test-session", "q1", "a1")
    memory.add_turn("test-session", "q2", "a2")
    memory.add_turn("test-session", "q3", "a3")
    messages = memory.get_messages("test-session")

    assert_true(len(messages) == 4, "session memory max_turns trimming failed")
    assert_true(messages[0]["content"] == "q2", "session memory did not keep recent turns")
    assert_true(messages[-1]["content"] == "a3", "session memory latest answer missing")

    memory.clear("test-session")
    assert_true(memory.get_messages("test-session") == [], "session clear failed")
    print("OK session read/write")


def test_redis_disconnect_fallback() -> None:
    cache = RedisCache[str, list](
        max_size=4,
        ttl_seconds=60,
        redis_config=RedisTestConfig(port=1),
        value_codec="json",
    )

    cache.set("fallback:key", [1, 2, 3])
    assert_true(cache.get("fallback:key") == [1, 2, 3], "fallback TTLCache failed")
    print("OK redis disconnect fallback")


def main() -> None:
    previous = os.environ.get("USE_REDIS")
    os.environ["USE_REDIS"] = "true"
    try:
        test_embedding_cache()
        test_retrieval_cache()
        test_session_memory()
        test_redis_disconnect_fallback()
    finally:
        if previous is None:
            os.environ.pop("USE_REDIS", None)
        else:
            os.environ["USE_REDIS"] = previous


if __name__ == "__main__":
    main()
