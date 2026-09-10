import json

import redis


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

HISTORY_PREFIX = "chat_history:"


def get_history(session_id: str) -> list[dict]:
    """Get conversation history for a session."""
    key = f"{HISTORY_PREFIX}{session_id}"

    messages = redis_client.lrange(key, 0, -1)

    return [json.loads(message) for message in messages]


def save_history(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Save one message to conversation history."""
    key = f"{HISTORY_PREFIX}{session_id}"

    message = {
        "role": role,
        "content": content,
    }

    redis_client.rpush(
        key,
        json.dumps(message),
    )
