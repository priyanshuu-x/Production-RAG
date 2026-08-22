import redis
from fastapi import Request, HTTPException

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60


def check_rate_limit(request: Request):
    """
    FastAPI dependency: raises HTTP 429 if the client's IP has exceeded
    RATE_LIMIT_MAX_REQUESTS within RATE_LIMIT_WINDOW_SECONDS.
    """
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"

    current_count = redis_client.incr(key)

    if current_count == 1:
        # first request in this window -- set the expiry
        redis_client.expire(key, RATE_LIMIT_WINDOW_SECONDS)

    if current_count > RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds.",
        )