import hashlib
import json
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

CACHE_TTL_SECONDS = 60 * 60 * 24  # 1 day


def make_cache_key(question):
    """
    Normalizes the question (lowercase, stripped) before hashing,
    so trivial differences like extra spaces or capitalization
    still hit the same cache entry.
    """
    normalized = question.strip().lower()
    return f"query_cache:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def get_cached_response(question):
    key = make_cache_key(question)
    cached = redis_client.get(key)
    return json.loads(cached) if cached else None


def set_cached_response(question, response_data):
    key = make_cache_key(question)
    redis_client.set(key, json.dumps(response_data), ex=CACHE_TTL_SECONDS)