import hashlib
from functools import lru_cache


def compute_hash(question: str, schema: str) -> str:
    """Generate a stable hash string for caching."""
    key = (question + schema).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


# LRU cache for Text2Cypher results (mapping key → query)
# maxsize controls memory usage (adjust to needs)
@lru_cache(maxsize=256)
def cached_t2c_result(key: str, result: str) -> str:
    """
    LRU cache wrapper.
    IMPORTANT: LRU cache functions must be pure (no side-effects).
    We pass key and result so LRU stores them.
    """
    return result
