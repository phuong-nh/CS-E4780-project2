import hashlib
import numpy as np
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    """Represents a cached Text2Cypher result."""
    query: str
    is_valid: bool
    question: str = ""
    schema: str = ""
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def increment_hits(self):
        """Increment the hit counter."""
        self.hit_count += 1
        self.timestamp = datetime.now()


class Text2CypherLRUCache:
    """
    LRU (Least Recently Used) cache for Text2Cypher query results.
    
    Caches validated Cypher queries to avoid redundant LLM calls for similar questions.
    Uses hash(question + schema) as the cache key.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        use_semantic_similarity: bool = True,
        similarity_threshold: float = 0.90
    ):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of entries to store
            use_semantic_similarity: Enable semantic similarity matching
            similarity_threshold: Minimum similarity score for cache hit
        """
        self.max_size = max_size
        self.use_semantic_similarity = use_semantic_similarity
        self.similarity_threshold = similarity_threshold
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        self.encoder = None
        if use_semantic_similarity:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                print("Falling back to exact matching only.")
                self.use_semantic_similarity = False
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "insertions": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
        }
    
    def _generate_key(self, question: str, schema: str) -> str:
        """
        Generate a cache key from question only.
        Schema is not included to allow cache hits even when pruned schema varies.
        
        Args:
            question: The user question
            schema: The graph schema string (not used in key, kept for API compatibility)
            
        Returns:
            SHA256 hash of the normalized question
        """
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _encode_question(self, question: str) -> Optional[np.ndarray]:
        """
        Encode a question into an embedding vector.
        
        Args:
            question: The question to encode
            
        Returns:
            Embedding vector or None if encoder not available
        """
        if not self.encoder:
            return None
        try:
            return self.encoder.encode(question, convert_to_numpy=True)
        except Exception:
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)
    
    def _find_similar_entry(
        self,
        question: str,
        schema: str,
        embedding: Optional[np.ndarray]
    ) -> Optional[Tuple[str, CacheEntry, float]]:
        """
        Find semantically similar cache entry.
        Schema match is not required since questions should map to similar queries
        regardless of minor schema variations from pruning.
        
        Args:
            question: The question to match
            schema: The schema (not used for filtering)
            embedding: Question embedding
            
        Returns:
            Tuple of (key, entry, similarity_score) if found, None otherwise
        """
        if not self.use_semantic_similarity or embedding is None:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for key, entry in self.cache.items():
            # Removed exact schema match requirement to allow cache hits
            # even when pruned schema varies slightly between runs
            
            if entry.embedding is None:
                continue
            
            similarity = self._cosine_similarity(embedding, entry.embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = (key, entry, similarity)
        
        return best_match
    
    def get(self, question: str, schema: str) -> Optional[str]:
        """
        Retrieve a cached query if it exists.
        First tries exact match, then semantic similarity if enabled.
        
        Args:
            question: The user question
            schema: The graph schema string
            
        Returns:
            Cached Cypher query if found, None otherwise
        """
        key = self._generate_key(question, schema)
        
        if key in self.cache:
            entry = self.cache.pop(key)
            entry.increment_hits()
            self.cache[key] = entry
            self.stats["hits"] += 1
            self.stats["exact_hits"] += 1
            return entry.query
        
        if self.use_semantic_similarity:
            embedding = self._encode_question(question)
            similar_match = self._find_similar_entry(question, schema, embedding)
            
            if similar_match:
                match_key, entry, similarity = similar_match
                entry = self.cache.pop(match_key)
                entry.increment_hits()
                self.cache[match_key] = entry
                self.stats["hits"] += 1
                self.stats["semantic_hits"] += 1
                return entry.query
        
        self.stats["misses"] += 1
        return None
    
    def put(
        self,
        question: str,
        schema: str,
        query: str,
        is_valid: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store a query in the cache.
        
        Args:
            question: The user question
            schema: The graph schema string
            query: The generated Cypher query
            is_valid: Whether the query passed validation
            metadata: Optional metadata (e.g., repair attempts, rules applied)
        """
        key = self._generate_key(question, schema)
        
        if key in self.cache:
            self.cache.pop(key)
        
        embedding = None
        if self.use_semantic_similarity:
            embedding = self._encode_question(question)
        
        entry = CacheEntry(
            query=query,
            is_valid=is_valid,
            question=question.strip(),
            schema=schema.strip(),
            embedding=embedding,
            metadata=metadata or {}
        )
        self.cache[key] = entry
        self.stats["insertions"] += 1
        
        if len(self.cache) > self.max_size:
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
    
    def invalidate(self, question: str, schema: str) -> bool:
        """
        Remove a specific entry from the cache.
        
        Args:
            question: The user question
            schema: The graph schema string
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        key = self._generate_key(question, schema)
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def clear(self):
        """Clear all cache entries and reset statistics."""
        self.cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "insertions": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        exact_hit_rate = self.stats["exact_hits"] / total_requests if total_requests > 0 else 0.0
        semantic_hit_rate = self.stats["semantic_hits"] / total_requests if total_requests > 0 else 0.0
        
        stats = {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "exact_hit_rate": exact_hit_rate,
            "semantic_hit_rate": semantic_hit_rate,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0.0,
            "semantic_enabled": self.use_semantic_similarity,
            "similarity_threshold": self.similarity_threshold if self.use_semantic_similarity else None,
        }
        
        return stats
    
    def get_top_queries(self, n: int = 10) -> list[Tuple[str, CacheEntry]]:
        """
        Get the top N most frequently accessed queries.
        
        Args:
            n: Number of top queries to return
            
        Returns:
            List of (key, entry) tuples sorted by hit count
        """
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].hit_count,
            reverse=True
        )
        return sorted_entries[:n]
    
    def reset_statistics(self):
        """Reset statistics while keeping cached entries."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "insertions": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
        }
    
    def __len__(self) -> int:
        """Return the current number of cached entries."""
        return len(self.cache)
    
    def __contains__(self, key_tuple: Tuple[str, str]) -> bool:
        """Check if a (question, schema) pair is in the cache."""
        question, schema = key_tuple
        key = self._generate_key(question, schema)
        return key in self.cache


def create_cache(
    max_size: int = 1000,
    use_semantic_similarity: bool = True,
    similarity_threshold: float = 0.90
) -> Text2CypherLRUCache:
    """
    Create a new Text2Cypher LRU cache.
    
    Args:
        max_size: Maximum number of entries
        use_semantic_similarity: Enable semantic similarity matching
        similarity_threshold: Minimum similarity for cache hit (0-1)
        
    Returns:
        Configured Text2CypherLRUCache instance
    """
    return Text2CypherLRUCache(
        max_size=max_size,
        use_semantic_similarity=use_semantic_similarity,
        similarity_threshold=similarity_threshold
    )
