"""
Self-refinement loop for Text2Cypher query generation and validation.
Implements generate -> validate -> repair cycle with history tracking.
"""

import kuzu
import dspy
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from query_post_processor import CypherPostProcessor
from text2cypher_cache import Text2CypherLRUCache


class QueryAttempt(BaseModel):
    """Records a single query generation attempt."""
    query: str
    is_valid: bool
    error_message: Optional[str] = None
    attempt_number: int


class QueryHistory(BaseModel):
    """Tracks the history of query generation attempts."""
    attempts: List[QueryAttempt] = Field(default_factory=list)
    
    def add_attempt(self, query: str, is_valid: bool, error_message: Optional[str] = None):
        """Add a new attempt to the history."""
        attempt = QueryAttempt(
            query=query,
            is_valid=is_valid,
            error_message=error_message,
            attempt_number=len(self.attempts) + 1
        )
        self.attempts.append(attempt)
    
    def get_failed_attempts(self) -> List[QueryAttempt]:
        """Get all failed attempts."""
        return [a for a in self.attempts if not a.is_valid]
    
    def format_for_prompt(self) -> str:
        """Format history for inclusion in repair prompt."""
        if not self.attempts:
            return ""
        
        history_text = "Previous attempts and their errors:\n\n"
        for attempt in self.attempts:
            history_text += f"Attempt {attempt.attempt_number}:\n"
            history_text += f"Query: {attempt.query}\n"
            if not attempt.is_valid:
                history_text += f"Error: {attempt.error_message}\n"
            history_text += "\n"
        
        return history_text


class QueryValidator:
    """
    Validates Cypher queries using EXPLAIN without executing them.
    Uses Kuzu's EXPLAIN to perform dry-run validation.
    """
    
    def __init__(self, conn: kuzu.Connection):
        self.conn = conn
        self.validation_count = 0
        self.validation_cache: Dict[str, Tuple[bool, str]] = {}
    
    def validate_syntax(self, query: str) -> Tuple[bool, str]:
        """
        Validate query syntax using EXPLAIN.
        
        Args:
            query: Cypher query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        self.validation_count += 1
        
        if query in self.validation_cache:
            return self.validation_cache[query]
        
        try:
            explain_query = f"EXPLAIN {query}"
            self.conn.execute(explain_query)
            result = (True, "Valid query")
            self.validation_cache[query] = result
            return result
        except Exception as e:
            error_msg = str(e)
            result = (False, error_msg)
            self.validation_cache[query] = result
            return result
    
    def get_stats(self) -> Dict:
        """Get validation statistics."""
        return {
            "total_validations": self.validation_count,
            "cache_size": len(self.validation_cache),
            "cache_hit_rate": 1 - (self.validation_count / max(len(self.validation_cache), 1))
        }


class RepairQuery(dspy.Signature):
    """
    DSPy signature for repairing invalid Cypher queries.
    Takes the original context plus error history to avoid repeating mistakes.
    """
    
    original_question: str = dspy.InputField(desc="The original user question")
    graph_schema: str = dspy.InputField(desc="The graph schema")
    examples: str = dspy.InputField(desc="Relevant examples of correct queries")
    previous_attempts: str = dspy.InputField(desc="History of failed attempts and their errors")
    current_error: str = dspy.InputField(desc="The specific error from the last attempt")
    
    repaired_query: str = dspy.OutputField(desc="The corrected Cypher query")


class SelfRefiningText2Cypher:
    """
    Self-refining Text2Cypher module that validates and repairs queries.
    Implements generate -> validate -> repair loop with history tracking.
    """
    
    def __init__(
        self,
        text2cypher_module,
        validator: QueryValidator,
        max_attempts: int = 3,
        enable_repair: bool = True,
        enable_post_processing: bool = True,
        enable_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize the self-refining module.
        
        Args:
            text2cypher_module: The base Text2Cypher DSPy module
            validator: QueryValidator instance
            max_attempts: Maximum number of repair attempts
            enable_repair: Whether to enable automatic repair
            enable_post_processing: Whether to enable rule-based post-processing
            enable_cache: Whether to enable LRU caching of queries
            cache_size: Maximum number of cached queries (default: 1000)
        """
        self.text2cypher = text2cypher_module
        self.validator = validator
        self.max_attempts = max_attempts
        self.enable_repair = enable_repair
        self.enable_post_processing = enable_post_processing
        self.enable_cache = enable_cache
        self.repair_module = dspy.ChainOfThought(RepairQuery) if enable_repair else None
        self.post_processor = CypherPostProcessor() if enable_post_processing else None
        self.cache = Text2CypherLRUCache(max_size=cache_size) if enable_cache else None
        
        self.stats = {
            "total_queries": 0,
            "successful_first_try": 0,
            "successful_after_repair": 0,
            "failed_after_max_attempts": 0,
            "total_repair_attempts": 0,
            "queries_post_processed": 0,
            "post_processing_fixed_errors": 0
        }
    
    def generate_and_validate(
        self,
        question: str,
        schema: str,
        examples: str
    ) -> Tuple[Optional[str], QueryHistory]:
        """
        Generate a Cypher query and validate it, with automatic repair if needed.
        
        Args:
            question: The user's question
            schema: The graph schema
            examples: Formatted exemplar examples
            
        Returns:
            Tuple of (final_query, history)
        """
        self.stats["total_queries"] += 1
        history = QueryHistory()
        
        if self.enable_cache and self.cache:
            cached_query = self.cache.get(question, schema)
            if cached_query:
                history.add_attempt(cached_query, True, None)
                self.stats["successful_first_try"] += 1
                return cached_query, history
        
        try:
            initial_result = self.text2cypher(
                question=question,
                input_schema=schema,
                examples=examples
            )
            initial_query = initial_result.query.query
        except Exception as e:
            history.add_attempt("", False, f"Generation failed: {str(e)}")
            self.stats["failed_after_max_attempts"] += 1
            return None, history
        
        if self.enable_post_processing and self.post_processor:
            processed_query, applied_rules = self.post_processor.process(initial_query)
            if applied_rules:
                self.stats["queries_post_processed"] += 1
                is_valid_before = self.validator.validate_syntax(initial_query)[0]
                is_valid_after = self.validator.validate_syntax(processed_query)[0]
                if not is_valid_before and is_valid_after:
                    self.stats["post_processing_fixed_errors"] += 1
            initial_query = processed_query
        
        is_valid, error_message = self.validator.validate_syntax(initial_query)
        history.add_attempt(initial_query, is_valid, error_message if not is_valid else None)
        
        if is_valid:
            self.stats["successful_first_try"] += 1
            
            if self.enable_cache and self.cache:
                self.cache.put(
                    question,
                    schema,
                    initial_query,
                    is_valid=True,
                    metadata={"repair_attempts": 0, "post_processing": self.enable_post_processing}
                )
            
            return initial_query, history
        
        if not self.enable_repair:
            self.stats["failed_after_max_attempts"] += 1
            return None, history
        
        current_query = initial_query
        current_error = error_message
        
        for attempt_num in range(2, self.max_attempts + 1):
            self.stats["total_repair_attempts"] += 1
            
            try:
                repair_result = self.repair_module(
                    original_question=question,
                    graph_schema=schema,
                    examples=examples,
                    previous_attempts=history.format_for_prompt(),
                    current_error=current_error
                )
                
                repaired_query = repair_result.repaired_query
                
                if self.enable_post_processing and self.post_processor:
                    repaired_query, applied_rules = self.post_processor.process(repaired_query)
                    if applied_rules:
                        self.stats["queries_post_processed"] += 1
                
                is_valid, error_message = self.validator.validate_syntax(repaired_query)
                history.add_attempt(repaired_query, is_valid, error_message if not is_valid else None)
                
                if is_valid:
                    self.stats["successful_after_repair"] += 1
                    
                    if self.enable_cache and self.cache:
                        self.cache.put(
                            question,
                            schema,
                            repaired_query,
                            is_valid=True,
                            metadata={"repair_attempts": attempt_num - 1, "post_processing": self.enable_post_processing}
                        )
                    
                    return repaired_query, history
                
                current_query = repaired_query
                current_error = error_message
                
            except Exception as e:
                history.add_attempt("", False, f"Repair failed: {str(e)}")
                break
        
        self.stats["failed_after_max_attempts"] += 1
        return None, history
    
    def get_stats(self) -> Dict:
        """Get statistics about the refinement process."""
        total = self.stats["total_queries"]
        if total == 0:
            stats = self.stats.copy()
            if self.enable_cache and self.cache:
                stats["cache"] = self.cache.get_statistics()
            return stats
        
        stats = {
            **self.stats,
            "first_try_success_rate": self.stats["successful_first_try"] / total,
            "repair_success_rate": self.stats["successful_after_repair"] / max(total - self.stats["successful_first_try"], 1),
            "overall_success_rate": (self.stats["successful_first_try"] + self.stats["successful_after_repair"]) / total,
            "avg_repair_attempts": self.stats["total_repair_attempts"] / max(total - self.stats["successful_first_try"], 1)
        }
        
        # Add cache statistics if enabled
        if self.enable_cache and self.cache:
            stats["cache"] = self.cache.get_statistics()
        
        return stats


def create_self_refining_text2cypher(
    conn: kuzu.Connection,
    text2cypher_module,
    max_attempts: int = 3,
    enable_repair: bool = True,
    enable_post_processing: bool = True,
    enable_cache: bool = True,
    cache_size: int = 1000
) -> SelfRefiningText2Cypher:
    """
    Factory function to create a self-refining Text2Cypher module.
    
    Args:
        conn: Kuzu database connection
        text2cypher_module: The base Text2Cypher module
        max_attempts: Maximum repair attempts
        enable_repair: Whether to enable repair
        enable_post_processing: Whether to enable rule-based post-processing
        enable_cache: Enable LRU caching of queries
        cache_size: Maximum cache size (default: 1000)
        
    Returns:
        SelfRefiningText2Cypher instance
    """
    validator = QueryValidator(conn)
    return SelfRefiningText2Cypher(
        text2cypher_module=text2cypher_module,
        validator=validator,
        max_attempts=max_attempts,
        enable_repair=enable_repair,
        enable_post_processing=enable_post_processing,
        enable_cache=enable_cache,
        cache_size=cache_size
    )
