import re
from typing import Dict, List, Tuple, Optional


class CypherPostProcessor:
    """
    Post-processes generated Cypher queries to fix common errors.
    Applies rule-based transformations before validation.
    """
    
    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize the post-processor.
        
        Args:
            schema: Graph schema for validation (optional)
        """
        self.schema = schema
        self.stats = {
            "total_queries": 0,
            "queries_modified": 0,
            "rules_applied": {},
        }
    
    def process(self, query: str) -> Tuple[str, List[str]]:
        """
        Apply all post-processing rules to the query.
        
        Args:
            query: The Cypher query to process
            
        Returns:
            Tuple of (processed_query, list_of_applied_rules)
        """
        self.stats["total_queries"] += 1
        original_query = query
        applied_rules = []
        
        query = self._normalize_whitespace(query, applied_rules)
        query = self._remove_extra_semicolons(query, applied_rules)
        query = self._normalize_quotes(query, applied_rules)
        query = self._fix_count_syntax(query, applied_rules)
        query = self._fix_lowercase_comparisons(query, applied_rules)
        query = self._fix_string_contains(query, applied_rules)
        query = self._fix_property_names(query, applied_rules)
        query = self._normalize_country_names(query, applied_rules)
        query = self._fix_relationship_direction(query, applied_rules)
        query = self._add_missing_return(query, applied_rules)
        
        if query != original_query:
            self.stats["queries_modified"] += 1
        
        for rule in applied_rules:
            self.stats["rules_applied"][rule] = self.stats["rules_applied"].get(rule, 0) + 1
        
        return query, applied_rules
    
    def _normalize_whitespace(self, query: str, applied_rules: List[str]) -> str:
        processed = re.sub(r'\s+', ' ', query.strip())
        
        if processed != query:
            applied_rules.append("normalize_whitespace")
        
        return processed
    
    def _remove_extra_semicolons(self, query: str, applied_rules: List[str]) -> str:
        processed = query.rstrip(';').strip()
        
        if processed != query:
            applied_rules.append("remove_extra_semicolons")
        
        return processed
    
    def _normalize_quotes(self, query: str, applied_rules: List[str]) -> str:
        processed = query
        
        pattern = r'"([^"]*)"'
        
        def replace_quotes(match):
            content = match.group(1)
            if "'" in content or content.strip() in ["", " "]:
                return match.group(0)
            return f"'{content}'"
        
        processed = re.sub(pattern, replace_quotes, query)
        
        if processed != query:
            applied_rules.append("normalize_quotes")
        
        return processed
    
    def _fix_count_syntax(self, query: str, applied_rules: List[str]) -> str:
        processed = query
        
        processed = re.sub(
            r'\bCOUNT\s+(\w+)\b',
            r'count(\1)',
            processed,
            flags=re.IGNORECASE
        )
        
        processed = re.sub(
            r'count\(DISTINCT\(([^)]+)\)\)',
            r'count(DISTINCT \1)',
            processed,
            flags=re.IGNORECASE
        )
        
        if processed != query:
            applied_rules.append("fix_count_syntax")
        
        return processed
    
    def _fix_lowercase_comparisons(self, query: str, applied_rules: List[str]) -> str:
        processed = query
        
        processed = re.sub(r'\band\b', 'AND', processed)
        processed = re.sub(r'\bor\b', 'OR', processed)
        processed = re.sub(r'\bnot\b', 'NOT', processed)
        
        if processed != query:
            applied_rules.append("fix_lowercase_comparisons")
        
        return processed
    
    def _fix_string_contains(self, query: str, applied_rules: List[str]) -> str:
        """
        Convert Python 'in' operator to Cypher CONTAINS.
        Example: 'Einstein' in n.name -> n.name CONTAINS 'Einstein'
        """
        processed = query
        
        pattern = r"(['\"])([^'\"]+)\1\s+in\s+(\w+\.\w+)"
        
        def replace_with_contains(match):
            quote = match.group(1)
            value = match.group(2)
            prop = match.group(3)
            return f"{prop} CONTAINS {quote}{value}{quote}"
        
        processed = re.sub(pattern, replace_with_contains, processed, flags=re.IGNORECASE)
        
        if processed != query:
            applied_rules.append("fix_string_contains")
        
        return processed
    
    def _fix_property_names(self, query: str, applied_rules: List[str]) -> str:
        """Fix common property name mistakes (lowercase to camelCase)."""
        processed = query
        
        property_fixes = {
            r'\bfirstname\b': 'firstName',
            r'\blastname\b': 'lastName',
            r'\bbirthdate\b': 'birthDate',
            r'\bdeathdate\b': 'deathDate',
            r'\bbirthcountry\b': 'birthCountry',
            r'\bdeathcountry\b': 'deathCountry',
            r'\bbirthcity\b': 'birthCity',
            r'\bdeathcity\b': 'deathCity',
            r'\bknownname\b': 'knownName',
        }
        
        for pattern, replacement in property_fixes.items():
            new_processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
            if new_processed != processed:
                processed = new_processed
        
        if processed != query:
            applied_rules.append("fix_property_names")
        
        return processed
    
    def _normalize_country_names(self, query: str, applied_rules: List[str]) -> str:
        """Normalize various US country name variations to USA."""
        processed = query
        
        us_variations = [
            r"(['\"])United States of America\1",
            r"(['\"])United States\1",
            r"(['\"])U\.S\.A\.\1",
            r"(['\"])U\.S\.\1",
            r"(['\"])US\1",
            r"(['\"])America\1",
        ]
        
        for pattern in us_variations:
            new_processed = re.sub(pattern, r"\1us\1", processed, flags=re.IGNORECASE)
            if new_processed != processed:
                processed = new_processed
        
        if processed != query:
            applied_rules.append("normalize_country_names")
        
        return processed
    
    def _fix_relationship_direction(self, query: str, applied_rules: List[str]) -> str:
        """Ensure consistent relationship syntax."""
        processed = query
        
        processed = re.sub(r'-\s*\[\s*:', '-[:', processed)
        processed = re.sub(r':\s*([^\]]+)\s*\]', r':\1]', processed)
        processed = re.sub(r'\]\s*-\s*>', ']->', processed)
        processed = re.sub(r'<\s*-\s*\[', '<-[', processed)
        
        if processed != query:
            applied_rules.append("fix_relationship_direction")
        
        return processed
    
    def _add_missing_return(self, query: str, applied_rules: List[str]) -> str:
        processed = query
        
        if not re.search(r'\bRETURN\b', processed, re.IGNORECASE):
            if re.search(r'\bMATCH\b', processed, re.IGNORECASE):
                processed += ' RETURN *'
                applied_rules.append("add_missing_return")
        
        return processed
    
    def get_stats(self) -> Dict:
        total = self.stats["total_queries"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "modification_rate": self.stats["queries_modified"] / total,
            "most_common_rules": sorted(
                self.stats["rules_applied"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def reset_stats(self):
        self.stats = {
            "total_queries": 0,
            "queries_modified": 0,
            "rules_applied": {},
        }


def post_process_query(query: str, schema: Optional[Dict] = None) -> Tuple[str, List[str]]:
    """
    Convenience function to post-process a single query.
    
    Args:
        query: Cypher query to process
        schema: Optional graph schema
        
    Returns:
        Tuple of (processed_query, applied_rules)
    """
    processor = CypherPostProcessor(schema)
    return processor.process(query)
