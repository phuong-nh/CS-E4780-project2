"""
Comprehensive GraphRAG Pipeline Benchmarking Script

Compares baseline (no enhancements) vs enhanced (with all Task 1 & 2 features) pipelines.
Measures timing at deep granularity and stores results with answers for accuracy comparison.
"""

import os
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

import dspy
import kuzu
from dotenv import load_dotenv
from dspy.adapters.baml_adapter import BAMLAdapter
from pydantic import BaseModel, Field

from exemplar_selector import select_exemplars
from self_refining_text2cypher import create_self_refining_text2cypher

load_dotenv()

BENCHMARK_QUESTIONS = [
    "How many laureates received prizes in Literature?",
    "Which scholars won both Physics and Chemistry prizes?",
    "List all laureates who died after 2010.",
    "Who are the laureates affiliated with Stanford University?",
    "How many male Medicine laureates are there?",
    "List scholars who were born in the United States.",
    "Which laureates were born in Sweden?",
    "What institutions in Switzerland have affiliated laureates?",
    "How many laureates were born in Canada?",
    "List all prizes awarded between 1960 and 1970.",
]


class Query(BaseModel):
    query: str = Field(description="Valid Cypher query with no newlines")


class Property(BaseModel):
    name: str
    type: str = Field(description="Data type of the property")


class Node(BaseModel):
    label: str
    properties: list[Property] | None


class Edge(BaseModel):
    label: str = Field(description="Relationship label")
    from_: Node = Field(alias="from", description="Source node label")
    to: Node = Field(alias="from", description="Target node label")
    properties: list[Property] | None


class GraphSchema(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


class PruneSchema(dspy.Signature):
    """
    Understand the given labelled property graph schema and the given user question. Your task
    is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
    relevant to the question.
        - The schema is a list of nodes and edges in a property graph.
        - The nodes are the entities in the graph.
        - The edges are the relationships between the nodes.
        - Properties of nodes and edges are their attributes, which helps answer the question.
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    pruned_schema: GraphSchema = dspy.OutputField()

class Text2CypherBase(dspy.Signature):
    """
    Translate the question into a valid Cypher query that respects the graph schema.

    <SYNTAX>
    - When matching on Scholar names, ALWAYS match on the `knownName` property
    - For countries, cities, continents and institutions, you can match on the `name` property
    - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
    - Always strive to respect the relationship direction (FROM/TO) using the schema information.
    - When comparing string properties, ALWAYS do the following:
        - Lowercase the property values before comparison
        - Use the WHERE clause
        - Use the CONTAINS operator to check for presence of one substring in the other
    - DO NOT use APOC as the database does not support it.
    </SYNTAX>

    <RETURN_RESULTS>
    - If the result is an integer, return it as an integer (not a string).
    - When returning results, return property values rather than the entire node or relationship.
    - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
    - NO Cypher keywords should be returned by your query.
    </RETURN_RESULTS>
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    query: Query = dspy.OutputField()
    
class Text2CypherEnhanced(dspy.Signature):
    """
    Translate the question into a valid Cypher query that respects the graph schema.
    Use the provided examples to guide your query generation.

    <SYNTAX>
    - When matching on Scholar names, ALWAYS match on the `knownName` property
    - For countries, cities, continents and institutions, you can match on the `name` property
    - Use short, concise alphanumeric strings as names of variable bindings (e.g., `a1`, `r1`, etc.)
    - Always strive to respect the relationship direction (FROM/TO) using the schema information.
    - When comparing string properties, ALWAYS do the following:
        - Lowercase the property values before comparison
        - Use the WHERE clause
        - Use the CONTAINS operator to check for presence of one substring in the other
    - DO NOT use APOC as the database does not support it.
    </SYNTAX>

    <RETURN_RESULTS>
    - If the result is an integer, return it as an integer (not a string).
    - When returning results, return property values rather than the entire node or relationship.
    - Do not attempt to coerce data types to number formats (e.g., integer, float) in your results.
    - NO Cypher keywords should be returned by your query.
    </RETURN_RESULTS>
    """

    question: str = dspy.InputField()
    input_schema: str = dspy.InputField()
    examples: str = dspy.InputField(desc="Relevant examples of question-to-Cypher translations")
    query: Query = dspy.OutputField()


class AnswerQuestion(dspy.Signature):
    """
    - Use the provided question, the generated Cypher query and the context to answer the question.
    - If the context is empty, state that you don't have enough information to answer the question.
    - When dealing with dates, mention the month in full.
    """

    question: str = dspy.InputField()
    cypher_query: str = dspy.InputField()
    context: str = dspy.InputField()
    response: str = dspy.OutputField()


class KuzuDatabaseManager:
    """Manages Kuzu database connection and schema retrieval."""

    def __init__(self, db_path: str = "nobel.kuzu"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)

    @property
    def get_schema_dict(self) -> dict[str, list[dict]]:
        response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
        nodes = [row[1] for row in response]
        response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
        rel_tables = [row[1] for row in response]
        relationships = []
        for tbl_name in rel_tables:
            response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
            for row in response:
                relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})
        schema = {"nodes": [], "edges": []}

        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
            for row in node_properties:
                node_schema["properties"].append({"name": row[1], "type": row[2]})
            schema["nodes"].append(node_schema)

        for rel in relationships:
            edge = {
                "label": rel["name"],
                "from": rel["from"],
                "to": rel["to"],
                "properties": [],
            }
            rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
            for row in rel_properties:
                edge["properties"].append({"name": row[1], "type": row[2]})
            schema["edges"].append(edge)
        return schema


class BaselineGraphRAG:
    """Baseline GraphRAG without enhancements (no cache, no post-processing, no self-refinement)."""
    
    def __init__(self, db_manager: KuzuDatabaseManager):
        self.db_manager = db_manager
        self.prune = dspy.Predict(PruneSchema)
        self.text2cypher = dspy.ChainOfThought(Text2CypherBase)
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
    
    def run(self, question: str, schema: str) -> Dict[str, Any]:
        """Run baseline pipeline and return results with timing."""
        timings = {}
        t_total = time.perf_counter()
        
        # Prune schema
        t_prune = time.perf_counter()
        prune_result = self.prune(question=question, input_schema=schema)
        pruned_schema = prune_result.pruned_schema
        timings["prune"] = time.perf_counter() - t_prune
        
        timings["exemplar_selection"] = 0.0
        
        # Text2Cypher (basic)
        t_t2c = time.perf_counter()
        try:
            result = self.text2cypher(
                question=question,
                input_schema=str(pruned_schema),
            )
            query = result.query.query
        except Exception as e:
            timings["text2cypher"] = time.perf_counter() - t_t2c
            timings["total"] = time.perf_counter() - t_total
            return {
                "question": question,
                "query": None,
                "context": None,
                "answer": None,
                "timings": timings,
                "error": str(e)
            }
        timings["text2cypher"] = time.perf_counter() - t_t2c
        
        # Execute query
        t_db = time.perf_counter()
        try:
            result = self.db_manager.conn.execute(query)
            context = [item for row in result for item in row]
        except Exception as e:
            context = None
        timings["db_query"] = time.perf_counter() - t_db
        
        # Generate answer
        if context is not None:
            t_answer = time.perf_counter()
            answer = self.generate_answer(
                question=question,
                cypher_query=query,
                context=str(context)
            )
            timings["answer_generation"] = time.perf_counter() - t_answer
        else:
            answer = None
            timings["answer_generation"] = 0.0
        
        timings["total"] = time.perf_counter() - t_total
        
        return {
            "question": question,
            "query": query,
            "context": context,
            "answer": answer.response if answer else None,
            "timings": timings
        }


class EnhancedGraphRAG:
    """Enhanced GraphRAG with all Task 1 & 2 features."""
    
    def __init__(self, db_manager: KuzuDatabaseManager):
        self.db_manager = db_manager
        self.prune = dspy.Predict(PruneSchema)
        
        # Enhanced Text2Cypher with all features
        base_text2cypher = dspy.ChainOfThought(Text2CypherEnhanced)
        self.refining_text2cypher = create_self_refining_text2cypher(
            conn=db_manager.conn,
            text2cypher_module=base_text2cypher,
            max_attempts=3,
            enable_repair=True,
            enable_post_processing=True,
            enable_cache=True,
            cache_size=100
        )
        
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
    
    def run(self, question: str, schema: str) -> Dict[str, Any]:
        """Run enhanced pipeline and return results with deep timing."""
        timings = {}
        t_total = time.perf_counter()
        
        # Prune schema
        t_prune = time.perf_counter()
        prune_result = self.prune(question=question, input_schema=schema)
        pruned_schema = prune_result.pruned_schema
        timings["prune"] = time.perf_counter() - t_prune
        
        # Exemplar selection
        t_exemplar = time.perf_counter()
        selected_exemplars = select_exemplars(question, top_k=5)
        exemplar_text = "\n".join([
            f"Example {i+1}:\nQuestion: {ex['question']}\nCypher: {ex['cypher']}\n"
            for i, ex in enumerate(selected_exemplars)
        ])
        timings["exemplar_selection"] = time.perf_counter() - t_exemplar
        
        # Text2Cypher with self-refinement, post-processing, and cache
        t2c_timings = {}
        query, history = self.refining_text2cypher.generate_and_validate(
            question=question,
            schema=str(pruned_schema),
            examples=exemplar_text,
            timings=t2c_timings
        )
        timings["text2cypher_breakdown"] = t2c_timings
        
        # Execute query
        t_db = time.perf_counter()
        if query:
            try:
                result = self.db_manager.conn.execute(query)
                context = [item for row in result for item in row]
            except Exception as e:
                context = None
        else:
            context = None
        timings["db_query"] = time.perf_counter() - t_db
        
        # Generate answer
        if context is not None:
            t_answer = time.perf_counter()
            answer = self.generate_answer(
                question=question,
                cypher_query=query,
                context=str(context)
            )
            timings["answer_generation"] = time.perf_counter() - t_answer
        else:
            answer = None
            timings["answer_generation"] = 0.0
        
        timings["total"] = time.perf_counter() - t_total
        
        # Get pipeline stats
        pipeline_stats = self.refining_text2cypher.get_stats()
        
        return {
            "question": question,
            "query": query,
            "context": context,
            "answer": answer.response if answer else None,
            "timings": timings,
            "pipeline_stats": pipeline_stats,
            "query_history": [a.model_dump() for a in history.attempts]
        }


def run_benchmark(num_runs: int = 3) -> Dict[str, Any]:
    """Run comprehensive benchmark comparing baseline vs enhanced."""
    
    print("="*80)
    print("GraphRAG Pipeline Benchmark")
    print("="*80)
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")
    print(f"Runs per question: {num_runs}")
    print(f"Total runs: {len(BENCHMARK_QUESTIONS) * num_runs * 2}")
    print("="*80)
    
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    lm = dspy.LM(
        model="openrouter/google/gemini-2.5-flash-lite",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        max_tokens=100000,
    )
    dspy.configure(lm=lm, adapter=BAMLAdapter())
    
    db_manager = KuzuDatabaseManager("nobel.kuzu")
    schema = str(db_manager.get_schema_dict)
    
    print("\nInitializing pipelines...")
    baseline = BaselineGraphRAG(db_manager)
    enhanced = EnhancedGraphRAG(db_manager)
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(BENCHMARK_QUESTIONS),
            "runs_per_question": num_runs,
            "model": "google/gemini-2.5-flash-lite"
        },
        "baseline": [],
        "enhanced": []
    }
    
    # Run baseline
    print("\n" + "="*80)
    print("BASELINE PIPELINE (No Enhancements)")
    print("="*80)
    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[{i}/{len(BENCHMARK_QUESTIONS)}] {question}")
        run_results = []
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)
            result = baseline.run(question, schema)
            run_results.append(result)
            print(f"  ({result['timings']['total']*1000:.0f}ms)")
        results["baseline"].append({
            "question": question,
            "runs": run_results
        })
    
    # Run enhanced
    print("\n" + "="*80)
    print("ENHANCED PIPELINE")
    print("="*80)
    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n[{i}/{len(BENCHMARK_QUESTIONS)}] {question}")
        run_results = []
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)
            result = enhanced.run(question, schema)
            run_results.append(result)
            print(f"  ({result['timings']['total']*1000:.0f}ms)")
        results["enhanced"].append({
            "question": question,
            "runs": run_results
        })
    
    return results


def save_results(results: Dict[str, Any]):
    """Save benchmark results to files."""
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = output_dir / f"benchmark_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {results_file}")


def main():
    """Main benchmark execution."""
    
    print("\n" + "="*80)
    print("Starting GraphRAG Benchmark")
    print("="*80)
    
    results = run_benchmark(num_runs=3)
    
    save_results(results)
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
