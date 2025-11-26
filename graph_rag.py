import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return


@app.cell
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager)[0]

    query = result["query"]
    answer_pred = result.get("answer")
    timings = result.get("timings", {})
    cache_stats = result.get("cache_stats", {})

    answer = (
        answer_pred.response
        if answer_pred is not None
        else "No answer (empty context)."
    )
    return answer, query, timings, cache_stats


@app.cell
def _(answer, mo, query, timings, cache_stats):
    if timings:
        timing_lines = "\n".join(
            f"- **{stage}**: {t*1000:.1f} ms" for stage, t in timings.items()
        )
        timing_md = "### Timing breakdown\n" + timing_lines
    else:
        timing_md = "### Timing breakdown\nNo timing data."
    
    if cache_stats:
        hit_rate = cache_stats.get("hit_rate", 0)
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        size = cache_stats.get("current_size", 0)
        max_size = cache_stats.get("max_size", 0)
        cache_md = f"\n\n### Cache Statistics\n- **Hit Rate**: {hit_rate:.1%}\n- **Hits/Misses**: {hits}/{misses}\n- **Cache Size**: {size}/{max_size}"
    else:
        cache_md = ""

    mo.hstack(
        [
            mo.md(f"""### Query\n```{query}```"""),
            mo.md(f"""### Answer\n{answer}\n\n{timing_md}{cache_md}"""),
        ]
    )
    return


@app.cell
def _(GraphSchema, Query, dspy):
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


    class Text2Cypher(dspy.Signature):
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
    return AnswerQuestion, PruneSchema, Text2Cypher


@app.cell
def _(BAMLAdapter, OPENROUTER_API_KEY, dspy):
    # Using OpenRouter. Switch to another LLM provider as needed
    lm = dspy.LM(
        model="openrouter/tngtech/deepseek-r1t2-chimera:free",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    dspy.configure(lm=lm, adapter=BAMLAdapter())
    return


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "ldbc_1.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]  # type: ignore
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]  # type: ignore
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:  # type: ignore
                    node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:  # type: ignore
                    edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
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
    return GraphSchema, Query


@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    Text2Cypher,
    create_self_refining_text2cypher,
    dspy,
    select_exemplars,
):
    import time

    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self, db_manager: KuzuDatabaseManager):
            super().__init__()
            self.db_manager = db_manager
            self.prune = dspy.Predict(PruneSchema)
            
            base_text2cypher = dspy.ChainOfThought(Text2Cypher)
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

        def get_cypher_query(
            self,
            question: str,
            input_schema: str,
            timings: dict[str, float] | None = None,
        ) -> tuple[str | None, Any]:
            t_prune_start = time.perf_counter()
            prune_result = self.prune(question=question, input_schema=input_schema)
            schema = prune_result.pruned_schema
            if timings is not None:
                timings["prune"] = time.perf_counter() - t_prune_start
            
            selected_exemplars = select_exemplars(question, top_k=5)
            exemplar_text = "\n".join([
                f"Example {i+1}:\nQuestion: {ex['question']}\nCypher: {ex['cypher']}\n"
                for i, ex in enumerate(selected_exemplars)
            ])
            
            t_t2c_start = time.perf_counter()
            query, history = self.refining_text2cypher.generate_and_validate(
                question=question,
                schema=schema,
                examples=exemplar_text
            )
            if timings is not None:
                timings["text2cypher"] = time.perf_counter() - t_t2c_start
            
            return query, history

        def run_query(
            self,
            db_manager: KuzuDatabaseManager,
            question: str,
            input_schema: str,
            timings: dict[str, float] | None = None,
        ) -> tuple[str, list[Any] | None]:
            """
            Run a query synchronously on the database.
            """
            query, history = self.get_cypher_query(
                question=question,
                input_schema=input_schema,
                timings=timings,
            )
            
            if query is None:
                print(f"Failed to generate valid query after {len(history.attempts)} attempts")
                return None, None
            
            t_db = time.perf_counter()
            try:
                result = db_manager.conn.execute(query)
                results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None
            if timings is not None:
                timings["db_query"] = time.perf_counter() - t_db
            return query, results

        def forward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            timings: dict[str, float] = {}
            t_total = time.perf_counter()

            final_query, final_context = self.run_query(
                db_manager, question, input_schema, timings=timings
            )
            if final_context is None or final_query is None:
                print("Empty results obtained from the graph database. Please retry with a different question.")
                timings["total"] = time.perf_counter() - t_total
                return {
                    "question": question,
                    "query": final_query,
                    "answer": None,
                    "timings": timings,
                }

            t_answer = time.perf_counter()
            answer = self.generate_answer(
                question=question, cypher_query=final_query, context=str(final_context)
            )
            timings["answer_generation"] = time.perf_counter() - t_answer
            timings["total"] = time.perf_counter() - t_total

            pipeline_stats = self.refining_text2cypher.get_stats()
            cache_stats = pipeline_stats.get("cache", {})

            response = {
                "question": question,
                "query": final_query,
                "answer": answer,
                "timings": timings,
                "cache_stats": cache_stats,
            }
            return response

        async def aforward(self, db_manager: KuzuDatabaseManager, question: str, input_schema: str):
            return self.forward(db_manager, question, input_schema)


    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager) -> list[Any]:
        schema = str(db_manager.get_schema_dict)
        rag = GraphRAG(db_manager=db_manager)
        # Run pipeline
        results = []
        for question in questions:
            response = rag(db_manager=db_manager, question=question, input_schema=schema)
            results.append(response)
        return results

    return (run_graph_rag,)


@app.cell
def _(mo, timings):
    if not timings:
        mo.md("No timing data.")
    else:
        total = timings.get("total", sum(timings.values()))
        rows = []
        for stage, t in timings.items():
            if stage == "total":
                continue
            pct = (t / total) * 100 if total > 0 else 0
            rows.append(f"{stage:18} | {'█' * int(pct/5)} {t*1000:.1f} ms ({pct:.1f}%)")
        text = "```text\n" + "\n".join(rows) + "\n```"
        mo.md("### Flamegraph-style view\n" + text)
    return


@app.cell
def _():
    import marimo as mo
    import os
    from textwrap import dedent
    from typing import Any

    import dspy
    import kuzu
    from dotenv import load_dotenv
    from dspy.adapters.baml_adapter import BAMLAdapter
    from pydantic import BaseModel, Field
    from exemplar_selector import select_exemplars
    from self_refining_text2cypher import create_self_refining_text2cypher

    load_dotenv()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Field,
        OPENROUTER_API_KEY,
        create_self_refining_text2cypher,
        dspy,
        kuzu,
        mo,
        select_exemplars,
    )


if __name__ == "__main__":
    app.run()
