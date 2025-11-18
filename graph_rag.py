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
def _(BAMLAdapter, OPENROUTER_API_KEY, dspy):
    # Using OpenRouter. Switch to another LLM provider as needed
    lm = dspy.LM(
        model="openrouter/google/gemini-2.0-flash-001",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    dspy.configure(lm=lm, adapter=BAMLAdapter())
    return


@app.cell
def _(OPENROUTER_API_KEY):
    import requests
    import numpy as np

    def get_embedding(text: str):
        """Call OpenRouter embedding model."""
        url = "https://openrouter.ai/api/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/text-embedding-3-small",
            "input": text
        }

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        return np.array(data["data"][0]["embedding"])

    return get_embedding, np

@app.cell
def _(get_embedding, np):
    import json

    with open("few_shot_examples.json") as f:
        EXAMPLE_BANK = json.load(f)

    # Precompute embeddings
    for ex in EXAMPLE_BANK:
        ex["embedding"] = get_embedding(ex["question"]).tolist()

    return EXAMPLE_BANK


@app.cell
def _(EXAMPLE_BANK, get_embedding, np):
    def cosine_similarity(a, b):
        """Compute cosine similarity between 2 vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def select_few_shot_examples(user_question, k=3):
        q_vec = get_embedding(user_question)

        scored = []
        for ex in EXAMPLE_BANK:
            ex_vec = np.array(ex["embedding"])
            score = cosine_similarity(q_vec, ex_vec)
            scored.append((score, ex))

        scored.sort(key=lambda x: -x[0])
        return [ex for _, ex in scored[:k]]

    def build_fewshot_prompt(selected_examples):
        block = ""
        for ex in selected_examples:
            block += (
                f"Example Question: {ex['question']}\n"
                f"Example Cypher:\n```cypher\n{ex['cypher']}\n```\n\n"
            )
        return block.strip()

    return select_few_shot_examples, build_fewshot_prompt



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

    query = result['query']
    answer = result['answer'].response
    return answer, query


@app.cell
def _(answer, mo, query):
    mo.hstack([mo.md(f"""### Query\n```{query}```"""), mo.md(f"""### Answer\n{answer}""")])
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
def _(dspy, Text2Cypher, select_few_shot_examples, build_fewshot_prompt):

    class DynamicText2Cypher(dspy.Module):
        """
        Wrap Text2Cypher to inject dynamic few-shot exemplars.
        """

        def __init__(self):
            super().__init__()
            self.base = dspy.ChainOfThought(Text2Cypher)

        def forward(self, question: str, input_schema: str):
            # Pick top examples
            selected = select_few_shot_examples(question, k=3)
            fewshot_block = build_fewshot_prompt(selected)

            enhanced_question = (
                "Translate the question into Cypher using the graph schema.\n\n"
                "Here are relevant examples:\n\n"
                f"{fewshot_block}\n\n"
                "Now translate the following question:\n"
                f"{question}"
            )

            return self.base(question=enhanced_question, input_schema=input_schema)

    return DynamicText2Cypher

@app.cell
def _(KuzuDatabaseManager):
    def validate_cypher_syntax(db_manager: KuzuDatabaseManager, query: str):
        """
        Validate query using Kuzu EXPLAIN.
        Returns: (True, None) if valid, else (False, error_message)
        """
        try:
            db_manager.conn.execute(f"EXPLAIN {query}")
            return True, None
        except Exception as e:
            return False, str(e)

    return validate_cypher_syntax

@app.cell
def _(dspy):
    class RepairCypher(dspy.Signature):
        """
        Repair an invalid Cypher query using the error message.
        Return ONLY a corrected Cypher query.
        """
        question: str = dspy.InputField()
        bad_query: str = dspy.InputField()
        error_message: str = dspy.InputField()
        fixed_query: str = dspy.OutputField(description="Corrected Cypher query")

    repair_cypher = dspy.ChainOfThought(RepairCypher)

    return repair_cypher


@app.cell
def _(dspy, DynamicText2Cypher, validate_cypher_syntax, repair_cypher):
    class RefinedText2Cypher:
        """
        Self-refinement wrapper:
        1. Generate initial query
        2. Validate via EXPLAIN
        3. If invalid, repair and retry
        """
        def __init__(self, max_iters=3):
            self.max_iters = max_iters
            self.generator = DynamicText2Cypher()

        def generate(self, db_manager, question, input_schema):
            # First attempt
            result = self.generator(question=question, input_schema=input_schema)
            query = result.query

            # Self-refinement loop
            for _ in range(self.max_iters):
                ok, error = validate_cypher_syntax(db_manager, query)
                if ok:
                    return query  # Valid query!

                # Attempt repair using the repair LLM
                repaired = repair_cypher(
                    question=question,
                    bad_query=query,
                    error_message=error,
                )
                query = repaired.fixed_query

            return query  # Return last repaired version

        # ❤️ KEY FIX → Make module "callable" by DSPy
        def __call__(self, db_manager, question, input_schema):
            query = self.generate(db_manager, question, input_schema)
            return type("Obj", (), {"query": query})()


    return RefinedText2Cypher




@app.cell
def _(
    AnswerQuestion,
    Any,
    KuzuDatabaseManager,
    PruneSchema,
    Query,
    RefinedText2Cypher,
    dspy,
):
    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self):
            super().__init__()
            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = RefinedText2Cypher(max_iters=3)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)

        def get_cypher_query(
            self,
            db_manager: KuzuDatabaseManager,
            question: str,
            input_schema: str,
        ) -> Query:
            # 1. Prune schema based on the question
            prune_result = self.prune(question=question, input_schema=input_schema)
            pruned_schema = prune_result.pruned_schema

            # Convert pruned schema (GraphSchema Pydantic model) to JSON string
            # so the LLM can see a clean schema representation
            schema_str = pruned_schema.model_dump_json()

            # 2. Run self-refining Text2Cypher on the pruned schema
            t2c_result = self.text2cypher(
                db_manager=db_manager,
                question=question,
                input_schema=schema_str,
            )
            cypher_query = t2c_result.query

            # 3. Wrap in Query model (note: use keyword arg!)
            return Query(query=cypher_query)

        def run_query(
            self,
            db_manager: KuzuDatabaseManager,
            question: str,
            input_schema: str,
        ) -> tuple[str, list[Any] | None]:
            """
            Run a query synchronously on the database.
            """
            query_model = self.get_cypher_query(
                db_manager=db_manager,
                question=question,
                input_schema=input_schema,
            )
            query = query_model.query

            try:
                result = db_manager.conn.execute(query)
                results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None

            return query, results

        def forward(
            self,
            db_manager: KuzuDatabaseManager,
            question: str,
            input_schema: str,
        ):
            final_query, final_context = self.run_query(
                db_manager=db_manager,
                question=question,
                input_schema=input_schema,
            )
            if final_context is None:
                print(
                    "Empty results obtained from the graph database. "
                    "Please retry with a different question."
                )
                return {}
            else:
                answer = self.generate_answer(
                    question=question,
                    cypher_query=final_query,
                    context=str(final_context),
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response

        async def aforward(
            self,
            db_manager: KuzuDatabaseManager,
            question: str,
            input_schema: str,
        ):
            final_query, final_context = self.run_query(
                db_manager=db_manager,
                question=question,
                input_schema=input_schema,
            )
            if final_context is None:
                print(
                    "Empty results obtained from the graph database. "
                    "Please retry with a different question."
                )
                return {}
            else:
                answer = self.generate_answer(
                    question=question,
                    cypher_query=final_query,
                    context=str(final_context),
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response

    def run_graph_rag(questions: list[str], db_manager: KuzuDatabaseManager) -> list[Any]:
        # Full schema as a string for the PruneSchema signature
        schema = str(db_manager.get_schema_dict)
        rag = GraphRAG()

        results = []
        for question in questions:
            response = rag(
                db_manager=db_manager,
                question=question,
                input_schema=schema,
            )
            results.append(response)
        return results

    return (run_graph_rag,)


@app.cell
def _():
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

    load_dotenv()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Field,
        OPENROUTER_API_KEY,
        dspy,
        kuzu,
        mo,
    )


if __name__ == "__main__":
    app.run()
