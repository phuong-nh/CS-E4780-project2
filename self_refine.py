from typing import Any


class RefinedText2Cypher:
    """
    Self-refinement wrapper:
    1. Generate initial query via generator
    2. Validate via EXPLAIN
    3. If invalid, repair and retry
    """

    def __init__(self, generator, validator, repairer, postprocessor, max_iters=3):
        """
        generator: module that generates Cypher from text
        validator: validate_cypher_syntax(db_manager, query) -> (bool, error_message)
        repairer: repair_cypher(question, bad_query, error_message).fixed_query
        postprocessor: postprocess_cypher(query)
        """
        self.generator = generator
        self.validator = validator
        self.repairer = repairer
        self.postprocess = postprocessor
        self.max_iters = max_iters

    def generate(self, db_manager, question: str, input_schema: str) -> str:
        # 1. initial generation
        result = self.generator(question=question, input_schema=input_schema)
        query = result.query

        # 2. refinement loop
        for _ in range(self.max_iters):
            ok, error = self.validator(db_manager, query)
            if ok:
                return self.postprocess(query)

            # repair
            repaired = self.repairer(
                question=question,
                bad_query=query,
                error_message=error,
            )
            query = self.postprocess(repaired.fixed_query)

        # end of loop → return last attempt
        return self.postprocess(query)

    def __call__(self, db_manager, question, input_schema):
        query = self.generate(db_manager, question, input_schema)
        # DSPy contracts expect an object with `.query`
        return type("Result", (), {"query": query})()
