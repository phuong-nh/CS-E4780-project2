import dspy
from typing import Callable, List, Dict


class DynamicText2Cypher(dspy.Module):
    """
    Wraps Text2Cypher to inject dynamically selected few-shot examples.
    """

    def __init__(
        self,
        text2cypher_signature: Callable,
        select_examples: Callable[[str, int], List[Dict]],
        build_prompt: Callable[[List[Dict]], str],
        k: int = 3,
    ):
        super().__init__()
        self.text2cypher_signature = text2cypher_signature
        self.select_examples = select_examples
        self.build_prompt = build_prompt
        self.k = k

        # Base DSPy predictor
        self.base = dspy.ChainOfThought(self.text2cypher_signature)

    def forward(self, question: str, input_schema: str):
        # 1. dynamically pick few-shot examples
        selected = self.select_examples(question, self.k)

        # 2. build the prompt
        fewshot_block = self.build_prompt(selected)

        enhanced_question = (
            "Translate the question into a Cypher query compatible with the graph schema.\n\n"
            "Here are relevant examples:\n\n"
            f"{fewshot_block}\n\n"
            "Now translate the following question into Cypher:\n"
            f"{question}"
        )

        # 3. run DSPy CoT
        return self.base(question=enhanced_question, input_schema=input_schema)
