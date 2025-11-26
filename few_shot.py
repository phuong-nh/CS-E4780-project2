# few_shot.py

import os
import json
import requests
import numpy as np


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def get_embedding(text: str) -> np.ndarray:
    """Call OpenRouter embedding model safely and return a vector."""
    if not OPENROUTER_API_KEY:
        # Fallback: zero vector if key is missing
        return np.zeros(1536, dtype=float)

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/text-embedding-3-small",
        "input": text,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if "error" in data:
            # Print error for debug, return zero vector to avoid crashes
            print("OpenRouter Embedding Error:", data["error"])
            return np.zeros(1536, dtype=float)

        return np.array(data["data"][0]["embedding"], dtype=float)
    except Exception as e:
        print("Embedding request failed:", e)
        return np.zeros(1536, dtype=float)


# Load few-shot example bank once
EXAMPLE_BANK_PATH = "few_shot_examples.json"
try:
    with open(EXAMPLE_BANK_PATH) as f:
        EXAMPLE_BANK = json.load(f)
except FileNotFoundError:
    print(f"[few_shot] Could not find {EXAMPLE_BANK_PATH}. Using empty example bank.")
    EXAMPLE_BANK = []


# Precompute embeddings for each example
for ex in EXAMPLE_BANK:
    if "question" in ex:
        ex["embedding"] = get_embedding(ex["question"]).tolist()
    else:
        ex["embedding"] = np.zeros(1536, dtype=float).tolist()


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between 2 vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def select_few_shot_examples(user_question: str, k: int = 3):
    """Return top-k examples from EXAMPLE_BANK that are closest to the question."""
    if not EXAMPLE_BANK:
        return []

    q_vec = get_embedding(user_question)
    scored = []

    for ex in EXAMPLE_BANK:
        ex_vec = np.array(ex.get("embedding", []), dtype=float)
        sim = cosine_similarity(q_vec, ex_vec)
        scored.append((sim, ex))

    scored.sort(key=lambda x: -x[0])
    return [ex for _, ex in scored[:k]]


def build_fewshot_prompt(selected_examples: list[dict]) -> str:
    """Create the prompt block containing few-shot examples."""
    block = ""
    for ex in selected_examples:
        q = ex.get("question", "")
        c = ex.get("cypher", "")
        block += (
            f"Example Question: {q}\n"
            f"Example Cypher:\n```cypher\n{c}\n```\n\n"
        )
    return block.strip()
