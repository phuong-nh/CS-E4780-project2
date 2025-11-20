import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from exemplars import EXEMPLARS


class ExemplarSelector:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', exemplars: List[Dict] = None):
        """
        Initialize the exemplar selector.
        
        Args:
            model_name: Name of the sentence transformer model to use
            exemplars: List of exemplar dictionaries with 'question' and 'cypher' keys
        """
        self.model = SentenceTransformer(model_name)
        self.exemplars = exemplars if exemplars is not None else EXEMPLARS
        
        self.exemplar_questions = [ex["question"] for ex in self.exemplars]
        self.exemplar_embeddings = self.model.encode(
            self.exemplar_questions,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def get_similar_exemplars(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Find the top-k most similar exemplars to the given question.
        
        Args:
            question: The input question to find similar exemplars for
            top_k: Number of top similar exemplars to return
            
        Returns:
            List of top-k most similar exemplar dictionaries
        """
        question_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        similarities = self._cosine_similarity(
            question_embedding,
            self.exemplar_embeddings
        )
        
        top_k = min(top_k, len(self.exemplars))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        selected_exemplars = []
        for idx in top_indices:
            exemplar = self.exemplars[idx].copy()
            exemplar['similarity_score'] = float(similarities[idx])
            selected_exemplars.append(exemplar)
        
        return selected_exemplars
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query embedding and all exemplar embeddings.
        
        Args:
            a: Query embedding of shape (1, dim)
            b: Exemplar embeddings of shape (n_exemplars, dim)
            
        Returns:
            Array of similarity scores of shape (n_exemplars,)
        """
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        similarities = np.dot(b_norm, a_norm.T).flatten()
        
        return similarities
    
    def format_exemplars_for_prompt(self, exemplars: List[Dict]) -> str:
        """
        Format selected exemplars into a string suitable for inclusion in a prompt.
        
        Args:
            exemplars: List of exemplar dictionaries
            
        Returns:
            Formatted string with exemplars
        """
        formatted = "Here are some relevant examples of question-to-Cypher translations:\n\n"
        
        for i, ex in enumerate(exemplars, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Question: {ex['question']}\n"
            formatted += f"Cypher: {ex['cypher']}\n"
            if 'similarity_score' in ex:
                formatted += f"(Similarity: {ex['similarity_score']:.3f})\n"
            formatted += "\n"
        
        return formatted


# Global instance for reuse
_exemplar_selector = None

def get_exemplar_selector() -> ExemplarSelector:
    global _exemplar_selector
    if _exemplar_selector is None:
        _exemplar_selector = ExemplarSelector()
    return _exemplar_selector


def select_exemplars(question: str, top_k: int = 3) -> List[Dict]:
    """
    Function to select exemplars for a question.
    
    Args:
        question: The input question
        top_k: Number of exemplars to select
        
    Returns:
        List of selected exemplars
    """
    selector = get_exemplar_selector()
    return selector.get_similar_exemplars(question, top_k)
