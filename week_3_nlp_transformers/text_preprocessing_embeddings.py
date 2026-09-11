"""Text normalization, TF-IDF, and optional dense embeddings."""

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str) -> str:
    """Lowercase text, remove punctuation, and normalize whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text.lower())).strip()


def tfidf_embeddings(documents: List[str], max_features: int = 5000) -> tuple[np.ndarray, TfidfVectorizer]:
    """Generate a dense TF-IDF matrix and fitted vectorizer."""
    if not documents or any(not isinstance(document, str) for document in documents):
        raise ValueError("documents must be a non-empty list of strings")
    if max_features < 1:
        raise ValueError("max_features must be positive")
    vectorizer = TfidfVectorizer(preprocessor=normalize_text, max_features=max_features)
    return vectorizer.fit_transform(documents).toarray(), vectorizer


def word_embedding_lookup(tokens: List[str], embedding_dim: int = 32, seed: int = 42) -> np.ndarray:
    """Create deterministic demonstration embeddings for a token list."""
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")
    generator = np.random.default_rng(seed)
    return generator.normal(0, 1, size=(len(tokens), embedding_dim)).astype(np.float32)
