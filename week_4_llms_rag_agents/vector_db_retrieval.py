"""Dependency-light local vector retrieval with a production-friendly interface."""

from dataclasses import dataclass
import hashlib
from typing import Callable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


class LocalVectorStore:
    """In-memory cosine-similarity store; replace the backend behind this API."""

    def __init__(self, embedding_function: Callable[[Sequence[str]], np.ndarray]) -> None:
        self.embedding_function = embedding_function
        self._chunks: List[DocumentChunk] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("chunks must contain non-empty text")
        embeddings = np.asarray(self.embedding_function([chunk.text for chunk in chunks]), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("embedding function must return one vector per chunk")
        self._chunks.extend(chunks)
        self._embeddings = embeddings if not self._chunks[:-len(chunks)] else np.vstack([self._embeddings, embeddings])

    def search(self, query: str, top_k: int = 3) -> List[tuple[DocumentChunk, float]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        if not self._chunks:
            return []
        query_vector = np.asarray(self.embedding_function([query]), dtype=np.float32)[0]
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vector)
        similarities = np.divide(self._embeddings @ query_vector, norms, out=np.zeros_like(norms), where=norms != 0)
        indices = np.argsort(-similarities)[:top_k]
        return [(self._chunks[index], float(similarities[index])) for index in indices]


def chunk_document(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """Split text by words with bounded overlap."""
    if not text.strip() or chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("text, chunk_size, and overlap are invalid")
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[start:start + chunk_size]) for start in range(0, len(words), step)]


def hashing_embeddings(texts: Sequence[str], dimensions: int = 128) -> np.ndarray:
    """Deterministic local baseline embedding using stable token hashes."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    matrix = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            matrix[row, int.from_bytes(digest[:8], "little") % dimensions] += 1.0
        norm = np.linalg.norm(matrix[row])
        if norm:
            matrix[row] /= norm
    return matrix
