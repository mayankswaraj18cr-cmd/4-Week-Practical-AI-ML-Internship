"""Retrieval-augmented prompt construction."""

from typing import Callable, List

from .vector_db_retrieval import DocumentChunk, LocalVectorStore, chunk_document


class RAGPipeline:
    """Index documents, retrieve evidence, and inject it into a prompt."""

    def __init__(self, store: LocalVectorStore, generator: Callable[[str], str]) -> None:
        self.store, self.generator = store, generator

    def index(self, document_id: str, text: str, metadata: dict[str, str] | None = None, chunk_size: int = 200) -> None:
        chunks = chunk_document(text, chunk_size=chunk_size, overlap=min(20, chunk_size - 1))
        self.store.add([DocumentChunk(f"{document_id}-{index}", value, metadata or {}) for index, value in enumerate(chunks)])

    def build_prompt(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk, _ in results)
        return f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        return self.generator(self.build_prompt(question, top_k))
