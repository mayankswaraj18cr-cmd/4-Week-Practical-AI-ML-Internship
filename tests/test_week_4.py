import numpy as np
import pytest

from week_4_llms_rag_agents.agentic_workflow import AgentWorkflow, Tool
from week_4_llms_rag_agents.rag_pipeline import RAGPipeline
from week_4_llms_rag_agents.vector_db_retrieval import LocalVectorStore, chunk_document, hashing_embeddings


def test_chunking_and_retrieval() -> None:
    assert len(chunk_document("one two three four", chunk_size=2, overlap=0)) == 2
    store = LocalVectorStore(hashing_embeddings)
    from week_4_llms_rag_agents.vector_db_retrieval import DocumentChunk
    store.add([DocumentChunk("a", "cats purr", {}), DocumentChunk("b", "dogs bark", {})])
    assert store.search("cats", top_k=1)[0][0].chunk_id == "a"


def test_rag_injects_context() -> None:
    pipeline = RAGPipeline(LocalVectorStore(hashing_embeddings), lambda prompt: prompt)
    pipeline.index("doc", "Python is a programming language.", chunk_size=10)
    assert "Python" in pipeline.answer("What is Python?")


def test_agent_invokes_tool_and_finishes() -> None:
    calls = []
    def model(prompt: str, tools: list[str]) -> dict[str, object]:
        calls.append(prompt)
        return {"tool": "add", "arguments": {"left": 2, "right": 3}} if len(calls) == 1 else {"final": "5"}
    workflow = AgentWorkflow(model, [Tool("add", "add numbers", lambda left, right: left + right)])
    assert workflow.run("calculate") == "5"
    assert len(calls) == 2
