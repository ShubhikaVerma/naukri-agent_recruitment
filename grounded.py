"""Task 4b: grounded generation under MOCK_LLM.

Retrieval finds the relevant chunks; generation then works ONLY from the parent documents
of those chunks (parent-document expansion), so a fixed-size chunk that starts or ends
mid-sentence can never produce a broken or truncated answer. Below the calibrated
similarity threshold the agent refuses with an "I don't know" fallback."""
import json
import os
from pathlib import Path

import numpy as np

from knowledge_base import load_documents
from rag_index import (
    get_collections, search, embed, sentence_chunks,
    COLLECTION_FIXED, COLLECTION_SENTENCE,
)
from queries import DEMO_QUERIES, OUT_OF_SCOPE_DEMO

MOCK_LLM = os.environ.get("MOCK_LLM", "true").lower() == "true"
FALLBACK = "I don't know based on the available policy documents."
LLM_CALLS = {"count": 0}          # used later to prove cache hits (Task 16)
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
ANSWER_SENTENCES = 2              # how many sentences the mock LLM quotes

_doc_sentences = {}


def doc_sentences(doc_id: str) -> list:
    """All complete sentences of one knowledge-base document."""
    if not _doc_sentences:
        for d in load_documents():
            _doc_sentences[d["doc_id"]] = sentence_chunks(d["text"])
    return _doc_sentences[doc_id]

def reset_cache() -> None:
    """Forget cached document sentences (called after a document is added at runtime)."""
    _doc_sentences.clear()


def load_threshold(collection_name: str) -> float:
    data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if collection_name not in data:
        raise RuntimeError(
            "No calibrated threshold for {}; rerun calibrate.py".format(collection_name))
    return data[collection_name]["threshold"]


def mock_llm_generate(query: str, doc_ids: list) -> tuple:
    """Deterministic stand-in for an LLM. Extractive, so it can only repeat what the
    retrieved documents say: it returns the sentences most similar to the query.
    Returns (answer_text, source_doc_ids_of_the_sentences_used)."""
    if not MOCK_LLM:
        raise NotImplementedError("Wire a real LLM here behind MOCK_LLM=false (optional).")
    LLM_CALLS["count"] += 1

    candidates = []                                   # (doc_id, position, sentence)
    for doc_id in doc_ids:
        for pos, sent in enumerate(doc_sentences(doc_id)):
            candidates.append((doc_id, pos, sent))

    scores = np.array(embed([c[2] for c in candidates])) @ np.array(embed([query])[0])
    top = sorted(np.argsort(-scores)[:ANSWER_SENTENCES])        # keep reading order
    chosen = [candidates[i] for i in top]
    answer = " ".join(c[2] for c in chosen)
    return answer, sorted({c[0] for c in chosen})


def grounded_answer(query: str, collection, collection_name: str, k: int = 3) -> dict:
    threshold = load_threshold(collection_name)
    hits = search(collection, query, k=k)
    top1 = hits[0]["similarity"]
    if top1 < threshold:
        return {"query": query, "answer": FALLBACK, "grounded": False,
                "top1_similarity": top1, "threshold": threshold, "sources": []}

    doc_ids = []                                      # parent docs of chunks above threshold
    for h in hits:
        if h["similarity"] >= threshold and h["doc_id"] not in doc_ids:
            doc_ids.append(h["doc_id"])
    answer, sources = mock_llm_generate(query, doc_ids)
    return {"query": query, "answer": answer, "grounded": True,
            "top1_similarity": top1, "threshold": threshold, "sources": sources}


def show(result: dict):
    tag = "ANSWERED" if result["grounded"] else "FALLBACK"
    print("  [{}] top1={:.3f} (threshold {})".format(
        tag, result["top1_similarity"], result["threshold"]))
    print("  Answer : {}".format(result["answer"]))
    if result["sources"]:
        print("  Sources: {}".format(", ".join(result["sources"])))


if __name__ == "__main__":
    fixed, sent = get_collections()
    queries = [d["query"] for d in DEMO_QUERIES] + [OUT_OF_SCOPE_DEMO]
    for name, col in [(COLLECTION_FIXED, fixed), (COLLECTION_SENTENCE, sent)]:
        print("\n################ {} ################".format(name))
        for q in queries:
            print("\nQ: {}".format(q))
            show(grounded_answer(q, col, name))