"""Task 4b: grounded generation under MOCK_LLM.

Retrieval finds the relevant chunks; generation then works ONLY from the parent documents
of those chunks (parent-document expansion), so a fixed-size chunk that starts or ends
mid-sentence can never produce a broken or truncated answer. Below the calibrated
similarity threshold the agent refuses with an "I don't know" fallback."""
import copy
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from knowledge_base import load_documents
from rag_index import (
    get_collections, search, embed, sentence_chunks,
    COLLECTION_FIXED, COLLECTION_SENTENCE,
)
from queries import DEMO_QUERIES, OUT_OF_SCOPE_DEMO
from runtime_guard import record_llm_call

MOCK_LLM = os.environ.get("MOCK_LLM", "true").lower() == "true"
FALLBACK = "I don't know based on the available policy documents."
LLM_CALLS = {"count": 0}          # generation calls made (a cache hit must NOT increase this)
SEARCH_CALLS = {"count": 0}       # vector searches made   (a cache hit must NOT increase this)
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
ANSWER_SENTENCES = 2              # the mock LLM quotes at least this many sentences...
MAX_ANSWER_SENTENCES = 3          # ...and at most this many (one per clause of a compound question)
CLAUSE_SPLIT_RE = re.compile(r"\s+and\s+|;\s*|\?\s+", re.IGNORECASE)

_doc_sentences = {}


def doc_sentences(doc_id: str) -> list:
    """All complete sentences of one knowledge-base document."""
    if not _doc_sentences:
        for d in load_documents():
            _doc_sentences[d["doc_id"]] = sentence_chunks(d["text"])
    return _doc_sentences[doc_id]


# ------------------------------------------------------------------ Task 16: response cache
# In-memory, keyed by the NORMALISED query text, around the grounded-generation step only
# (retrieval + generation for knowledge-base questions). Application records are never cached,
# and the keys are the already-masked question text, so no phone number is stored.
CACHE_MAX_ENTRIES = 256
CACHE_STATS = {"hits": 0, "misses": 0}
_answer_cache = OrderedDict()                     # (collection, normalised query, k) -> result
_cache_lock = threading.Lock()


def normalize_query(query: str) -> str:
    """Lower-case, drop punctuation (keeping hyphens, e.g. APP-0002), collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", query.lower())).strip()


def reset_cache() -> None:
    """Forget cached document sentences AND cached answers. Called whenever the knowledge base
    changes (POST /add-document), so no stale answer can outlive the documents it came from."""
    _doc_sentences.clear()
    with _cache_lock:
        _answer_cache.clear()


def load_threshold(collection_name: str) -> float:
    data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    if collection_name not in data:
        raise RuntimeError(
            "No calibrated threshold for {}; rerun calibrate.py".format(collection_name))
    return data[collection_name]["threshold"]


def split_clauses(query: str) -> list:
    """Split a compound question ("how long is X and when is Y reviewed?") into its parts."""
    parts = [p.strip(" ?.,") for p in CLAUSE_SPLIT_RE.split(query)]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts if len(parts) > 1 else [query]


def mock_llm_generate(query: str, doc_ids: list, floor: float = 0.0) -> tuple:
    """Deterministic stand-in for an LLM. Extractive, so it can only repeat what the
    retrieved documents say. For a compound question it quotes the best sentence for EACH
    clause (if that sentence is similar enough, >= floor); it then tops up to
    ANSWER_SENTENCES using the whole question. Returns (answer_text, source_doc_ids_used)."""
    if not MOCK_LLM:
        raise NotImplementedError("Wire a real LLM here behind MOCK_LLM=false (optional).")
    LLM_CALLS["count"] += 1

    candidates = []                                   # (doc_id, position, sentence), in reading order
    for doc_id in doc_ids:
        for pos, sent in enumerate(doc_sentences(doc_id)):
            candidates.append((doc_id, pos, sent))
    vectors = np.array(embed([c[2] for c in candidates]))

    chosen = []
    clauses = split_clauses(query)
    if len(clauses) > 1:
        for clause in clauses:
            scores = vectors @ np.array(embed([clause])[0])
            best = int(np.argmax(scores))
            if scores[best] >= floor and best not in chosen:
                chosen.append(best)

    whole = vectors @ np.array(embed([query])[0])
    for i in np.argsort(-whole):
        if len(chosen) >= ANSWER_SENTENCES:
            break
        if int(i) not in chosen:
            chosen.append(int(i))

    picked = [candidates[i] for i in sorted(chosen[:MAX_ANSWER_SENTENCES])]
    answer = " ".join(c[2] for c in picked)
    record_llm_call(query + " ".join(c[2] for c in candidates), answer)   # counts against the request budget
    return answer, sorted({c[0] for c in picked})


def grounded_answer(query: str, collection, collection_name: str, k: int = 3) -> dict:
    threshold = load_threshold(collection_name)
    SEARCH_CALLS["count"] += 1
    hits = search(collection, query, k=k)
    top1 = hits[0]["similarity"]
    if top1 < threshold:
        return {"query": query, "answer": FALLBACK, "grounded": False,
                "top1_similarity": top1, "threshold": threshold, "sources": []}

    doc_ids = []                                      # parent docs of chunks above threshold
    for h in hits:
        if h["similarity"] >= threshold and h["doc_id"] not in doc_ids:
            doc_ids.append(h["doc_id"])
    answer, sources = mock_llm_generate(query, doc_ids, floor=threshold)
    return {"query": query, "answer": answer, "grounded": True,
            "top1_similarity": top1, "threshold": threshold, "sources": sources}


def cached_grounded_answer(query: str, collection, collection_name: str, k: int = 3) -> dict:
    """grounded_answer() behind the response cache. A hit skips the vector search AND the
    generation call; the result carries "cache": "hit" or "miss"."""
    key = (collection_name, normalize_query(query), k)
    with _cache_lock:
        if key in _answer_cache:
            _answer_cache.move_to_end(key)
            CACHE_STATS["hits"] += 1
            return {**copy.deepcopy(_answer_cache[key]), "cache": "hit"}
    result = grounded_answer(query, collection, collection_name, k)            # miss: do the real work
    with _cache_lock:
        CACHE_STATS["misses"] += 1
        _answer_cache[key] = copy.deepcopy(result)
        while len(_answer_cache) > CACHE_MAX_ENTRIES:
            _answer_cache.popitem(last=False)                                   # evict least recently used
    return {**result, "cache": "miss"}


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