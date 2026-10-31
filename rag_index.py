"""Task 3: two chunking strategies -> embeddings -> two separate ChromaDB collections."""
import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from knowledge_base import load_documents, verify

MODEL_NAME = "all-MiniLM-L6-v2"          # free, local, ~90 MB, downloaded once
# Stored outside the project so OneDrive doesn't try to sync the SQLite files.
CHROMA_DIR = os.environ.get("NAUKRI_CHROMA_DIR", str(Path.home() / ".naukri_chroma"))

FIXED_CHUNK_WORDS = 40                    # window size
FIXED_OVERLAP_WORDS = 10                  # words shared between neighbouring chunks

COLLECTION_FIXED = "kb_fixed"
COLLECTION_SENTENCE = "kb_sentence"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------- chunking strategies ----------

def fixed_size_chunks(text: str, size: int = FIXED_CHUNK_WORDS, overlap: int = FIXED_OVERLAP_WORDS) -> list[str]:
    """Sliding window of `size` words, moving forward by `size - overlap` words."""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def sentence_chunks(text: str) -> list[str]:
    """One chunk per sentence."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def build_chunks(docs: list[dict], strategy: str) -> list[dict]:
    splitter = {"fixed": fixed_size_chunks, "sentence": sentence_chunks}[strategy]
    chunks = []
    for d in docs:
        for i, piece in enumerate(splitter(d["text"])):
            chunks.append({
                "id": f"{d['doc_id']}::{strategy}::{i}",
                "text": piece,
                "metadata": {"doc_id": d["doc_id"], "title": d["title"],
                             "strategy": strategy, "chunk_index": i},
            })
    return chunks


# ---------- embedding + indexing ----------

def embed(texts: list[str]) -> list[list[float]]:
    # normalised vectors, so cosine similarity = dot product
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))


def index_collection(client, name: str, chunks: list[dict]):
    """Rebuild the collection from scratch, then upsert every chunk."""
    try:
        client.delete_collection(name)
    except Exception:
        pass  # didn't exist yet
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return col


def build_all():
    """Index both strategies. Returns (fixed_collection, sentence_collection)."""
    docs = load_documents()
    verify(docs)
    client = get_client()
    fixed = index_collection(client, COLLECTION_FIXED, build_chunks(docs, "fixed"))
    sent = index_collection(client, COLLECTION_SENTENCE, build_chunks(docs, "sentence"))
    return fixed, sent


def get_collections():
    """Open the already-built collections (used by later tasks)."""
    client = get_client()
    return client.get_collection(COLLECTION_FIXED), client.get_collection(COLLECTION_SENTENCE)


# ---------- retrieval ----------

def search(collection, query: str, k: int = 3) -> list[dict]:
    res = collection.query(query_embeddings=embed([query]), n_results=k)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "doc_id": res["metadatas"][0][i]["doc_id"],
            "chunk_index": res["metadatas"][0][i]["chunk_index"],
            "similarity": round(1 - res["distances"][0][i], 4),   # chroma cosine distance = 1 - cos sim
            "text": res["documents"][0][i],
        })
    return hits
def add_document_to_index(doc: dict) -> dict:
    """Upsert one new document into BOTH existing collections without rebuilding them."""
    fixed, sent = get_collections()
    counts = {}
    for key, col in (("fixed", fixed), ("sentence", sent)):
        chunks = build_chunks([doc], key)
        col.upsert(ids=[c["id"] for c in chunks],
                   embeddings=embed([c["text"] for c in chunks]),
                   documents=[c["text"] for c in chunks],
                   metadatas=[c["metadata"] for c in chunks])
        counts[key] = len(chunks)
    return counts


if __name__ == "__main__":
    fixed, sent = build_all()
    print(f"Chroma dir: {CHROMA_DIR}")
    print(f"{COLLECTION_FIXED}:    {fixed.count()} chunks "
          f"({FIXED_CHUNK_WORDS} words, {FIXED_OVERLAP_WORDS} overlap)")
    print(f"{COLLECTION_SENTENCE}: {sent.count()} chunks (one per sentence)\n")

    query = "How long do I have to confirm an interview slot?"
    print(f"Sample query: {query}")
    for label, col in [("FIXED", fixed), ("SENTENCE", sent)]:
        print(f"\n--- {label} top 3 ---")
        for h in search(col, query, k=3):
            print(f"  sim={h['similarity']:.3f}  [{h['doc_id']} #{h['chunk_index']}]  {h['text'][:90]}...")