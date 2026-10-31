"""Task 5: document-level precision/recall for BOTH chunking strategies.

Method: retrieve top-k chunks -> map each chunk to its parent doc_id -> dedup
-> score against the ground-truth relevant documents in queries.py.

    precision = |retrieved_docs ∩ relevant_docs| / |retrieved_docs|   (after dedup)
    recall    = |retrieved_docs ∩ relevant_docs| / |relevant_docs|
"""
import json
import sys
from pathlib import Path

from rag_index import get_collections, search, COLLECTION_FIXED, COLLECTION_SENTENCE
from queries import DEMO_QUERIES

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3      # chunks retrieved per query
OUT_FILE = Path(__file__).parent / "eval_chunking.json"


def retrieved_docs(collection, query: str, k: int) -> list:
    """Top-k chunks mapped to parent documents, deduplicated, rank order kept."""
    docs = []
    for hit in search(collection, query, k=k):
        if hit["doc_id"] not in docs:
            docs.append(hit["doc_id"])
    return docs


def score(retrieved: list, relevant: list) -> dict:
    hits = [d for d in retrieved if d in relevant]
    p = len(hits) / len(retrieved) if retrieved else 0.0
    r = len(hits) / len(relevant) if relevant else 0.0
    return {
        "retrieved": retrieved,
        "relevant": relevant,
        "hits": hits,
        "precision": p,
        "recall": r,
        "p_text": "{}/{} = {:.3f}".format(len(hits), len(retrieved), p),
        "r_text": "{}/{} = {:.3f}".format(len(hits), len(relevant), r),
    }


def mean_line(label: str, values: list, mean: float) -> str:
    terms = " + ".join("{:.3f}".format(v) for v in values)
    return "  MEAN {} = ({}) / {} = {:.3f}".format(label, terms, len(values), mean)


def main():
    fixed, sent = get_collections()
    print("Top-k = {} chunks per query; scoring at document level after dedup\n".format(K))
    results = {}
    for name, col in [(COLLECTION_FIXED, fixed), (COLLECTION_SENTENCE, sent)]:
        rows = []
        print("=" * 78)
        print(name)
        print("=" * 78)
        for i, item in enumerate(DEMO_QUERIES, 1):
            s = score(retrieved_docs(col, item["query"], K), item["relevant_docs"])
            s["query"] = item["query"]
            rows.append(s)
            print("Q{}: {}".format(i, item["query"]))
            print("    relevant : {}".format(s["relevant"]))
            print("    retrieved: {}".format(s["retrieved"]))
            print("    precision: {}   recall: {}".format(s["p_text"], s["r_text"]))

        precisions = [r["precision"] for r in rows]
        recalls = [r["recall"] for r in rows]
        avg_p = sum(precisions) / len(rows)
        avg_r = sum(recalls) / len(rows)
        print()
        print(mean_line("precision", precisions, avg_p))
        print(mean_line("recall   ", recalls, avg_r))
        print()
        results[name] = {"rows": rows, "mean_precision": avg_p, "mean_recall": avg_r}

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved to {}".format(OUT_FILE.name))
    print("\nSUMMARY (k = {})".format(K))
    for name, r in results.items():
        print("  {:<12} precision={:.3f}  recall={:.3f}".format(
            name, r["mean_precision"], r["mean_recall"]))


if __name__ == "__main__":
    main()