"""Task 4a: measure top-1 cosine similarity for in-scope vs out-of-scope queries
and set the 'I don't know' threshold between the two observed clusters."""
import json
from pathlib import Path

from rag_index import get_collections, search, COLLECTION_FIXED, COLLECTION_SENTENCE
from queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE

OUT_FILE = Path(__file__).parent / "calibration.json"


def top1(col, query: str) -> float:
    return search(col, query, k=1)[0]["similarity"]


def main():
    fixed, sent = get_collections()
    result = {}
    for name, col in [(COLLECTION_FIXED, fixed), (COLLECTION_SENTENCE, sent)]:
        ins = {q: top1(col, q) for q in CALIBRATION_IN_SCOPE}
        outs = {q: top1(col, q) for q in CALIBRATION_OUT_OF_SCOPE}
        print(f"\n=== {name} ===")
        print("In-scope top-1 similarity:")
        for q, s in ins.items():
            print(f"  {s:.3f}  {q}")
        print("Out-of-scope top-1 similarity:")
        for q, s in outs.items():
            print(f"  {s:.3f}  {q}")

        lowest_in, highest_out = min(ins.values()), max(outs.values())
        print(f"Lowest in-scope = {lowest_in:.3f} | Highest out-of-scope = {highest_out:.3f}")
        if highest_out >= lowest_in:
            print("!! Clusters overlap, so no clean threshold exists for this collection. "
                  "Paste this output to me instead of continuing.")
            continue
        threshold = round((lowest_in + highest_out) / 2, 3)
        print(f"Chosen threshold (midpoint of the gap): {threshold}")
        result[name] = {"threshold": threshold, "in_scope": ins, "out_of_scope": outs}

    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()