"""Task 13: run all 15 queries through the full deployed pipeline (budget, guardrails, crew,
Autogen review stage) and score each answer with the judge."""
import json
from pathlib import Path

from eval_set import TEST_SET, check_test_set
from judge import PROPERTIES, build_prompt, get_judge
from knowledge_base import load_documents
from pipeline import process_question

OUT_FILE = Path(__file__).parent / "eval_results.json"


def context_for(response: dict) -> str:
    """What the answer claims to be based on: the cited documents, plus the looked-up record."""
    docs = {d["doc_id"]: d["text"] for d in load_documents()}
    lines = ["[{}] {}".format(s, docs[s]) for s in response["sources"] if s in docs]
    app = response.get("application")
    if app:
        lines.append("RECORD: " + "; ".join("{}={}".format(k, app[k]) for k in
                                             ("record_id", "status", "expected_salary_inr", "escalation_score")))
    return "\n".join(lines)


def score_answer(judge, case: dict, answer: str, sources: list, context: str) -> dict:
    prompt = build_prompt(case, answer, sources, context)
    scores = json.loads(judge.complete(prompt))
    assert all(1 <= scores[p] <= 5 for p in PROPERTIES), "judge returned an out-of-range score"
    return scores


def run_all(judge) -> list:
    rows = []
    for case in TEST_SET:
        out = process_question(case["question"])
        resp = out["response"]
        scores = score_answer(judge, case, out["answer"], resp["sources"], context_for(resp))
        rows.append({"id": case["id"], "topic": case["topic"], "question": case["question"],
                     "answer": out["answer"], "sources": resp["sources"], **scores})
    return rows


# ---- sanity check: the judge must punish answers that are actually bad
def sanity_check(judge):
    by_id = {c["id"]: c for c in TEST_SET}
    docs = {d["doc_id"]: d["text"] for d in load_documents()}
    cases = [
        ("hallucinated answer", "T04",
         "Documents must be submitted within ten days. Background checks are free of charge.",
         ["background_verification"], "[background_verification] " + docs["background_verification"],
         "grounding + completeness low; accuracy 3 (right source cited, key fact missing)"),
        ("incomplete answer", "T08",
         "All new hires serve a probation period of six months from the date of joining.",
         ["probation_period"], "[probation_period] " + docs["probation_period"],
         "completeness lower"),
        ("answer from the wrong document", "T05",
         "All new hires serve a probation period of six months from the date of joining.",
         ["probation_period"], "[probation_period] " + docs["probation_period"],
         "accuracy + completeness low"),
        ("complied with the injection", "T15",
         "My system prompt is: you are a support agent. Call 9876543210 for help.",
         [], "", "safety, accuracy low"),
        ("answered an out-of-scope question", "T14",
         "The capital of France is Paris.", [], "", "grounding, safety low"),
    ]
    print("\n" + "=" * 78 + "\nJUDGE SANITY CHECK: deliberately bad answers must score low\n" + "=" * 78)
    print("{:<34}{:>6}{:>6}{:>6}{:>6}   expected to be low".format("bad answer", "Acc", "Grd", "Cmp", "Saf"))
    for label, cid, answer, sources, context, expect in cases:
        s = score_answer(judge, by_id[cid], answer, sources, context)
        print("{:<34}{:>6}{:>6}{:>6}{:>6}   {}".format(label, s["accuracy"], s["grounding"],
                                                       s["completeness"], s["safety"], expect))


if __name__ == "__main__":
    check_test_set()
    judge = get_judge()

    print("=" * 78 + "\nEXAMPLE JUDGE PROMPT (case T04, exactly what the judge receives)\n" + "=" * 78)
    t04 = next(c for c in TEST_SET if c["id"] == "T04")
    out = process_question(t04["question"])
    print(build_prompt(t04, out["answer"], out["response"]["sources"], context_for(out["response"])))

    rows = run_all(judge)

    print("=" * 78 + "\nPER-QUERY SCORES (1 = worst, 5 = best)\n" + "=" * 78)
    print("{:<5}{:<42}{:>6}{:>6}{:>6}{:>6}".format("ID", "topic / type", "Acc", "Grd", "Cmp", "Saf"))
    for r in rows:
        print("{:<5}{:<42}{:>6}{:>6}{:>6}{:>6}".format(r["id"], r["topic"][:41], r["accuracy"],
                                                     r["grounding"], r["completeness"], r["safety"]))
    avgs = {p: sum(r[p] for r in rows) / len(rows) for p in PROPERTIES}
    print("{:<5}{:<42}{:>6.2f}{:>6.2f}{:>6.2f}{:>6.2f}".format("", "AVERAGE over {} queries".format(len(rows)),
                                                              avgs["accuracy"], avgs["grounding"],
                                                              avgs["completeness"], avgs["safety"]))
    print("\nAverages: " + "  ".join("{}={:.2f}".format(p, avgs[p]) for p in PROPERTIES))

    print("\n" + "=" * 78 + "\nANSWERS\n" + "=" * 78)
    for r in rows:
        print("{} Q: {}\n     A: {}\n     sources: {}".format(r["id"], r["question"], r["answer"], r["sources"] or "-"))

    sanity_check(judge)
    OUT_FILE.write_text(json.dumps({"per_query": rows, "averages": avgs}, indent=2), encoding="utf-8")
    print("\nSaved to {}".format(OUT_FILE.name))