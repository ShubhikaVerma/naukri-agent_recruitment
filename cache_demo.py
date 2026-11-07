"""Task 16 transcript: response caching with before/after evidence (call counters + timing)."""
import time

import grounded
from crew_tools import RAG_COLLECTION, TOOL_CALLS
from grounded import (CACHE_STATS, LLM_CALLS, SEARCH_CALLS, cached_grounded_answer, normalize_query,
                      reset_cache)
from pipeline import process_question
from rag_index import embed, get_collections

BAR = "=" * 78
QUESTION = "How many days do I have to submit documents for background verification?"


def counters() -> dict:
    return {"llm": LLM_CALLS["count"], "search": SEARCH_CALLS["count"],
            "hits": CACHE_STATS["hits"], "misses": CACHE_STATS["misses"]}


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000


if __name__ == "__main__":
    fixed, sent = get_collections()
    col = fixed if RAG_COLLECTION == "kb_fixed" else sent
    embed(["warm up the model so timings are not distorted by loading it"])
    results = []

    print(BAR + "\nPART 1: the grounded-generation step, called repeatedly\n" + BAR)
    reset_cache()
    cached_grounded_answer("warm-up query about nothing in particular", col, RAG_COLLECTION)   # warm the code paths
    reset_cache()
    print("normalised key: {!r}\n".format(normalize_query(QUESTION)))
    print("{:<34}{:>9}{:>11}{:>10}{:>12}".format("call", "ms", "LLM calls", "searches", "cache"))
    rows = []
    for label, q in [("1. first ask (cold)", QUESTION),
                     ("2. identical question", QUESTION),
                     ("3. same, different case/spacing", "  HOW many days do I have to submit documents  for background verification ?? "),
                     ("4. a different question", "What is the standard notice period when I join?")]:
        before = counters()
        res, ms = timed(lambda q=q: cached_grounded_answer(q, col, RAG_COLLECTION))
        after = counters()
        rows.append((label, res, ms, after["llm"] - before["llm"], after["search"] - before["search"]))
        print("{:<34}{:>9.2f}{:>11}{:>10}{:>12}".format(
            label, ms, after["llm"] - before["llm"], after["search"] - before["search"], res["cache"].upper()))
    cold, hit1, hit2, other = rows
    results += [
        cold[1]["cache"] == "miss" and cold[3] == 1 and cold[4] == 1,
        hit1[1]["cache"] == "hit" and hit1[3] == 0 and hit1[4] == 0,
        hit2[1]["cache"] == "hit" and hit2[3] == 0,
        other[1]["cache"] == "miss",
        hit1[1]["answer"] == cold[1]["answer"] and hit1[1]["sources"] == cold[1]["sources"],
    ]
    print("\n  [{}] first ask: 1 LLM call + 1 vector search".format("PASS" if results[0] else "FAIL"))
    print("  [{}] identical repeat: cache HIT, 0 LLM calls, 0 searches ({:.2f} ms vs {:.2f} ms cold)".format(
        "PASS" if results[1] else "FAIL", hit1[2], cold[2]))
    print("  [{}] different case/spacing/punctuation also hits (normalised key)".format("PASS" if results[2] else "FAIL"))
    print("  [{}] a genuinely different question still misses".format("PASS" if results[3] else "FAIL"))
    print("  [{}] the cached answer is identical to the original".format("PASS" if results[4] else "FAIL"))

    print("\n" + BAR + "\nPART 2: end to end through the full pipeline (crew + review + budget meter)\n" + BAR)
    reset_cache()
    print("{:<22}{:>9}{:>9}{:>13}{:>13}{:>10}".format("request", "ms", "tokens", "model calls", "gen. calls", "searches"))
    e2e = []
    for label in ("first (cache cold)", "second (identical)"):
        before = counters()
        out, ms = timed(lambda: process_question(QUESTION))
        after = counters()
        u = out["usage"]
        e2e.append((u["total_tokens"], u["model_calls"], after["llm"] - before["llm"],
                    after["search"] - before["search"], [c["tool"] for c in TOOL_CALLS], out["answer"]))
        print("{:<22}{:>9.1f}{:>9}{:>13}{:>13}{:>10}".format(
            label, ms, u["total_tokens"], u["model_calls"], after["llm"] - before["llm"], after["search"] - before["search"]))
    print("\n  tools invoked both times: {}   (the crew still runs; only the redundant work is skipped)".format(e2e[1][4]))
    ok = (e2e[0][2] == 1 and e2e[1][2] == 0 and e2e[0][3] == 1 and e2e[1][3] == 0
          and e2e[1][0] < e2e[0][0] and e2e[0][5] == e2e[1][5])
    results.append(ok)
    print("  [{}] second request: 0 generation calls, 0 searches, fewer tokens ({} vs {}), same answer".format(
        "PASS" if ok else "FAIL", e2e[1][0], e2e[0][0]))

    print("\n" + BAR + "\nPART 3: invalidation when the knowledge base changes\n" + BAR)
    reset_cache()                                   # exactly what POST /add-document calls
    before = counters()
    res, _ = timed(lambda: cached_grounded_answer(QUESTION, col, RAG_COLLECTION))
    after = counters()
    ok = res["cache"] == "miss" and after["llm"] - before["llm"] == 1
    results.append(ok)
    print("  after reset_cache(): the identical question is a cache {} again ({} generation call)".format(
        res["cache"].upper(), after["llm"] - before["llm"]))
    print("  [{}] no stale answer survives a knowledge-base change".format("PASS" if ok else "FAIL"))

    print("\n" + BAR)
    print("ALL CACHE CHECKS PASSED" if all(results) else "SOME CACHE CHECKS FAILED")
    print("cache entries: {}  hits: {}  misses: {}".format(len(grounded._answer_cache), CACHE_STATS["hits"], CACHE_STATS["misses"]))