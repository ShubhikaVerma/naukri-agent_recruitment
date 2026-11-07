"""Task 14 transcript: the Autogen review stage approving one draft unchanged and revising others."""
import json

from guardrails import guarded_ask
from review_stage import (EDITOR_NAME, REVIEWER_NAME, VerdictModel, build_context, review_draft_sync)


def show(title: str, draft: str, context: str, expect_approved: bool):
    result = review_draft_sync(draft, context)
    v = result["verdict"]
    print("=" * 78 + "\n" + title + "\n" + "=" * 78)
    print("Draft answer   : {}".format(draft))
    print("Speakers       : {}   (stop reason: {})".format(result["speakers"], result["stop_reason"]))
    print("Reviewer report: {}".format(result["reviewer_report"].replace("\n", " | ")))
    print("Structured verdict (validated {}):".format(VerdictModel.__name__))
    print(json.dumps(v.model_dump(), indent=2))
    print("Final answer is {} the draft".format("IDENTICAL to" if not result["changed"] else "DIFFERENT from"))
    ok = (v.approved == expect_approved) and (result["changed"] != expect_approved)
    print("[{}] expected approved={}\n".format("PASS" if ok else "FAIL", expect_approved))
    return ok, result


if __name__ == "__main__":
    q = "How many days do I have to submit documents for background verification?"
    out = guarded_ask(q)
    draft, context = out["answer"], build_context(out["response"])

    ok1, r1 = show("CASE 1: genuine crew draft -> must be APPROVED UNCHANGED", draft, context, True)

    injected = draft + " Late submissions are accepted for a fee of Rs 500 per day."
    ok2, r2 = show("CASE 2: draft with a deliberately injected UNGROUNDED claim -> must be REVISED",
                   injected, context, False)

    status = guarded_ask("What is the status of APP-0002?")
    s_draft, s_context = status["answer"], build_context(status["response"])
    ok3, _ = show("CASE 3: genuine status draft -> must be APPROVED UNCHANGED", s_draft, s_context, True)

    wrong = s_draft.replace("920000", "1500000")
    ok4, _ = show("CASE 4: status draft whose salary contradicts the record -> must be REVISED",
                  wrong, s_context, False)

    print("=" * 78)
    print("bounded at 2 turns: each case had exactly the speakers {} and {} -> {}".format(
        REVIEWER_NAME, EDITOR_NAME,
        "PASS" if r1["speakers"] == ["user", REVIEWER_NAME, EDITOR_NAME] == r2["speakers"] else "FAIL"))
    print("revised answer keeps the supported sentences verbatim -> {}".format(
        "PASS" if r2["verdict"].final_answer == draft else "FAIL"))
    print("\nALL CASES PASSED" if all([ok1, ok2, ok3, ok4]) else "\nSOME CASES FAILED")