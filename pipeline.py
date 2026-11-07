"""The full request path (Tasks 10, 14, 15): budget -> guardrails + crew -> Autogen review.

    process_question(q)
      1. runtime budget: pre-flight estimate, then metering of every model call
      2. guarded_ask: input guardrails -> CrewAI crew -> output groundedness check
      3. review stage: Autogen team approves the draft unchanged or revises it
    Returns the same dict shape as guarded_ask, plus "review" and "usage".
"""
import os

import runtime_guard
from guardrails import guarded_ask, refusal
from review_stage import REVIEW_REFUSAL, build_context, review_draft_sync
from schemas import SupportResponse

REVIEW_ENABLED = os.environ.get("REVIEW_STAGE", "true").lower() == "true"


def process_question(question: str, budget: int = None, preflight: bool = True) -> dict:
    meter = runtime_guard.begin_request(question, budget=budget, preflight=preflight)
    try:
        try:
            out = guarded_ask(question)
            out["review"] = {"status": "skipped", "reason": "no policy or status claim to review"}

            resp = SupportResponse.model_validate(out["response"])
            if REVIEW_ENABLED and resp.answer_type not in ("fallback", "conversation"):
                result = review_draft_sync(out["answer"], build_context(out["response"]))
                verdict = result["verdict"]
                out["review"] = {"status": "approved" if not result["changed"] else "revised",
                                 "reason": verdict.reason}
                if result["changed"]:
                    if verdict.final_answer == REVIEW_REFUSAL:
                        final = refusal(REVIEW_REFUSAL)
                    else:
                        final = resp.model_copy(update={"answer": verdict.final_answer})
                    out["answer"], out["response"] = final.answer, final.model_dump()
                    out["events"].append({"guardrail": "review_stage", "action": "revised_answer",
                                          "detail": verdict.reason})
        except runtime_guard.BudgetExceeded:
            raise
        except Exception as exc:
            if meter.exceeded is not None:           # a framework wrapped our budget error
                raise meter.exceeded from exc
            raise
        out["usage"] = meter.summary()
        return out
    finally:
        runtime_guard.end_request()