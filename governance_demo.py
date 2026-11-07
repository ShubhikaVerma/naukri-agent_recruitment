"""Task 15 transcript: Application-layer governance (least autonomy + risk class) and
Runtime-layer governance (per-request token/cost budget)."""
import json

from fastapi.testclient import TestClient

import runtime_guard
from crew import build_crew, make_agent, run_crew, tool_wiring
from crew_tools import RagTool, StatusTool, TOOL_CALLS
from governance import (ALLOWED_TOOL_ROLES, LEAST_AUTONOMY_EXPLANATION, RISK_JUSTIFICATION, RISK_LEVEL,
                        RISK_SCHEME, LeastAutonomyViolation)
from metered_llm import MeteredMockLLM
from pipeline import process_question
from runtime_guard import BudgetExceeded

BAR = "=" * 78


def attempt(label: str, fn):
    try:
        fn()
        print("  [NOT BLOCKED] {}   <-- governance gap".format(label))
        return False
    except LeastAutonomyViolation as exc:
        print("  [BLOCKED] {}\n            {}: {}".format(label, type(exc).__name__, exc))
        return True


if __name__ == "__main__":
    results = []

    print(BAR + "\nAPPLICATION LAYER 1/2: RISK CLASSIFICATION\n" + BAR)
    for level, examples in RISK_SCHEME:
        print("  {:<7}{}{}".format(level, examples, "   <-- this system" if level == RISK_LEVEL else ""))
    print("\n" + RISK_JUSTIFICATION)

    print("\n" + BAR + "\nAPPLICATION LAYER 2/2: LEAST AUTONOMY\n" + BAR)
    print("Permission table (ALLOWED_TOOL_ROLES):")
    for tool, roles in ALLOWED_TOOL_ROLES.items():
        print("  {:<32} -> {}".format(tool, sorted(roles)))
    print("\nWiring of the real crew:")
    for role, tools in tool_wiring(build_crew("demo")).items():
        print("  {:<20} tools: {}".format(role, tools or "none"))

    print("\nAttempts to break the rule:")
    llm = MeteredMockLLM(model="mock-llm")
    results.append(attempt(
        "A. wire the Lookup Agent's tool to the Response Composer",
        lambda: make_agent("Response Composer", "g", "b", [StatusTool(bound_role="Lookup Agent")], llm)))
    results.append(attempt(
        "B. wire the lookup tool to the Retrieval Agent",
        lambda: make_agent("Retrieval Agent", "g", "b", [StatusTool(bound_role="Retrieval Agent")], llm)))
    results.append(attempt(
        "C. skip make_agent() and call the lookup tool as the Composer (call-time guard)",
        lambda: StatusTool(bound_role="Response Composer").run(record_id="APP-0001")))
    results.append(attempt(
        "D. call the policy-search tool as the Lookup Agent (call-time guard)",
        lambda: RagTool(bound_role="Lookup Agent").run(query="notice period")))

    out = run_crew("Can you check APP-0010 and tell me the notice period policy?")
    callers = {(c["tool"], c["agent"]) for c in TOOL_CALLS}
    print("\nLegitimate run, who called what: {}".format(sorted(callers)))
    ok = callers == {("rag_lookup", "Retrieval Agent"), ("check_job_application_status", "Lookup Agent")}
    print("  [{}] only the Lookup Agent called check_job_application_status".format("PASS" if ok else "FAIL"))
    results.append(ok)
    print("\nHow the guard works:\n" + LEAST_AUTONOMY_EXPLANATION)

    print("\n" + BAR + "\nRUNTIME LAYER: PER-REQUEST BUDGET\n" + BAR)
    print("Budget: {} tokens per request  (= ${:.4f} at ${}/1k tokens)\n".format(
        runtime_guard.TOKEN_BUDGET, runtime_guard.TOKEN_BUDGET / 1000 * runtime_guard.PRICE_PER_1K_TOKENS_USD,
        runtime_guard.PRICE_PER_1K_TOKENS_USD))
    print("Measured usage of normal requests (crew + review team, all model calls):")
    print("  {:>7} {:>6} {:>9}   question".format("tokens", "calls", "cost USD"))
    used = []
    for q in ["How many days do I have to submit documents for background verification?",
              "What is the status of APP-0002?",
              "Can you check APP-0010 and tell me the notice period policy?",
              "What is the capital of France?"]:
        u = process_question(q)["usage"]
        used.append(u["total_tokens"])
        print("  {:>7} {:>6} {:>9.5f}   {}".format(u["total_tokens"], u["model_calls"], u["cost_usd"], q[:52]))
    results.append(max(used) <= runtime_guard.TOKEN_BUDGET)
    print("  [{}] every normal request fits the budget (max {} of {})".format(
        "PASS" if results[-1] else "FAIL", max(used), runtime_guard.TOKEN_BUDGET))

    print("\nCase 1: a deliberately OVERSIZED request (4,000 characters) is rejected up front")
    huge = "How many days do I have to submit documents for background verification? " + "please explain in detail " * 160
    TOOL_CALLS.clear()
    try:
        process_question(huge)
        print("  [NOT REJECTED]  <-- the budget was silently exceeded")
        results.append(False)
    except BudgetExceeded as exc:
        print("  question length : {} characters".format(len(huge)))
        print("  estimate        : {} tokens vs budget {}".format(exc.used, exc.budget))
        print("  rejected        : {}".format(exc))
        print("  tools invoked   : {}   (no model was called)".format(len(TOOL_CALLS)))
        results.append(exc.stage == "pre-flight" and len(TOOL_CALLS) == 0)
        print("  [{}] rejected before any model call".format("PASS" if results[-1] else "FAIL"))

    print("\nCase 2: the mid-flight backstop. Pre-flight is switched off and the budget set to 1,500 tokens,")
    print("so a normal request starts but must be stopped the moment it passes the cap")
    try:
        process_question("How many days do I have to submit documents for background verification?",
                         budget=1500, preflight=False)
        print("  [NOT STOPPED]  <-- the budget was silently exceeded")
        results.append(False)
    except BudgetExceeded as exc:
        print("  stopped         : {}".format(exc))
        results.append(exc.stage == "mid-flight" and exc.used > exc.budget)
        print("  [{}] aborted mid-flight instead of finishing over budget".format("PASS" if results[-1] else "FAIL"))

    print("\nCase 3: over HTTP the same rejection becomes 413 (budget temporarily set to 1,500 tokens)")
    from app import app
    original = runtime_guard.TOKEN_BUDGET
    runtime_guard.TOKEN_BUDGET = 1500
    try:
        with TestClient(app) as client:
            r = client.post("/ask", json={"question": "What is the status of APP-0002?"})
    finally:
        runtime_guard.TOKEN_BUDGET = original
    print("  POST /ask -> HTTP {}  {}".format(r.status_code, json.dumps(r.json())[:150]))
    results.append(r.status_code == 413)
    print("  [{}] HTTP 413 returned".format("PASS" if results[-1] else "FAIL"))

    print("\n" + BAR)
    print("ALL GOVERNANCE CHECKS PASSED" if all(results) else "SOME GOVERNANCE CHECKS FAILED")