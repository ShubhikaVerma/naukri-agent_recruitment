"""Task 15 (Application layer): least autonomy + risk classification.

Least autonomy means every tool can be used by exactly the agents that need it, and nothing
else. ALLOWED_TOOL_ROLES is the single source of truth. It is enforced twice:
  1. at WIRING time  (crew.make_agent refuses to attach a tool to the wrong agent), and
  2. at CALL time    (each tool re-checks its own bound role before it runs), so even code that
                      bypasses make_agent cannot use a tool it is not entitled to.
"""
ALLOWED_TOOL_ROLES = {
    "rag_lookup": {"Retrieval Agent"},
    "check_job_application_status": {"Lookup Agent"},
}


class LeastAutonomyViolation(PermissionError):
    """An agent tried to use (or be given) a tool it is not allowed to have."""


def check_wiring(tool_name: str, bound_role: str, agent_role: str) -> None:
    allowed = ALLOWED_TOOL_ROLES.get(tool_name)
    if allowed is None:
        raise LeastAutonomyViolation("tool '{}' is not registered in ALLOWED_TOOL_ROLES".format(tool_name))
    if bound_role != agent_role:
        raise LeastAutonomyViolation("tool '{}' is bound to '{}', so it cannot be given to '{}'".format(
            tool_name, bound_role or "nobody", agent_role))
    if agent_role not in allowed:
        raise LeastAutonomyViolation("'{}' is not permitted to use '{}' (allowed: {})".format(
            agent_role, tool_name, sorted(allowed)))


def check_call(tool_name: str, bound_role: str) -> None:
    allowed = ALLOWED_TOOL_ROLES.get(tool_name, set())
    if bound_role not in allowed:
        raise LeastAutonomyViolation("'{}' is not permitted to call '{}' (allowed: {})".format(
            bound_role or "an unbound caller", tool_name, sorted(allowed)))


LEAST_AUTONOMY_EXPLANATION = (
    "Each tool instance is bound to one agent role, and ALLOWED_TOOL_ROLES says which roles may use "
    "which tool: only the Lookup Agent may use check_job_application_status, and only the Retrieval "
    "Agent may use rag_lookup. The guard has two layers. At wiring time make_agent() refuses to attach "
    "a tool to any agent whose role does not match the tool's binding or is not on its allow-list, so "
    "wiring the lookup tool to the Response Composer raises LeastAutonomyViolation before the crew exists. "
    "At call time the tool checks its own bound role again before running, so even code that builds an "
    "Agent directly and skips make_agent() is stopped when it tries to call the tool. The Composer has no "
    "tools at all: it can only combine what the other agents report.")

RISK_SCHEME = [("Low", "summarization, transcription"),
               ("Medium", "code generation, customer-support tickets"),
               ("High", "medical data, hiring decisions, financial data")]
RISK_LEVEL = "High"
RISK_JUSTIFICATION = (
    "HIGH. The assistant reads individual job-application records, including expected salary (financial "
    "data), and produces an escalation recommendation that can influence how a candidate's application is "
    "handled, so it sits next to hiring decisions even though it is read-only and never decides anything "
    "itself. Answering policy questions alone would be Medium (customer support), but errors or leaks here "
    "affect real people's employment and pay, so the stricter controls apply: least autonomy, phone-number "
    "masking, groundedness refusal, an independent review team, a per-request token budget, and "
    "structured audit logging. This project uses fabricated data only.")