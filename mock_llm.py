"""MOCK_LLM for CrewAI: a deterministic, keyless, offline BaseLLM subclass.

It speaks CrewAI's ReAct text protocol:
    call 1 : "Thought ... / Action: <tool> / Action Input: {json}"      (if a tool is needed)
    call 2 : "Thought: I now know the final answer / Final Answer: ..."  (after the observation)

Two silent pitfalls from the brief are handled explicitly:
  1. Tool results are parsed ONLY from the model's own previous (assistant-role) message.
     CrewAI's system prompt contains the template text "Observation: the result of the
     action", so searching the whole conversation would match it before any tool has run.
  2. Tools are chosen by their declared ARGUMENT SCHEMA (a `query` field vs a `record_id`
     field), never by substring-matching the tool's name (a tool named `rag_lookup`
     contains "lookup" but is the retrieval tool).
"""
import json
import re

from crewai.llms.base_llm import BaseLLM

RECORD_ID_RE = re.compile(r"\bAPP-\d{4}\b", re.IGNORECASE)
# Words that signal the user also wants a hiring-policy answer. A question that names an
# application and contains none of them is a pure record lookup: no policy search is needed.
POLICY_CUE_RE = re.compile(
    r"\b(policy|policies|process|procedure|rules?|guidelines?|eligib\w*|allowed|bonus|notice|probation|"
    r"remote|referral|verification|retention|transfer|negotiat\w*|diversity|exit|interview|"
    r"how (long|many|much)|can i|do i)\b", re.IGNORECASE)


def _as_messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def _tool_fields(tool) -> set:
    schema = getattr(tool, "args_schema", None)
    return set(schema.model_fields) if schema is not None else set()


def _find_tool(agent, field: str):
    """Return the agent's tool whose argument schema declares `field`."""
    for t in (getattr(agent, "tools", None) or []):
        if field in _tool_fields(t):
            return t
    return None


def _question(messages) -> str:
    text = next((m["content"] for m in messages if m["role"] == "user"), "")
    found = re.search(r"USER QUESTION:\s*(.+)", text)
    return found.group(1).strip() if found else text.strip()


def _last_observation(messages):
    """Text after the last 'Observation:' inside the model's OWN last assistant message."""
    mine = [m for m in messages if m["role"] == "assistant"]
    if not mine:
        return None
    body = mine[-1]["content"]
    if "Observation:" not in body:
        return None
    return body.rsplit("Observation:", 1)[1].strip()


def _action(tool_name: str, args: dict, thought: str) -> str:
    return "Thought: {}\nAction: {}\nAction Input: {}".format(thought, tool_name, json.dumps(args))


def _final(text: str) -> str:
    return "Thought: I now know the final answer\nFinal Answer: {}".format(text)


def _format_observation(obs: str) -> str:
    """Turn a tool's JSON observation into a marked line (with a JSON payload) that the
    Response Composer can parse."""
    try:
        data = json.loads(obs)
    except (ValueError, TypeError):
        return "STATUS: NOT_FOUND"
    if "grounded" in data:                                   # retrieval result
        if data["grounded"]:
            return "POLICY: " + json.dumps({"answer": data["answer"], "sources": data["sources"]})
        return "POLICY: NO_GROUNDED_CONTEXT"
    if "found" in data:                                      # lookup result
        if not data["found"]:
            return "STATUS: NOT_FOUND"
        keys = ("record_id", "status", "expected_salary_inr", "escalation_score", "recommend_escalation")
        return "STATUS: " + json.dumps({k: data[k] for k in keys})
    return "STATUS: NOT_FOUND"


def _payload(line_match):
    """JSON payload of a POLICY:/STATUS: line, or None for NO_GROUNDED_CONTEXT / NOT_FOUND / NOT_APPLICABLE."""
    if not line_match:
        return None
    text = line_match.group(1).strip()
    return json.loads(text) if text.startswith("{") else None


def _status_sentence(app: dict) -> str:
    return ("Application {rid} is '{st}', expected salary INR {sal}, escalation score {sc} ({esc}).").format(
        rid=app["record_id"], st=app["status"], sal=app["expected_salary_inr"], sc=app["escalation_score"],
        esc="escalation recommended" if app["recommend_escalation"] else "no escalation needed")


def _compose(context: str) -> str:
    """Build the structured response (JSON text matching schemas.SupportResponse)."""
    policy_line = re.search(r"(?m)^POLICY:\s*(.+)$", context)
    status_line = re.search(r"(?m)^STATUS:\s*(.+)$", context)
    policy, app = _payload(policy_line), _payload(status_line)

    tools_used = []
    if policy_line and "NOT_APPLICABLE" not in policy_line.group(1):
        tools_used.append("rag_lookup")
    if status_line and "NOT_APPLICABLE" not in status_line.group(1):
        tools_used.append("check_job_application_status")

    answer_parts = []
    if policy:
        answer_parts.append(policy["answer"])
    if app:
        answer_parts.append(_status_sentence(app))

    if policy and app:
        answer_type = "policy_and_status"
    elif policy:
        answer_type = "policy"
    elif app:
        answer_type = "application_status"
    else:
        answer_type = "fallback"
        if status_line and "NOT_FOUND" in status_line.group(1):
            answer_parts.append("I could not find that application record.")
        else:
            answer_parts.append("I don't know based on the available policy documents or application records.")

    return json.dumps({
        "answer": " ".join(answer_parts),
        "answer_type": answer_type,
        "grounded": answer_type != "fallback",
        "sources": policy["sources"] if policy else [],
        "application": app,
        "tools_used": tools_used,
    })


class MockLLM(BaseLLM):
    llm_type: str = "mock"

    def supports_function_calling(self) -> bool:
        return False            # forces CrewAI's ReAct text protocol

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        msgs = _as_messages(messages)

        # Step 2: a tool has already run -> turn its observation into the final answer.
        obs = _last_observation(msgs)
        if obs is not None:
            return _final(_format_observation(obs))

        # Step 1: decide whether this agent should call one of ITS tools.
        question = _question(msgs)
        rag = _find_tool(from_agent, "query")
        lookup = _find_tool(from_agent, "record_id")
        if rag is not None:
            if RECORD_ID_RE.search(question) and not POLICY_CUE_RE.search(question):
                return _final("POLICY: NOT_APPLICABLE")          # pure application lookup
            return _action(rag.name, {"query": question}, "I should search the policy knowledge base.")
        if lookup is not None:
            rid = RECORD_ID_RE.search(question)
            if rid:
                return _action(lookup.name, {"record_id": rid.group(0).upper()},
                               "The user named an application, so I should look it up.")
            return _final("STATUS: NOT_APPLICABLE")

        # No tools on this agent: it is the Response Composer.
        last_user = [m["content"] for m in msgs if m["role"] == "user"][-1]
        return _final(_compose(last_user))