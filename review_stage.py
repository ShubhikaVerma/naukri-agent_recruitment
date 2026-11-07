"""Task 14: Autogen review stage that sits AFTER the CrewAI Composer's draft answer.

Team: a 2-agent RoundRobinGroupChat, bounded with max_turns=2 (one turn each):
  1. Policy_Compliance_Reviewer  checks every claim in the draft against the retrieved context
  2. Final_Editor                returns a structured VerdictModel (output_content_type) that
                                 either approves the draft unchanged or revises it
The team is built with custom_message_types=[StructuredMessage[VerdictModel]]; without it the
run crashes with "ValueError: Message type ... is not registered".

Under MOCK_LLM (graded mode: no keys, no network) each agent talks to MockReviewClient, a
deterministic ChatCompletionClient that applies the review rules to the conversation. It reads
the reviewer's report from the message whose SOURCE is the reviewer (never by searching the
whole prompt for a marker), the same lesson as the CrewAI mock.
"""
import asyncio
import json
import os
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from pydantic import BaseModel, Field

from knowledge_base import load_documents
from runtime_guard import record_llm_call

REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"
REVIEW_REFUSAL = ("I can't verify that answer against the policy documents and application records, "
                  "so I won't provide it.")


class VerdictModel(BaseModel):
    approved: bool = Field(description="True only if the draft is approved unchanged.")
    final_answer: str = Field(min_length=1, description="The answer to send to the user.")
    reason: str = Field(description="Why it was approved, or what was changed and why.")


REVIEWER_PROMPT = (
    "You are a policy-compliance reviewer for a hiring-support assistant. You receive a DRAFT ANSWER "
    "and the RETRIEVED CONTEXT it should be based on. Check every sentence of the draft. A sentence is "
    "acceptable only if the context states it, or (for an application-status sentence) every value in it "
    "matches the RECORD line. Reply with 'VERDICT: CLEAN' if every sentence is acceptable; otherwise "
    "reply with 'VERDICT: ISSUES' followed by one line '- UNSUPPORTED: <sentence>' per bad sentence.")
EDITOR_PROMPT = (
    "You are the final editor. Using the draft, the context and the reviewer's report, return a verdict. "
    "If the reviewer found no issues, approve the draft unchanged (approved=true, final_answer=the draft). "
    "Otherwise remove every unsupported sentence, keep the rest word for word, and set approved=false. "
    "If nothing supportable remains, refuse politely. Always give a short reason.")


# ------------------------------------------------------------------ helpers shared by the mock
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _sections(text: str) -> dict:
    parts = re.split(r"^### ([A-Z ]+)\n", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def build_context(response: dict) -> str:
    """The retrieved context an answer is based on: its cited documents plus the looked-up record."""
    docs = {d["doc_id"]: d["text"] for d in load_documents()}
    lines = ["[{}] {}".format(s, docs[s]) for s in response["sources"] if s in docs]
    app = response.get("application")
    if app:
        lines.append("RECORD: " + "; ".join("{}={}".format(k, app[k]) for k in
                                             ("record_id", "status", "expected_salary_inr", "escalation_score")))
    return "\n".join(lines)


def build_task(draft: str, context: str) -> str:
    return ("Review the draft answer against the retrieved context.\n"
            "### DRAFT ANSWER\n{}\n### RETRIEVED CONTEXT\n{}\n".format(draft, context or "(empty)"))


def unsupported_sentences(draft: str, context: str) -> list:
    ctx = {_norm(s) for s in _sentences(re.sub(r"^\[[a-z_0-9]+\] ", "", context, flags=re.MULTILINE))}
    record = re.search(r"^RECORD: (.+)$", context, flags=re.MULTILINE)
    values = [kv.split("=", 1)[1] for kv in record.group(1).split("; ")] if record else []
    bad = []
    for sentence in _sentences(draft):
        if values and sentence.startswith("Application ") and all(v in sentence for v in values):
            continue
        if _norm(sentence) in ctx:
            continue
        bad.append(sentence)
    return bad


# ------------------------------------------------------------------ the mock model client
class MockReviewClient(ChatCompletionClient):
    """Deterministic ChatCompletionClient. role='reviewer' writes a report; role='editor' writes
    the verdict JSON that the Final_Editor's output_content_type parses."""

    def __init__(self, role: str):
        self.role = role
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # -- what the model "says"
    def _respond(self, messages) -> str:
        task = next(m.content for m in messages if getattr(m, "source", "") == "user")
        sec = _sections(task)
        draft, context = sec["DRAFT ANSWER"], sec["RETRIEVED CONTEXT"]
        bad = unsupported_sentences(draft, "" if context == "(empty)" else context)
        if self.role == "reviewer":
            if not bad:
                return "VERDICT: CLEAN. Every sentence is supported by the retrieved context."
            return "VERDICT: ISSUES\n" + "\n".join("- UNSUPPORTED: " + s for s in bad)

        # editor: read the reviewer's report from the message whose SOURCE is the reviewer
        report = [m.content for m in messages if getattr(m, "source", "") == REVIEWER_NAME][-1]
        flagged = [ln[len("- UNSUPPORTED: "):] for ln in report.splitlines() if ln.startswith("- UNSUPPORTED: ")]
        if "VERDICT: CLEAN" in report and not flagged:
            return json.dumps({"approved": True, "final_answer": draft,
                               "reason": "Reviewer found every sentence supported; draft approved unchanged."})
        kept = [s for s in _sentences(draft) if s not in flagged]
        final = " ".join(kept) if kept else REVIEW_REFUSAL
        return json.dumps({"approved": False, "final_answer": final,
                           "reason": "Removed {} unsupported sentence(s): {}".format(
                               len(flagged), " | ".join(flagged))})

    # -- ChatCompletionClient interface
    async def create(self, messages, *, tools=(), tool_choice="auto", json_output=None,
                     extra_create_args=None, cancellation_token=None) -> CreateResult:
        text = self._respond(messages)
        prompt = sum(len(str(m.content)) for m in messages) // 4
        usage = RequestUsage(prompt_tokens=prompt, completion_tokens=len(text) // 4)
        self._last = usage
        self._total = RequestUsage(prompt_tokens=self._total.prompt_tokens + usage.prompt_tokens,
                                   completion_tokens=self._total.completion_tokens + usage.completion_tokens)
        record_llm_call("".join(str(m.content) for m in messages), text)      # per-request budget meter
        return CreateResult(finish_reason="stop", content=text, usage=usage, cached=False)

    async def create_stream(self, messages, *, tools=(), tool_choice="auto", json_output=None,
                            extra_create_args=None, cancellation_token=None):
        yield await self.create(messages)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._last

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages, *, tools=()) -> int:
        return sum(len(str(m.content)) for m in messages) // 4

    def remaining_tokens(self, messages, *, tools=()) -> int:
        return 1_000_000

    @property
    def capabilities(self):
        return {"vision": False, "function_calling": False, "json_output": True}

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         family="unknown", structured_output=True)


def get_clients():
    if os.environ.get("MOCK_LLM", "true").lower() == "true":
        return MockReviewClient("reviewer"), MockReviewClient("editor")
    raise NotImplementedError("Optional: return two real chat-completion clients here when MOCK_LLM=false.")


# ------------------------------------------------------------------ the team
def build_team():
    reviewer_client, editor_client = get_clients()
    reviewer = AssistantAgent(REVIEWER_NAME, model_client=reviewer_client, system_message=REVIEWER_PROMPT,
                              description="Checks each claim in the draft against the retrieved context.")
    editor = AssistantAgent(EDITOR_NAME, model_client=editor_client, system_message=EDITOR_PROMPT,
                            output_content_type=VerdictModel,
                            description="Approves the draft unchanged or revises it; returns a structured verdict.")
    team = RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[VerdictModel]])
    return team, (reviewer_client, editor_client)


async def review_draft(draft: str, context: str) -> dict:
    team, clients = build_team()
    result = await team.run(task=build_task(draft, context))
    verdict = next((m.content for m in reversed(result.messages) if isinstance(m, StructuredMessage)), None)
    if not isinstance(verdict, VerdictModel):
        raise ValueError("the review team did not return a VerdictModel")
    usage = {"prompt_tokens": sum(c.total_usage().prompt_tokens for c in clients),
             "completion_tokens": sum(c.total_usage().completion_tokens for c in clients)}
    return {"verdict": verdict, "changed": verdict.final_answer != draft,
            "speakers": [m.source for m in result.messages], "stop_reason": result.stop_reason,
            "reviewer_report": next(m.to_text() for m in result.messages if m.source == REVIEWER_NAME),
            "usage": usage}


def review_draft_sync(draft: str, context: str) -> dict:
    """For use from ordinary (non-async) code, including FastAPI worker threads."""
    return asyncio.run(review_draft(draft, context))