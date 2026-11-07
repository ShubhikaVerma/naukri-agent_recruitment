"""Task 8 (+ Tasks 10/11/14/15 integration): session memory wrapped around the full pipeline.

LangChain's RunnableWithMessageHistory + InMemoryChatMessageHistory keep one message
history per session_id. The history is used in two real ways:
  1. Follow-up questions that refer back to an earlier application ("is IT escalated?")
     are resolved to the last record id the user mentioned in THIS session.
  2. "What was my first question?" is answered from the stored history.
Input guardrails run BEFORE the history is written, so the history (and everything
downstream) only ever contains masked text. Memory is in-process only (lost on restart).
Note: RunnableWithMessageHistory raises a LangChainDeprecationWarning pointing to
LangGraph persistence. That is expected; the class still works and the warning is not silenced.
"""
import re

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from guardrails import (INJECTION_REFUSAL, detect_injection, injection_event,
                        mask_phone, pii_event, refusal)
from pipeline import process_question
from schemas import SupportResponse

RECORD_ID_RE = re.compile(r"\bAPP-\d{4}\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"\b(it|its|that|this|same|that one|the application)\b", re.IGNORECASE)
APPLICATION_TOPIC_RE = re.compile(r"\b(status|salary|escalat\w*|score|stage|expected|application)\b", re.IGNORECASE)
RECALL_RE = re.compile(
    r"(what did i (ask|say)|(my|the) (first|previous|last|earlier) (question|message))", re.IGNORECASE)

_STORE = {}                                  # session_id -> InMemoryChatMessageHistory
NO_USAGE = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0, "cost_usd": 0.0}
LAST_RUN = {"tool_calls": [], "resolved_question": "", "events": [], "response": None,
            "review": {"status": "skipped"}, "usage": NO_USAGE}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def _last_record_id(history: list):
    """Most recent application id the USER mentioned in this session. Only human messages are
    scanned, so an id quoted inside one of the agent's own replies can never be picked up."""
    for msg in reversed(history):
        if msg.type != "human":
            continue
        found = RECORD_ID_RE.findall(str(msg.content))
        if found:
            return found[-1].upper()
    return None


def _conversation_reply(text: str) -> dict:
    return SupportResponse(answer=text, answer_type="conversation", grounded=False,
                           sources=[], application=None, tools_used=[]).model_dump()


def _turn(inputs: dict) -> str:
    question = inputs["input"]
    history = inputs["history"]                       # this session's messages only
    LAST_RUN.update(tool_calls=[], resolved_question=question, events=[], response=None,
                    review={"status": "skipped"}, usage=NO_USAGE)

    # 2. Recall question: answered purely from the stored history.
    if RECALL_RE.search(question):
        humans = [m.content for m in history if m.type == "human"]
        text = ("I have no earlier messages in this conversation." if not humans else
                "Your first question in this conversation was: \"{}\"".format(humans[0]))
        LAST_RUN["response"] = _conversation_reply(text)
        return text

    # 1. Follow-up that refers back to an application: resolve it from the history.
    if (not RECORD_ID_RE.search(question)
            and REFERENCE_RE.search(question) and APPLICATION_TOPIC_RE.search(question)):
        rid = _last_record_id(history)
        if rid is None:
            text = ("I don't know which application you mean. "
                    "Please give a record id such as APP-0001.")
            LAST_RUN["response"] = _conversation_reply(text)
            return text
        question = "{} (application {})".format(question, rid)
        LAST_RUN["resolved_question"] = question

    out = process_question(question)                 # budget + guardrails + crew + review stage
    LAST_RUN.update(tool_calls=out["tool_calls"], events=out["events"], response=out["response"],
                    review=out["review"], usage=out["usage"])
    return out["answer"]


_chain = RunnableWithMessageHistory(
    RunnableLambda(_turn),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat(session_id: str, question: str) -> dict:
    """One conversational turn. History is loaded and saved automatically per session_id."""
    history_len = lambda: len(get_session_history(session_id).messages)

    hits = detect_injection(question)
    if hits:                                          # blocked before it can enter the history
        return {"session_id": session_id, "answer": INJECTION_REFUSAL,
                "response": refusal(INJECTION_REFUSAL).model_dump(), "events": [injection_event(hits)],
                "masked_question": question, "resolved_question": question,
                "tool_calls": [], "history_length": history_len(),
                "review": {"status": "skipped"}, "usage": NO_USAGE}

    masked, n = mask_phone(question)                  # the history only ever stores masked text
    answer = _chain.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
    return {"session_id": session_id, "answer": answer, "response": LAST_RUN["response"],
            "events": ([pii_event(n)] if n else []) + LAST_RUN["events"],
            "masked_question": masked, "resolved_question": LAST_RUN["resolved_question"],
            "tool_calls": list(LAST_RUN["tool_calls"]), "history_length": history_len(),
            "review": LAST_RUN["review"], "usage": LAST_RUN["usage"]}