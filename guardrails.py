"""Task 10: input-side and output-side guardrails.

INPUT  : (1) PII masking of the fixed-format phone number, (2) prompt-injection detection.
OUTPUT : groundedness check - every policy sentence in an answer must be a sentence of a
         cited knowledge-base document, and every status figure must match the looked-up
         record; otherwise the answer is refused.

Out of scope for masking (free text / no reliable pattern under a keyless MOCK_LLM-only
masker): candidate name, expected salary, background-check results. Use fabricated data only.

mask_phone is reused by the JSON-Lines logger in Task 12 so a raw phone number never
reaches disk.
"""
import re

from knowledge_base import load_documents
from schemas import SupportResponse

# ---------------------------------------------------------------- input: PII masking
PHONE_RE = re.compile(r"(?<!\w)(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\w)")
PHONE_MASK = "[PHONE_MASKED]"


def mask_phone(text: str) -> tuple:
    """Replace Indian mobile numbers (+91-9876543210, 98765 43210, 09876543210 ...)."""
    return PHONE_RE.subn(PHONE_MASK, text)


# ---------------------------------------------------------------- input: injection
INJECTION_PATTERNS = {
    "ignore-instructions": r"\bignore\b.{0,30}\b(previous|prior|above|earlier|all|any|your)\b.{0,30}\b(instructions?|prompts?|rules?)\b",
    "disregard-instructions": r"\bdisregard\b.{0,30}\b(instructions?|rules?|guidelines?|prompt)\b",
    "reveal-system-prompt": r"\b(reveal|show|print|repeat|display|leak)\b.{0,30}\b(system|hidden|initial|your)\b.{0,15}\b(prompt|instructions?)\b",
    "role-override": r"\byou are now\b|\bpretend\b.{0,20}\b(to be|you are|you're)\b",
    "jailbreak-keyword": r"\bdeveloper mode\b|\bjailbreak\b|\bDAN mode\b",
    "forget-instructions": r"\bforget\b.{0,15}\b(everything|all|your|the)\b.{0,15}\b(above|previous|prior|instructions|rules)\b",
    "override-safety": r"\b(override|bypass)\b.{0,20}\b(safety|rules?|instructions?|guardrails?|filters?|restrictions?)\b",
}
_INJECTION_RES = {name: re.compile(p, re.IGNORECASE | re.DOTALL) for name, p in INJECTION_PATTERNS.items()}


def detect_injection(text: str) -> list:
    """Names of every injection pattern the text matches (empty list = clean)."""
    return [name for name, rx in _INJECTION_RES.items() if rx.search(text)]


# ---------------------------------------------------------------- output: groundedness
def _split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


_kb_sentences = {}


def _doc_sentence_set(doc_id: str) -> set:
    if not _kb_sentences:
        for d in load_documents():
            _kb_sentences[d["doc_id"]] = {_norm(s) for s in _split_sentences(d["text"])}
    return _kb_sentences.get(doc_id, set())


def check_output_grounded(resp: SupportResponse) -> dict:
    """Return {"supported": bool, "unsupported": [sentences...]}."""
    if resp.answer_type in ("fallback", "conversation"):
        return {"supported": True, "unsupported": []}          # refusing / memory answers make no KB claim

    allowed = set()
    for doc_id in resp.sources:
        allowed |= _doc_sentence_set(doc_id)

    unsupported = []
    for sentence in _split_sentences(resp.answer):
        if resp.application and sentence.startswith("Application " + resp.application.record_id):
            a = resp.application
            facts = [a.record_id, a.status, str(a.expected_salary_inr), str(a.escalation_score)]
            if not all(f in sentence for f in facts):
                unsupported.append(sentence)                   # status sentence contradicts the record
        elif _norm(sentence) not in allowed:
            unsupported.append(sentence)                       # claim not found in the cited documents
    return {"supported": not unsupported, "unsupported": unsupported}


def reset_cache() -> None:
    """Forget cached KB sentences (called after a document is added at runtime)."""
    _kb_sentences.clear()


def pii_event(n: int) -> dict:
    return {"guardrail": "pii_mask", "action": "masked_phone",
            "detail": "{} phone number(s) replaced with {}".format(n, PHONE_MASK)}


def injection_event(hits: list) -> dict:
    return {"guardrail": "prompt_injection", "action": "blocked_request",
            "detail": "matched: {}".format(hits)}


INJECTION_REFUSAL = ("I can't follow instructions that try to change how I work. "
                     "I can help with Naukri.com hiring-policy questions and application status.")
UNSUPPORTED_REFUSAL = ("I can't verify that answer against the policy documents, so I won't provide it.")


def refusal(message: str) -> SupportResponse:
    return SupportResponse(answer=message, answer_type="fallback", grounded=False,
                           sources=[], application=None, tools_used=[])


def enforce_groundedness(resp: SupportResponse) -> tuple:
    """Output guardrail. Returns (response_to_send, event_or_None)."""
    verdict = check_output_grounded(resp)
    if verdict["supported"]:
        return resp, None
    return refusal(UNSUPPORTED_REFUSAL), {
        "guardrail": "output_groundedness", "action": "refused_answer",
        "detail": "unsupported sentence(s): {}".format(verdict["unsupported"])}


# ---------------------------------------------------------------- the guarded pipeline
def guarded_ask(question: str) -> dict:
    """Input guardrails -> crew -> output guardrail. Every guardrail that fires is reported."""
    from crew import run_crew                                   # lazy: guardrail helpers stay importable alone
    events = []

    hits = detect_injection(question)
    if hits:
        events.append(injection_event(hits))
        return {"answer": INJECTION_REFUSAL, "response": refusal(INJECTION_REFUSAL).model_dump(),
                "events": events, "masked_question": question, "tool_calls": []}

    masked, n = mask_phone(question)
    if n:
        events.append(pii_event(n))

    out = run_crew(masked)                                      # the crew only ever sees masked text
    draft = SupportResponse.model_validate(out["response"])
    if draft.answer_type == "fallback":                         # retrieval found no supporting context
        events.append({"guardrail": "output_groundedness", "action": "refused_no_supporting_context",
                       "detail": "retrieved context below the calibrated similarity threshold; fallback sent"})
    resp, event = enforce_groundedness(draft)
    if event:
        events.append(event)
    return {"answer": resp.answer, "response": resp.model_dump(), "events": events,
            "masked_question": masked, "tool_calls": out["tool_calls"]}