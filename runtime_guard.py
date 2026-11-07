"""Task 15 (Runtime layer): a per-request token / cost budget.

Every LLM call made while answering one request (the CrewAI crew AND the Autogen review team)
is added to one UsageMeter. The cap is enforced twice, so a request can never silently exceed it:
  1. PRE-FLIGHT : the request's cost is estimated from its size before any model is called;
                  an oversized request is rejected with zero model calls.
  2. MID-FLIGHT : after every call the meter is checked; the moment the running total passes the
                  budget the request is aborted with BudgetExceeded.
Token counts use the usual ~4 characters per token estimate (the mock models have no tokenizer).
"""
import math

TOKEN_BUDGET = 8000                   # tokens allowed per request, all model calls included
PRICE_PER_1K_TOKENS_USD = 0.002       # flat illustrative price, so the cap can be read as money too
# Pre-flight estimate = FIXED + PER_QUESTION_TOKEN * question_tokens. Measured by
# governance_demo.py on this crew: see the printed calibration table.
FIXED_REQUEST_TOKENS = 3500
PER_QUESTION_TOKEN = 6


class BudgetExceeded(Exception):
    def __init__(self, stage: str, used: int, budget: int, detail: str = ""):
        self.stage, self.used, self.budget = stage, used, budget
        super().__init__("{} budget check failed: {} tokens vs a budget of {}{}".format(
            stage, used, budget, " ({})".format(detail) if detail else ""))


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class UsageMeter:
    def __init__(self, budget: int):
        self.budget = budget
        self.prompt_tokens = self.completion_tokens = self.calls = 0
        self.exceeded = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return round(self.total_tokens / 1000 * PRICE_PER_1K_TOKENS_USD, 6)

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if self.total_tokens > self.budget:
            self.exceeded = BudgetExceeded("mid-flight", self.total_tokens, self.budget,
                                           "stopped after {} model call(s)".format(self.calls))
            raise self.exceeded

    def summary(self) -> dict:
        return {"total_tokens": self.total_tokens, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "model_calls": self.calls,
                "budget": self.budget, "cost_usd": self.cost_usd}


_current = None        # the meter of the request being processed (requests are serialised by a lock)


def preflight_estimate(question: str) -> int:
    return FIXED_REQUEST_TOKENS + PER_QUESTION_TOKEN * estimate_tokens(question)


def begin_request(question: str, budget: int = None, preflight: bool = True) -> UsageMeter:
    """Start metering a request. Rejects an oversized request before any model is called."""
    global _current
    budget = TOKEN_BUDGET if budget is None else budget
    if preflight:
        est = preflight_estimate(question)
        if est > budget:
            raise BudgetExceeded("pre-flight", est, budget,
                                 "estimated from a {}-character question; no model was called".format(len(question)))
    _current = UsageMeter(budget)
    return _current


def record_llm_call(prompt_text: str, completion_text: str) -> None:
    """Called by every mock model after each call. Raises BudgetExceeded when over budget."""
    if _current is not None:
        _current.add(estimate_tokens(prompt_text), estimate_tokens(completion_text))


def end_request() -> dict:
    global _current
    summary = _current.summary() if _current else {"total_tokens": 0, "prompt_tokens": 0,
                                                    "completion_tokens": 0, "model_calls": 0,
                                                    "budget": TOKEN_BUDGET, "cost_usd": 0.0}
    _current = None
    return summary


def current_meter():
    return _current