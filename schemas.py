"""Task 9: the structured-output schema every crew response must conform to."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dataset import STATUSES
from knowledge_base import kb_doc_ids


class ApplicationInfo(BaseModel):
    record_id: str = Field(pattern=r"^APP-\d{4}$")
    status: str
    expected_salary_inr: int = Field(gt=0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    recommend_escalation: bool

    @field_validator("status")
    @classmethod
    def status_is_known(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError("unknown status '{}'".format(v))
        return v


class SupportResponse(BaseModel):
    answer: str = Field(min_length=1)
    # "conversation" = answered from session memory or a clarification request (no KB lookup)
    answer_type: Literal["policy", "application_status", "policy_and_status", "fallback", "conversation"]
    grounded: bool
    sources: list[str] = Field(default_factory=list)
    application: Optional[ApplicationInfo] = None
    tools_used: list[str] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def sources_are_real_documents(cls, v: list) -> list:
        known = kb_doc_ids()                       # read live, so documents added at runtime count
        unknown = [s for s in v if s not in known]
        if unknown:
            raise ValueError("sources not in the knowledge base: {}".format(unknown))
        return v

    @model_validator(mode="after")
    def parts_match_answer_type(self):
        t = self.answer_type
        if t in ("fallback", "conversation"):
            if self.grounded or self.sources or self.application:
                raise ValueError("a {} answer must be ungrounded, with no sources or application".format(t))
            return self
        if not self.grounded:
            raise ValueError("a non-fallback answer must be grounded")
        wants_policy = t in ("policy", "policy_and_status")
        wants_status = t in ("application_status", "policy_and_status")
        if wants_policy and not self.sources:
            raise ValueError("a policy answer must cite at least one source document")
        if not wants_policy and self.sources:
            raise ValueError("an application_status answer must not cite policy sources")
        if wants_status and self.application is None:
            raise ValueError("a status answer must include the application details")
        if not wants_status and self.application is not None:
            raise ValueError("a policy answer must not include application details")
        return self


def validate_response(raw: str) -> SupportResponse:
    """Validate a crew's raw output against the schema. Raises pydantic.ValidationError."""
    return SupportResponse.model_validate_json(raw.strip())


__all__ = ["SupportResponse", "ApplicationInfo", "validate_response", "ValidationError"]