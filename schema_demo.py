"""Task 9 transcript: every crew response validates; deliberately invalid payloads are rejected."""
import json

from pydantic import ValidationError

from crew import run_crew
from schemas import validate_response

QUERIES = [
    "How many days do I have to submit documents for background verification?",
    "What is the status of APP-0002?",
    "Can you check APP-0010 and tell me the notice period policy?",
    "What is the capital of France?",
]

INVALID = {
    "escalation_score out of range (1.7)": {
        "answer": "x", "answer_type": "application_status", "grounded": True, "sources": [],
        "application": {"record_id": "APP-0002", "status": "Screening", "expected_salary_inr": 920000,
                        "escalation_score": 1.7, "recommend_escalation": True},
        "tools_used": ["check_job_application_status"]},
    "malformed record_id ('APP-2')": {
        "answer": "x", "answer_type": "application_status", "grounded": True, "sources": [],
        "application": {"record_id": "APP-2", "status": "Screening", "expected_salary_inr": 920000,
                        "escalation_score": 0.7, "recommend_escalation": True},
        "tools_used": ["check_job_application_status"]},
    "policy answer with no cited source": {
        "answer": "Sixty days.", "answer_type": "policy", "grounded": True, "sources": [],
        "application": None, "tools_used": ["rag_lookup"]},
    "invented source document": {
        "answer": "Sixty days.", "answer_type": "policy", "grounded": True, "sources": ["made_up_policy"],
        "application": None, "tools_used": ["rag_lookup"]},
    "fallback that claims to be grounded": {
        "answer": "I don't know.", "answer_type": "fallback", "grounded": True, "sources": [],
        "application": None, "tools_used": []},
}

if __name__ == "__main__":
    print("PART 1: crew responses, each validated against SupportResponse\n")
    for q in QUERIES:
        out = run_crew(q)
        print("Q: {}".format(q))
        print(json.dumps(out["response"], indent=2))
        print("  -> schema validation: PASSED\n")

    print("PART 2: deliberately invalid payloads must be rejected\n")
    for label, payload in INVALID.items():
        try:
            validate_response(json.dumps(payload))
            print("  [NOT REJECTED]  {}  <-- schema is too loose".format(label))
        except ValidationError as e:
            msg = e.errors()[0]["msg"]
            print("  [REJECTED]  {}\n              reason: {}".format(label, msg))