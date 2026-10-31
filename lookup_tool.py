"""Task 6: check_job_application_status tool with a designed escalation score.

escalation_score = W_FLAG * flagged_priority_review + W_AGE * (days_since_created / 30)

  - flagged_priority_review is 1.0 if True, else 0.0
  - days_since_created / 30 is the normalised age in [0, 1]: the longer an application has
    been open without resolution, the more urgent it is
  - W_FLAG = W_AGE = 0.5, so the score stays in [0, 1]

ESCALATION_THRESHOLD = 0.6. Justification from this dataset (seed 42, 48 records): see
`python escalation_analysis.py`. At 0.6, 6 of 48 applications (12.5%) are recommended for
escalation, roughly the top 12.5% of scores. Under these weights a non-flagged application
tops out at 0.5, so only flagged applications can escalate, and a flagged one must be at
least 6 days old (0.5 + 0.5 * 6/30 = 0.6).
"""
from dataset import JOB_APPLICATIONS

W_FLAG = 0.5
W_AGE = 0.5
MAX_DAYS = 30
ESCALATION_THRESHOLD = 0.6

_BY_ID = {r["record_id"]: r for r in JOB_APPLICATIONS}


def compute_escalation_score(flagged: bool, days_since_created: int) -> float:
    age = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(W_FLAG * (1.0 if flagged else 0.0) + W_AGE * age, 3)


def check_job_application_status(record_id: str) -> dict:
    """Look up one application. Returns status, expected_salary_inr and escalation_score."""
    key = str(record_id).strip().upper()
    rec = _BY_ID.get(key)
    if rec is None:
        return {"record_id": key, "found": False,
                "error": "No application found with record_id {}".format(key)}
    score = compute_escalation_score(rec["flagged_priority_review"], rec["days_since_created"])
    return {
        "record_id": key,
        "found": True,
        "status": rec["status"],
        "expected_salary_inr": rec["expected_salary_inr"],
        "escalation_score": score,
        "recommend_escalation": score >= ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    for rid in ["APP-0001", "app-0002", "APP-0010", "APP-9999"]:
        print(check_job_application_status(rid))