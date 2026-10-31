"""Seeded, deterministic generator for the Naukri.com job-application dataset.

Design choices (also stated in README.md):
  SEED = 42, N_RECORDS = 48
  Category weights : Software Engineer .30, Data Analyst .22, Product Manager .16,
                     HR Executive .16, Sales Associate .16
  Status weights   : Applied .35, Screening .25, Interview Scheduled .18,
                     Offered .10, Rejected .12
  Salary (INR/yr)  : per-category uniform ranges, rounded to Rs 10,000. Reasoning:
                     they approximate fresher-to-mid-level annual CTC expectations in
                     the Indian market (Rs 2.4L for entry sales up to Rs 36L for senior
                     engineers).
  flagged_priority_review : Bernoulli(0.18) per record; if the first draw leaves the
                     rate outside 10-30%, change SEED/weights and regenerate (never
                     hand-edit records).
"""
import random
from collections import Counter

SEED = 42
N_RECORDS = 48

CATEGORIES = ["Software Engineer", "Data Analyst", "Product Manager", "HR Executive", "Sales Associate"]
CATEGORY_WEIGHTS = [0.30, 0.22, 0.16, 0.16, 0.16]

STATUSES = ["Applied", "Screening", "Interview Scheduled", "Offered", "Rejected"]
STATUS_WEIGHTS = [0.35, 0.25, 0.18, 0.10, 0.12]

SALARY_RANGE_INR = {
    "Software Engineer": (600_000, 3_600_000),
    "Data Analyst": (400_000, 2_000_000),
    "Product Manager": (900_000, 3_200_000),
    "HR Executive": (300_000, 1_200_000),
    "Sales Associate": (240_000, 900_000),
}
FLAG_PROBABILITY = 0.18


def generate_job_applications(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        lo, hi = SALARY_RANGE_INR[category]
        salary = round(rng.randint(lo, hi), -4)
        records.append({
            "record_id": f"APP-{i:04d}",
            "category": category,
            "status": status,
            "expected_salary_inr": salary,
            "days_since_created": rng.randint(0, 30),
            "flagged_priority_review": rng.random() < FLAG_PROBABILITY,
            # fabricated, fixed-format phone used to demo the PII guardrail (Task 10)
            "contact_phone": f"+91-{rng.randint(6, 9)}{rng.randint(0, 999_999_999):09d}",
        })
    return records


JOB_APPLICATIONS = generate_job_applications()


def summarize(records=JOB_APPLICATIONS) -> dict:
    cat = Counter(r["category"] for r in records)
    st = Counter(r["status"] for r in records)
    flagged_pct = 100 * sum(r["flagged_priority_review"] for r in records) / len(records)
    return {"n": len(records), "per_category": dict(cat), "per_status": dict(st), "flagged_pct": round(flagged_pct, 2)}


def validate(records=JOB_APPLICATIONS) -> None:
    s = summarize(records)
    assert s["n"] >= 40
    assert all(s["per_category"].get(c, 0) >= 3 for c in CATEGORIES), "category minimum (>=3) violated"
    assert all(s["per_status"].get(x, 0) >= 1 for x in STATUSES), "status coverage violated"
    assert 10 <= s["flagged_pct"] <= 30, f"flagged % {s['flagged_pct']} outside 10-30 band"
    assert all(0 <= r["days_since_created"] <= 30 for r in records)


if __name__ == "__main__":
    validate()
    s = summarize()
    print(f"Records: {s['n']}  (seed={SEED})")
    print("Count per category:")
    for c in CATEGORIES:
        print(f"  {c:<18}{s['per_category'][c]}")
    print("Count per status:")
    for x in STATUSES:
        print(f"  {x:<20}{s['per_status'][x]}")
    print(f"flagged_priority_review=True: {s['flagged_pct']}%")
    print("Sample:", JOB_APPLICATIONS[0])