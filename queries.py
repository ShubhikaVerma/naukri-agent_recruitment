"""Shared query sets. Calibration queries are kept separate from the demo queries,
so the threshold is chosen on one set and tested on another."""

# Used only to calibrate the "I don't know" threshold (Task 4)
CALIBRATION_IN_SCOPE = [
    "Can a recruiter negotiate my salary above the approved budget?",
    "Who can see the results of my background check?",
    "When can an employee apply to move to a different team internally?",
    "Do exit interviews affect my final settlement?",
]
CALIBRATION_OUT_OF_SCOPE = [
    "What is the weather forecast for Bengaluru this weekend?",
    "How do I bake a chocolate cake?",
    "Which company pays software engineers the most?",   # hiring-adjacent: the hard case
]

# Demo queries for Task 4, reused for precision/recall in Task 5.
# "relevant_docs" is the ground truth: the documents that contain the answer.
DEMO_QUERIES = [
    {"query": "Who is eligible for a referral bonus and when is it paid?",
     "relevant_docs": ["referral_bonus"]},
    {"query": "How many days do I have to submit documents for background verification?",
     "relevant_docs": ["background_verification"]},
    {"query": "Can I work remotely during my probation?",
     "relevant_docs": ["remote_work_eligibility", "probation_period"]},
    {"query": "How long is my data kept if I am not hired?",
     "relevant_docs": ["applicant_data_retention"]},
    {"query": "What is the standard notice period when I join?",
     "relevant_docs": ["notice_period"]},
]
OUT_OF_SCOPE_DEMO = "What is the capital of France?"