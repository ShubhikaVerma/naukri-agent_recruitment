"""Justify ESCALATION_THRESHOLD from this dataset's own distribution."""
from dataset import JOB_APPLICATIONS
from lookup_tool import compute_escalation_score, ESCALATION_THRESHOLD, W_FLAG, W_AGE


def percentile(values, p):
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


days = [r["days_since_created"] for r in JOB_APPLICATIONS]
scores = [compute_escalation_score(r["flagged_priority_review"], r["days_since_created"])
          for r in JOB_APPLICATIONS]
n = len(scores)

print("Formula: score = {} * flagged + {} * (days_since_created / 30)".format(W_FLAG, W_AGE))
print("days_since_created percentiles: p50={:.0f}  p80={:.0f}  p90={:.0f}".format(
    percentile(days, 50), percentile(days, 80), percentile(days, 90)))
print("score percentiles:              p50={:.3f}  p80={:.3f}  p90={:.3f}  max={:.3f}".format(
    percentile(scores, 50), percentile(scores, 80), percentile(scores, 90), max(scores)))
print()
print("Candidate thresholds -> how many of {} applications would escalate".format(n))
for t in (0.5, 0.55, 0.6, 0.65, 0.7):
    k = sum(s >= t for s in scores)
    mark = "  <-- chosen" if t == ESCALATION_THRESHOLD else ""
    print("  >= {:.2f}: {:>2} records ({:.1f}%){}".format(t, k, 100 * k / n, mark))
print()
print("Records recommended for escalation at {}:".format(ESCALATION_THRESHOLD))
for r, s in sorted(zip(JOB_APPLICATIONS, scores), key=lambda x: -x[1]):
    if s >= ESCALATION_THRESHOLD:
        print("  {}  score={:.3f}  flagged={}  days={}  status={}".format(
            r["record_id"], s, r["flagged_priority_review"], r["days_since_created"], r["status"]))