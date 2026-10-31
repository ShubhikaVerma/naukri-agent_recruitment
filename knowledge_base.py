"""Task 2: load and verify the knowledge base in kb/*.md (+ add documents at runtime, Task 11)."""
import re
from pathlib import Path

KB_DIR = Path(__file__).parent / "kb"

# One file per topic required by the brief.
REQUIRED_TOPICS = {
    "job_application_eligibility": "job-application eligibility criteria",
    "interview_scheduling": "interview-scheduling process",
    "offer_negotiation": "offer-negotiation policy",
    "background_verification": "background-verification process",
    "notice_period": "notice-period policy",
    "referral_bonus": "referral-bonus policy",
    "internal_transfer": "internal-transfer eligibility",
    "probation_period": "probation-period policy",
    "remote_work_eligibility": "remote-work eligibility",
    "diversity_hiring": "diversity-hiring guidelines",
    "exit_interview": "exit-interview process",
    "applicant_data_retention": "applicant-data-retention policy",
}

DOC_ID_RE = re.compile(r"[a-z][a-z0-9_]{2,59}")


def count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text.strip()))


def load_documents() -> list[dict]:
    """Return [{doc_id, title, text}] for every kb/*.md file."""
    if not KB_DIR.exists():
        raise FileNotFoundError(f"KB folder not found: {KB_DIR}")
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        title_line, _, body = raw.partition("\n\n")
        docs.append({
            "doc_id": path.stem,
            "title": title_line.lstrip("# ").strip(),
            "text": body.strip(),
        })
    return docs


def kb_doc_ids() -> set:
    """Ids of every document currently in kb/ (includes documents added at runtime)."""
    return {p.stem for p in KB_DIR.glob("*.md")}


def save_document(doc_id: str, title: str, text: str) -> dict:
    """Validate and write one new knowledge-base document.
    Raises ValueError (bad id/content) or FileExistsError (id already used)."""
    if not DOC_ID_RE.fullmatch(doc_id):
        raise ValueError("doc_id must be 3-60 chars: lowercase letters, digits, underscores, starting with a letter")
    n = count_sentences(text)
    if not 2 <= n <= 5:
        raise ValueError("document must have 2-5 sentences, found {}".format(n))
    path = KB_DIR / (doc_id + ".md")
    if path.exists():
        raise FileExistsError("document '{}' already exists".format(doc_id))
    path.write_text("# {}\n\n{}\n".format(title.strip(), text.strip()), encoding="utf-8")
    return {"doc_id": doc_id, "title": title.strip(), "text": text.strip()}


def verify(docs: list[dict] | None = None) -> None:
    docs = docs if docs is not None else load_documents()
    ids = {d["doc_id"] for d in docs}
    missing = sorted(set(REQUIRED_TOPICS) - ids)
    assert not missing, f"found {len(docs)} docs in {KB_DIR}; missing: {missing}"
    assert len(docs) >= 12, f"need >=12 documents, found {len(docs)}"
    for d in docs:
        n = count_sentences(d["text"])
        assert 2 <= n <= 5, f"{d['doc_id']} has {n} sentences (need 2-5)"


if __name__ == "__main__":
    docs = load_documents()
    verify(docs)
    print(f"{len(docs)} documents, all required topics covered\n")
    for d in docs:
        print(f"{d['doc_id']:<30} {count_sentences(d['text'])} sentences | {d['title']}")