"""Task 12: ELK-style structured logging. One JSON object per line (JSON-Lines).

Every entry carries a trace_id and timing information. PII is masked with the SAME
function the input guardrail uses (guardrails.mask_phone), twice over:
  1. callers pass question text through safe_text() before it goes into an entry, and
  2. write_log() masks the fully serialised line as a final gate, so a fixed-format phone
     number cannot reach disk through any field, including ones added in the future.
"""
import datetime
import json
import os
import threading
import uuid
from pathlib import Path

from guardrails import mask_phone

_write_lock = threading.Lock()


def log_path() -> Path:
    """Read on every write so a demo or test can redirect the log with NAUKRI_LOG_PATH."""
    return Path(os.environ.get("NAUKRI_LOG_PATH", Path(__file__).parent / "logs" / "requests.jsonl"))


def new_trace_id() -> str:
    return uuid.uuid4().hex


def safe_text(text: str) -> str:
    """Mask fixed-format PII before the text is put into a log entry."""
    return mask_phone(text)[0]


def write_log(entry: dict) -> None:
    entry = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"), **entry}
    line = mask_phone(json.dumps(entry, ensure_ascii=False))[0]       # final gate
    path = log_path()
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")