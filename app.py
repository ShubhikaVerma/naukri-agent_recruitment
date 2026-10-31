"""Task 11 + 12: FastAPI deployment of the guarded, memory-enabled CrewAI support agent,
with one structured JSON-Lines log entry per request (trace id, timings, masked text).

HTTP      POST /ask            ask a question (optionally within a session)
          POST /add-document   add a policy document to the knowledge base and index it
          GET  /health         liveness check
WebSocket /ws/chat             real-time multi-turn chat; a client disconnecting mid-conversation
                               is caught (WebSocketDisconnect) and never affects other clients

Run:  uvicorn app:app --port 8000
Crew runs are serialised with a lock because the tool-call log and the per-turn state
(crew_tools.TOOL_CALLS, conversation.LAST_RUN) are process-global.
"""
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError

import conversation
import grounded
import guardrails
from knowledge_base import save_document
from logging_utils import new_trace_id, safe_text, write_log
from rag_index import add_document_to_index, embed
from schemas import SupportResponse

_crew_lock = threading.Lock()
DISCONNECT_LOG = []                               # session ids of clients that disconnected


# ----------------------------------------------------------------- request / response models
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, max_length=64,
                                      description="Reuse the returned id to continue a conversation.")


class GuardrailEvent(BaseModel):
    guardrail: str
    action: str
    detail: str


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    answer: str
    response: SupportResponse
    guardrail_events: list[GuardrailEvent]
    tools_invoked: list[str]


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,59}$")
    title: str = Field(min_length=3, max_length=120)
    text: str = Field(min_length=10, max_length=2000, description="2-5 sentences")


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    title: str
    chunks_fixed: int
    chunks_sentence: int


# ----------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    embed(["warm up the embedding model"])        # load the model once, before the first request
    yield


app = FastAPI(title="Naukri.com Support Agent", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    """Exactly one JSON-Lines entry per HTTP request, including rejected (422) and failed ones."""
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    request.state.log = {}                        # endpoints add masked question, answer type, timings...
    t0 = time.perf_counter()
    status_code, error = 500, None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Trace-ID"] = trace_id
        return response
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        fields = dict(request.state.log)
        timing = dict(fields.pop("timing_ms", {}))
        timing["total"] = round((time.perf_counter() - t0) * 1000, 3)
        entry = {"event": "request", "trace_id": trace_id, "transport": "http",
                 "method": request.method, "path": request.url.path, "status_code": status_code,
                 **fields, "timing_ms": timing}
        if error:
            entry["error"] = error
        write_log(entry)


def answer_question(session_id: str, question: str) -> dict:
    """The single entry point every transport (HTTP, WebSocket) uses."""
    t0 = time.perf_counter()
    with _crew_lock:
        t1 = time.perf_counter()
        out = conversation.chat(session_id, question)
        t2 = time.perf_counter()
    out["timing_ms"] = {"queue_wait": round((t1 - t0) * 1000, 3), "processing": round((t2 - t1) * 1000, 3)}
    return out


def log_fields(out: dict, question: str) -> dict:
    """What gets logged about an answered question. The text is masked first."""
    return {"session_id": out["session_id"], "question": safe_text(question),
            "answer_type": out["response"]["answer_type"], "grounded": out["response"]["grounded"],
            "tools_invoked": [c["tool"] for c in out["tool_calls"]],
            "guardrail_events": [e["guardrail"] + ":" + e["action"] for e in out["events"]],
            "timing_ms": out["timing_ms"]}


def _to_ask_response(out: dict, trace_id: str) -> AskResponse:
    return AskResponse(
        trace_id=trace_id, session_id=out["session_id"], answer=out["answer"],
        response=SupportResponse.model_validate(out["response"]),
        guardrail_events=[GuardrailEvent(**e) for e in out["events"]],
        tools_invoked=[c["tool"] for c in out["tool_calls"]])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    session_id = req.session_id or uuid.uuid4().hex[:12]
    request.state.log.update(session_id=session_id, question=safe_text(req.question))
    try:
        out = answer_question(session_id, req.question)
    except ValidationError as exc:
        request.state.log["error"] = "ResponseValidationError"
        raise HTTPException(status_code=502, detail="the agent produced a response that failed "
                                                    "schema validation: {}".format(exc.errors()[0]["msg"]))
    request.state.log.update(log_fields(out, req.question))
    return _to_ask_response(out, request.state.trace_id)


@app.post("/add-document", response_model=AddDocumentResponse, status_code=201)
def add_document(req: AddDocumentRequest, request: Request):
    request.state.log.update(doc_id=req.doc_id, title=safe_text(req.title), text_chars=len(req.text))
    with _crew_lock:
        try:
            doc = save_document(req.doc_id, req.title, req.text)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        counts = add_document_to_index(doc)           # both ChromaDB collections
        grounded.reset_cache()                        # generation + groundedness check see the new text
        guardrails.reset_cache()
    request.state.log.update(chunks_fixed=counts["fixed"], chunks_sentence=counts["sentence"])
    return AddDocumentResponse(trace_id=request.state.trace_id, doc_id=doc["doc_id"], title=doc["title"],
                               chunks_fixed=counts["fixed"], chunks_sentence=counts["sentence"])


def _extract_question(raw: str) -> str:
    """Accept either plain text or JSON like {"question": "..."}."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return str(json.loads(raw).get("question", "")).strip()
        except ValueError:
            return ""
    return raw


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, session_id: Optional[str] = None):
    await websocket.accept()
    sid = session_id or uuid.uuid4().hex[:12]
    write_log({"event": "ws_connect", "trace_id": new_trace_id(), "transport": "websocket",
               "path": "/ws/chat", "session_id": sid})
    await websocket.send_json({"type": "connected", "session_id": sid})
    try:
        while True:
            question = _extract_question(await websocket.receive_text())
            trace_id, t0 = new_trace_id(), time.perf_counter()
            if not question or len(question) > 1000:
                write_log({"event": "request", "trace_id": trace_id, "transport": "websocket",
                           "path": "/ws/chat", "session_id": sid, "status": "rejected_invalid_message",
                           "timing_ms": {"total": round((time.perf_counter() - t0) * 1000, 3)}})
                await websocket.send_json({"type": "error", "detail": "send a question of 1-1000 characters"})
                continue

            fields = {"session_id": sid, "question": safe_text(question), "status": "ok"}
            try:
                out = await run_in_threadpool(answer_question, sid, question)
                fields.update(log_fields(out, question))
                payload = _to_ask_response(out, trace_id).model_dump()
                await websocket.send_json({"type": "answer", **payload})
            except WebSocketDisconnect:
                fields["status"] = "client_disconnected"
                raise
            except Exception as exc:                  # one bad turn must not kill the connection or server
                fields.update(status="error", error=type(exc).__name__)
                try:
                    await websocket.send_json({"type": "error", "detail": "request failed: {}".format(type(exc).__name__)})
                except Exception:
                    pass                              # the client is gone; the next receive_text() will say so
            finally:
                timing = dict(fields.pop("timing_ms", {}))
                timing["total"] = round((time.perf_counter() - t0) * 1000, 3)
                write_log({"event": "request", "trace_id": trace_id, "transport": "websocket",
                           "path": "/ws/chat", **fields, "timing_ms": timing})
    except WebSocketDisconnect:
        DISCONNECT_LOG.append(sid)
        write_log({"event": "ws_disconnect", "trace_id": new_trace_id(), "transport": "websocket",
                   "path": "/ws/chat", "session_id": sid})
        print("[ws] client {} disconnected; server keeps running".format(sid))