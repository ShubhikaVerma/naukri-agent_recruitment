# Naukri.com Domain Support Agent (CrewAI)

**Track completed: Naukri.com (Recruitment & HR)**

A domain support agent that answers hiring-policy questions from a knowledge base I wrote and looks up
job-application status from a dataset I designed. It remembers a conversation, is guarded against misuse,
has its own answers reviewed by a second independent agent team, runs under an explicit governance policy,
and is served by FastAPI.

**Everything runs under `MOCK_LLM`: zero API keys and no LLM network calls.** Every acceptance criterion is
demonstrated under `MOCK_LLM` alone; the transcripts in `transcripts/` were produced that way.

## Contents

1. [Task 1 dataset design choices](#task-1--dataset-design-choices-reproducible)
2. [How to run](#how-to-run)
3. [How the pieces fit](#how-the-pieces-fit)
4. [Part 1: dataset and RAG core](#part-1--dataset-and-rag-core)
5. [Part 2: CrewAI crew, memory, schema, guardrails](#part-2--crewai-orchestration-tools-memory-and-guardrails)
6. [Part 3: FastAPI, logging, evaluation](#part-3--fastapi-logging-and-evaluation)
7. [Part 4: review stage, governance, caching](#part-4--resilience-and-governance)
8. [Known limitations](#known-limitations)
9. [Task to transcript map](#task-to-transcript-map)

---

## Task 1 / dataset design choices (reproducible)

`dataset.py` is a seeded, deterministic generator (`random.Random(SEED)`); running it twice gives identical records.

| Choice | Value |
|---|---|
| Seed | `42` |
| Records | `48` |
| Category weights | Software Engineer 0.30, Data Analyst 0.22, Product Manager 0.16, HR Executive 0.16, Sales Associate 0.16 |
| Status weights | Applied 0.35, Screening 0.25, Interview Scheduled 0.18, Offered 0.10, Rejected 0.12 |
| `flagged_priority_review` | Bernoulli(0.18) per record |
| `days_since_created` | uniform integer 0 to 30 |

**Expected salary (INR per year), uniform per category, rounded to Rs 10,000:**

| Category | Range |
|---|---|
| Software Engineer | 6,00,000 to 36,00,000 |
| Data Analyst | 4,00,000 to 20,00,000 |
| Product Manager | 9,00,000 to 32,00,000 |
| HR Executive | 3,00,000 to 12,00,000 |
| Sales Associate | 2,40,000 to 9,00,000 |

*Reasoning (one sentence):* the ranges approximate fresher-to-mid-level annual CTC expectations in the Indian
market, from Rs 2.4 lakh for entry-level sales up to Rs 36 lakh for senior engineers.

**Result of the first draw (no regeneration, no hand-editing):**

| Category | Count | Status | Count |
|---|---|---|---|
| Software Engineer | 12 | Applied | 17 |
| Data Analyst | 11 | Screening | 14 |
| Product Manager | 10 | Interview Scheduled | 5 |
| HR Executive | 5 | Offered | 7 |
| Sales Associate | 10 | Rejected | 5 |

`flagged_priority_review=True`: **16.67%** (inside the required 10 to 30% band). `validate()` in `dataset.py`
asserts every structural threshold. Each record also carries a fabricated, fixed-format `contact_phone`, used to
demonstrate the PII guardrail.

---

## How to run

Tested on Windows, PowerShell, Python 3.11.9.

```powershell
python -m venv C:\venvs\naukri            # keep the venv outside any OneDrive folder
C:\venvs\naukri\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt           # requirements.lock.txt holds the exact versions I ran
.\run_all.ps1                             # regenerates every transcript in dependency order
.\run_all.ps1 -Only task14_review         # or just one step
```

| Need | Detail |
|---|---|
| API keys | none |
| Network | **One-time download** of the free `all-MiniLM-L6-v2` embedding model from Hugging Face on the first run; cached afterwards. No LLM call ever leaves the machine. |
| `MOCK_LLM` | defaults to `true`. The crew, the grounded generator, the judge and the review team all use deterministic mock models. A real LLM is only a hook (`NotImplementedError`) behind `MOCK_LLM=false`. |
| CrewAI telemetry | **Set, as required.** `crew.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` before `crewai` is imported. In the CrewAI version used (1.15.30) a separate trace-sharing prompt exists, so it also sets `CREWAI_TRACING_ENABLED=false` and builds the crew with `tracing=False`. |
| Vector store | ChromaDB, persisted to `~/.naukri_chroma` (outside the repo; override with `NAUKRI_CHROMA_DIR`). `python rag_index.py` rebuilds it from `kb/`. |
| Server | `uvicorn app:app --port 8000`; live WebSocket test: `python ws_live_test.py` in a second terminal. |

Expected warnings (not errors): `LangChainDeprecationWarning` for `RunnableWithMessageHistory` and
`InMemoryChatMessageHistory`. They are left visible, as the brief says.

---

## How the pieces fit

```
client (HTTP /ask, WebSocket /ws/chat)
  -> logging middleware        one JSON-Lines entry per request: trace id, timings, masked text
  -> conversation.chat         injection check, phone masking, then session memory (LangChain)
  -> pipeline.process_question
       runtime budget          pre-flight estimate, then every model call metered
       guardrails.guarded_ask  input guardrails -> CrewAI crew -> output groundedness check
           crew: Retrieval Agent (rag_lookup, response cache) -> Lookup Agent (status tool) -> Response Composer
       Autogen review team     Policy_Compliance_Reviewer + Final_Editor, approves or revises
  -> structured SupportResponse (Pydantic), validated before it leaves
```

---

## Part 1 / dataset and RAG core

**Knowledge base (Task 2).** `kb/` holds 12 documents of 4 sentences each, one per required topic:
job-application eligibility, interview scheduling, offer negotiation, background verification, notice period,
referral bonus, internal transfer, probation, remote work, diversity hiring, exit interview, applicant-data
retention. `knowledge_base.py` verifies the count, the topics and the 2 to 5 sentence rule.

**Chunking and indexing (Task 3).** Embeddings come from `all-MiniLM-L6-v2` (normalised, cosine space). Two
strategies, each in its own ChromaDB collection, indexed with `collection.upsert()`:

| Collection | Strategy | Chunks |
|---|---|---|
| `kb_fixed` | 40-word window, 10-word overlap | 30 |
| `kb_sentence` | one chunk per sentence | 48 |

**Grounded generation and the "I don't know" threshold (Task 4).** The threshold was calibrated on measured
top-1 cosine similarities (4 in-scope and 3 out-of-scope queries, including the hard case "Which company pays
software engineers the most?") and set at the midpoint of the observed gap, not a preset:

| Collection | In-scope top-1 (measured) | Out-of-scope top-1 (measured) | Chosen threshold |
|---|---|---|---|
| `kb_fixed` | 0.736, 0.493, 0.582, 0.652 | 0.122, 0.028, 0.246 | **0.369** (gap 0.246 to 0.493) |
| `kb_sentence` | 0.728, 0.499, 0.594, 0.777 | 0.174, 0.066, 0.268 | **0.384** (gap 0.268 to 0.499) |

Demonstrated on 5 in-scope queries plus 1 out-of-scope query ("What is the capital of France?") on both
collections; the out-of-scope query triggers the fallback (`transcripts/task4_grounded.txt`).

Generation is extractive, so it can only repeat what the documents say. After retrieval it works from the
parent documents of the retrieved chunks, because a fixed-size chunk can start or end mid-sentence. For a
compound question ("how long is X and when is Y reviewed?") it quotes the best sentence for each clause, at
most three sentences in total.

**Chunking comparison (Task 5).** Document-level precision and recall, top-k = 3 chunks, chunks mapped to
parent documents and deduplicated before scoring (full per-query arithmetic in `transcripts/task5_eval.txt`):

| Query | kb_fixed precision | kb_fixed recall | kb_sentence precision | kb_sentence recall |
|---|---|---|---|---|
| Q1 referral bonus | 1/1 = 1.000 | 1/1 = 1.000 | 1/1 = 1.000 | 1/1 = 1.000 |
| Q2 background verification | 1/2 = 0.500 | 1/1 = 1.000 | 1/2 = 0.500 | 1/1 = 1.000 |
| Q3 remote work during probation | 2/2 = 1.000 | 2/2 = 1.000 | 2/2 = 1.000 | 2/2 = 1.000 |
| Q4 data retention | 1/1 = 1.000 | 1/1 = 1.000 | 1/1 = 1.000 | 1/1 = 1.000 |
| Q5 notice period | 1/1 = 1.000 | 1/1 = 1.000 | 1/2 = 0.500 | 1/1 = 1.000 |
| **Mean** | **0.900** | **1.000** | **0.800** | **1.000** |

**Recommendation.** I deploy **fixed-size chunking with overlap** (`kb_fixed`). Both strategies reach recall
1.000, but fixed-size scores higher mean precision (0.900 against 0.800): the sentence collection pulled in an
extra document on Q2 (`applicant_data_retention`) and on Q5 (`probation_period`, which does mention the
shortened notice period, so that miss is partly my strict ground truth), while fixed-size missed only on Q2
(`interview_scheduling`). Caveat: this is five queries, and the gap is one query's worth. Fixed chunks also
start and end mid-sentence, which is why generation expands to the parent document.

---

## Part 2 / CrewAI orchestration, tools, memory and guardrails

**Second tool and escalation score (Task 6).** `check_job_application_status(record_id)` returns `status`,
`expected_salary_inr` and a designed score:

```
escalation_score = 0.5 * flagged_priority_review + 0.5 * (days_since_created / 30)      in [0, 1]
```

`days_since_created / 30` is read as "how long the application has been open without resolution", so a longer
wait raises urgency. **Recommended escalation threshold: 0.6.** From my generated data: `days_since_created`
percentiles p50 = 13, p80 = 22, p90 = 27; score percentiles p50 = 0.291, p80 = 0.477, p90 = 0.617. At 0.6, 6 of
48 applications (12.5%, roughly the top of the score distribution) are recommended for escalation. A
non-flagged application tops out at 0.5, so only flagged applications can escalate, and a flagged one must be
at least 6 days old (0.5 + 0.5 x 6/30 = 0.6). Evidence: `transcripts/task6_escalation.txt`.

**Crew (Task 7).** Three agents run by `Crew.kickoff()`: a Retrieval Agent (`rag_lookup`), a Lookup Agent
(`check_job_application_status`) and a Response Composer (no tools). `mock_llm.py` extends
`crewai.llms.base_llm.BaseLLM` and avoids both pitfalls in the brief: the tool result is parsed only from the
model's own previous assistant message (never from the system prompt, which contains "Observation: the result
of the action"), and tools are chosen by their declared argument schema (`query` or `record_id`), not by name.
`TOOL_CALLS` records which agent called which tool; the transcript shows the RAG tool on policy questions, the
lookup tool on status questions, and both on a mixed question.

**Session memory (Task 8).** LangChain `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`, one history
per `session_id`, in-process only. The history is genuinely used: a follow-up such as "is it recommended for
escalation?" resolves to the application the user named earlier in that session, and "what was my first
question?" is answered from the stored messages. `transcripts/task8_memory.txt` shows a multi-turn session and
a separate fresh session where the same questions correctly find nothing.

**Structured output (Task 9).** `schemas.SupportResponse` (Pydantic) with `answer`, `answer_type`
(`policy`, `application_status`, `policy_and_status`, `fallback`, `conversation`), `grounded`, `sources`,
`application` and `tools_used`. The composing task declares `output_pydantic=SupportResponse`, and `run_crew`
validates every raw response in code and cross-checks `tools_used` against the tools actually called.
Validators reject an out-of-range score, a malformed record id, a policy answer with no source, an invented
source document, and a "grounded" fallback (all shown in `transcripts/task9_schema.txt`).

**Guardrails (Task 10).**

| Guardrail | Behaviour |
|---|---|
| Phone masking (input) | Indian mobile formats become `[PHONE_MASKED]` before the crew sees them; demonstrated end to end (the tool input contains the placeholder, not the number). |
| Prompt-injection detection (input) | Pattern-based; blocks the request before any agent runs. The five normal demo queries are not flagged. |
| Groundedness (output) | Every policy sentence must be a sentence of a cited document, and every figure in a status sentence must match the looked-up record; otherwise the answer is refused. Fired on an out-of-scope question and on a deliberately invented "Rs 50,000 bonus" claim. |

Out of scope for masking (no reliable pattern under a keyless masker, as the brief states): candidate name,
expected salary, background-check results. Only fabricated data is used.

---

## Part 3 / FastAPI, logging and evaluation

**Deployment (Task 11).** `app.py`: `POST /ask` (optional `session_id`), `POST /add-document` (validates 2 to
5 sentences, writes `kb/<id>.md`, upserts into both collections, refreshes caches), `GET /health`, and
`WS /ws/chat`. All use Pydantic request and response models. The WebSocket handler catches
`WebSocketDisconnect` and keeps serving other clients. `api_demo.py` covers a client leaving between turns;
`ws_live_test.py` runs against a real `uvicorn` server and also drops a client while its reply is still being
computed.

**Structured logging (Task 12).** One JSON-Lines entry per request in `logs/requests.jsonl`, with `trace_id`
(also returned as the `X-Trace-ID` header), `timing_ms.total` plus `queue_wait` and `processing`, endpoint,
status, masked question, answer type, tools, guardrail events, review status and tokens used. Phone masking is
applied to the text and again to the finished line, so a raw phone number never reaches disk. Rejected
requests (HTTP 422) are logged too. `logs/demo_requests.jsonl` is the evidence file, and
`transcripts/task12_logging.txt` ends with 10 verification checks.

**Evaluation (Task 13).** 15 queries run through the full deployed pipeline: one per required topic, plus a
status lookup carrying a phone number, an out-of-scope question and a prompt injection. A judge prompt (printed
in the transcript) scores Accuracy, Grounding, Completeness and Safety from 1 to 5. Under `MOCK_LLM` the judge
is a deterministic rubric-follower that reads only the prompt; it is a stand-in, not a model's opinion.

| ID | Topic / type | Acc | Grd | Cmp | Saf |
|---|---|---|---|---|---|
| T01 | job_application_eligibility | 5 | 5 | 5 | 5 |
| T02 | interview_scheduling | 5 | 5 | 5 | 5 |
| T03 | offer_negotiation | 5 | 5 | 5 | 5 |
| T04 | background_verification | 5 | 5 | 5 | 5 |
| T05 | notice_period | 5 | 5 | 5 | 5 |
| T06 | referral_bonus | 5 | 5 | 5 | 5 |
| T07 | internal_transfer | 5 | 5 | 5 | 5 |
| T08 | probation_period | 5 | 5 | 5 | 5 |
| T09 | remote_work_eligibility | 5 | 5 | 5 | 5 |
| T10 | diversity_hiring | 5 | 5 | 5 | 5 |
| T11 | exit_interview | 5 | 5 | 3 | 5 |
| T12 | applicant_data_retention | 5 | 5 | 5 | 5 |
| T13 | edge: status lookup + phone number | 5 | 5 | 5 | 5 |
| T14 | edge: out of scope | 5 | 5 | 5 | 5 |
| T15 | edge: prompt injection | 5 | 5 | 5 | 5 |
| | **Average over 15** | **5.00** | **5.00** | **4.87** | **5.00** |

A sanity check in the same transcript feeds the judge five deliberately bad answers (hallucination,
incomplete, wrong document, injection compliance, out-of-scope answered) and shows it scoring them low.

**Evaluation drove a fix.** The first run (kept as `transcripts/task13_eval_baseline.txt`) averaged accuracy
4.87 / grounding 5.00 / completeness 4.60 / safety 5.00. T08 (probation length and review timing) and T11 lost
points because the generator ranked sentences against the whole question and picked two that matched the same
part. Quoting the best sentence per clause of a compound question fixed T08 (3/5/1/5 to 5/5/5/5). T11 still
scores 3 on completeness: for "is it mandatory?" the generator quotes "Final settlement is not dependent on
attending..." rather than "Participation is voluntary...".

---

## Part 4 / resilience and governance

**Autogen review stage (Task 14).** `review_stage.py` builds a two-agent `RoundRobinGroupChat`
(`Policy_Compliance_Reviewer`, `Final_Editor`) bounded with `max_turns=2`. The Final_Editor has
`output_content_type=VerdictModel` (`approved: bool`, `final_answer: str`, `reason: str`) and the team is
created with `custom_message_types=[StructuredMessage[VerdictModel]]`. The team receives the Composer's draft
and the retrieved context, and either approves the draft unchanged or removes unsupported sentences. Four
demos, all passing: a genuine policy draft approved unchanged, a draft with an injected ungrounded sentence
("Late submissions are accepted for a fee of Rs 500 per day.") revised, a genuine status draft approved, and a
status draft whose salary contradicts the record revised to a refusal. Each model call goes through
`MockReviewClient`, a deterministic `ChatCompletionClient`. Autogen version used: `autogen-agentchat` (see
`requirements.lock.txt`).

**Application layer: least autonomy and risk (Task 15).** Only the Lookup Agent may call
`check_job_application_status`; only the Retrieval Agent may call `rag_lookup`; the Composer has no tools.
`ALLOWED_TOOL_ROLES` in `governance.py` is the single source of truth and is enforced twice: `make_agent()`
refuses to wire a tool to the wrong agent (wiring time), and each tool instance re-checks its own bound role
before it runs (call time), so code that builds an agent directly and skips `make_agent()` is still stopped.
`transcripts/task15_governance.txt` shows four attempts to break the rule, all blocked, and a legitimate run
whose logged callers match the permission table.

**Risk classification: High.** The assistant reads individual job-application records, including expected
salary (financial data), and produces an escalation recommendation that can influence how a candidate's
application is handled, so it sits next to hiring decisions even though it is read-only and never decides
anything itself. Answering policy questions alone would be Medium (customer support), but errors or leaks
affect real people's employment and pay, so the stricter controls apply.

**Runtime layer: per-request budget (Task 15).** A cap of **8,000 tokens per request** covers every model call
(crew, grounded generation and review team) in one `UsageMeter`, with an illustrative flat price so it reads as
money too. It is enforced twice: *pre-flight* (an estimate of 3,500 + 6 tokens per question token, measured on
this crew, rejects an oversized request before any model is called) and *mid-flight* (the meter aborts the
request the moment the running total passes the cap). A deliberately oversized 4,000-character request is
rejected up front with no model calls; a normal request run with a tiny budget is aborted mid-flight; over HTTP
the rejection is a 413. Measured usage of normal requests is in the transcript. Token counts use about 4
characters per token, since the mock models have no tokenizer.

**Response caching (Task 16).** `grounded.cached_grounded_answer` wraps the grounded-generation step in an
in-memory cache keyed by the normalised query (lower-case, punctuation and spacing ignored), 256 entries,
least-recently-used eviction. It is cleared whenever the knowledge base changes (`POST /add-document`), so no
stale answer survives. It never caches application records. `transcripts/task16_cache.txt` gives before and
after evidence: the first ask makes 1 generation call and 1 vector search; the identical repeat makes 0 and 0
and is served in a small fraction of the time; a re-spaced and re-cased variant also hits; a different question
misses; and end to end the second identical request uses fewer tokens with the same answer.

---

## Known limitations

* **The judge is a rubric-following mock.** Grounding and Safety at 5.00 are close to guaranteed by design (the
  generator is extractive and the guardrails are deterministic), so they act as regression checks. The
  Accuracy and Completeness columns are where the system can actually lose points.
* **Relevance is not scored.** The generator can append a true but unrequested sentence (for example the
  probation sentence in T07); the judge scores it as grounded.
* **T11 completeness is 3** (see above).
* **Chunking comparison rests on five queries**; the 0.1 precision gap is one query's worth.
* **Prompt-injection detection is pattern-based**; a creatively reworded attack could slip past it. The
  groundedness check and the review stage are the stronger controls.
* **Reference resolution in memory is rule-based** (last application id the user typed), because there is no
  model to do coreference under `MOCK_LLM`. Memory is in-process only and is lost on restart.
* **The escalation score ignores status**, so a Rejected or Offered application can be recommended for
  escalation. The brief defines the score as flag plus recency, so I kept it as specified.
* **Crew runs are serialised with a lock** because the tool-call log and per-turn state are process-global.
  Fine for this scope, not for high concurrency.
* **Caching saves work, not wall-clock time**: the crew's own steps dominate request latency.
* **`/add-document` writes into `kb/`.** `api_demo.py` removes its demo document and rebuilds the index so the
  Part 1 numbers stay reproducible.
* **Phone numbers are the only masked field**, as the brief specifies.

---

## Task to transcript map

| Task | Code | Evidence |
|---|---|---|
| 1 Dataset | `dataset.py` | `transcripts/task1_dataset.txt` |
| 2 Knowledge base | `kb/`, `knowledge_base.py` | `transcripts/task2_kb.txt` |
| 3 Chunking + indexing | `rag_index.py` | `transcripts/task3_index.txt` |
| 4 Grounded generation + threshold | `calibrate.py`, `grounded.py`, `queries.py` | `task4_calibration.txt`, `task4_grounded.txt`, `calibration.json` |
| 5 Precision / recall | `eval_chunking.py` | `task5_eval.txt`, `eval_chunking.json` |
| 6 Lookup tool + escalation | `lookup_tool.py`, `escalation_analysis.py` | `task6_lookup.txt`, `task6_escalation.txt` |
| 7 CrewAI crew | `crew.py`, `crew_tools.py`, `mock_llm.py` | `task7_crew.txt` |
| 8 Session memory | `conversation.py`, `memory_demo.py` | `task8_memory.txt` |
| 9 Structured output | `schemas.py`, `schema_demo.py` | `task9_schema.txt` |
| 10 Guardrails | `guardrails.py`, `guardrails_demo.py` | `task10_guardrails.txt` |
| 11 FastAPI | `app.py`, `api_demo.py`, `ws_live_test.py` | `task11_api.txt` |
| 12 Logging | `logging_utils.py`, `logging_demo.py` | `task12_logging.txt`, `logs/demo_requests.jsonl` |
| 13 Evaluation | `eval_set.py`, `judge.py`, `evaluate.py` | `task13_eval.txt`, `task13_eval_baseline.txt`, `eval_results.json` |
| 14 Autogen review | `review_stage.py`, `review_demo.py` | `task14_review.txt` |
| 15 Governance + budget | `governance.py`, `runtime_guard.py`, `metered_llm.py`, `pipeline.py`, `governance_demo.py` | `task15_governance.txt` |
| 16 Response cache | `grounded.py`, `cache_demo.py` | `task16_cache.txt` |

All transcripts are in `transcripts/`. `run_all.ps1` regenerates them in dependency order.