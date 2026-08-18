# Week 1 v2: Minimal `/ask` Demo

This folder is the simplified class version of the Week 1 AI Engineering bootcamp demo.
Students run one final API and one small Streamlit page. The `stages/` files are optional
teaching references that show how the endpoint grows step by step.

## Memory store (Week 5 — capstone)

Jarvis has a **durable, cross-session memory** that survives process restarts and lives
outside the request/context window.

**In one paragraph (keep / write / live / retrieve / forget):** We **keep** a small set of
high-value durable facts — stable user preferences and project facts (e.g. billing
currency, main client) — as one-sentence statements, never raw chat turns. We **write**
through a gate: cheap rules pre-filter the turn, then a small model (`gpt-4o-mini`) decides
whether it is a durable fact and extracts `subject` + `fact`; only then do we embed and
insert (task-local chatter is dropped). It **lives** in an external **Neon Postgres database
with the `pgvector` extension** (serverless — survives redeploys and restarts), kept
separate from the app's ephemeral disk. We **retrieve** relevant facts before answering with
a hybrid query — `0.7 × cosine similarity + 0.3 × recency` — so a recent decision outranks
older chatter, and retrieval is by meaning (a query without the exact word still finds the
fact). We **forget** softly today via the recency term (older facts rank lower); hard
forgetting — a `stale_after` expiry, human-verified staging (`verified_by`), a real delete
path, and "new fact replaces the contradicted old one" — has schema columns ready but is
**deliberately deferred to the capstone** (the assignment is kept simple).

**Schema (`semantic_memory`, Postgres + pgvector):**

```sql
id bigserial PK · subject text · fact text · embedding vector(1536) ·
confidence real · source_event_id text · verified_by text · stale_after date ·
created_at timestamptz · updated_at timestamptz
```

Embeddings use `text-embedding-3-small` (1536 dims) — the same model as the RAG corpus.
Fields follow the Open Knowledge Format (OKF): `fact` ≈ description, `source_event_id` ≈
provenance, `verified_by` ≈ trust/staging, `stale_after` ≈ lifecycle. The write path lives
in `write_gate.py`, the store and hybrid read in `memory_store.py`, and the demo in the
"Pamięć (Week 5)" tab of `demo_page.py`.

**Deployment:** two Render services — `jarvis` (FastAPI backend, Docker) and `jarvis-ui`
(Streamlit, Python), both in Oregon. The memory tab lives in `jarvis-ui` and reaches Neon +
OpenAI via environment variables (`DATABASE_URL`, `OPENAI_API_KEY`). Cross-session recall is
demonstrated live: a fact written by a local process is recalled by the deployed app from a
query that never mentions it.

## What Students Will Build

A typed FastAPI endpoint that accepts a question and returns:

- `answer`: a structured answer object
- `tokens_used`: token usage returned by the model provider
- `model`: the model used for the request
- `latency_ms`: how long the request took
- `cost_usd`: an estimated request cost
- `attempts`: validation and retry details for the guardrail demo

The main idea: an LLM call becomes more useful in software when it has a predictable
request shape, a predictable response shape, and observable runtime metadata.

## Quick Start

Run these commands from this `week-1v2` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
test -f .env || cp .env.example .env
```

Open `.env` and add your key:

```bash
OPENAI_API_KEY=sk-...
```

## Terminal 1: Start the API

```bash
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Check that the API is running without spending tokens:

```bash
curl http://127.0.0.1:8000/health
```

You can also open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

## Terminal 2: Start the Demo Page

```bash
source .venv/bin/activate
streamlit run demo_page.py
```

Open:

```text
http://localhost:8501
```

Use the page to ask a question, switch models, inspect the JSON response, and copy the
equivalent `curl` request.

## Try the Guardrail Demo

Turn on **Force a bad first response to demo validation + retry** in the Streamlit page.
The API intentionally asks the model for malformed JSON on the first attempt, validates
that response with Pydantic, records the failure, and retries with structured output.

This is a small classroom-friendly example of a production habit: do not trust free-form
LLM output at the boundary of your application.

## Test With Curl

Normal request:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Retrieval-Augmented Generation in one sentence?", "model": "gpt-4o-mini"}'
```

Validation and retry demo:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a vector database?", "model": "gpt-4o-mini", "force_bad": true}'
```

## Instructor Flow

Use `main.py` and `demo_page.py` for the live student demo. Open the stage files only when
you want to explain how each capability was introduced:

| Stage | File | Teaching point |
|-------|------|----------------|
| 1 | `stages/stage_1_bare_ask.py` | Smallest typed `/ask`: question in, string answer out. |
| 2 | `stages/stage_2_structured_output.py` | Add a Pydantic `Answer` schema and OpenAI structured output. |
| 3 | `stages/stage_3_guardrails_and_observability.py` | Add validation retry, model selection, latency, and cost. |

Run one stage at a time if you want to teach the build-up live:

```bash
uvicorn stages.stage_1_bare_ask:app --host 127.0.0.1 --port 8000 --reload
uvicorn stages.stage_2_structured_output:app --host 127.0.0.1 --port 8000 --reload
uvicorn stages.stage_3_guardrails_and_observability:app --host 127.0.0.1 --port 8000 --reload
```

## Smoke Test

This starts the final API, checks `/health` and `/docs`, and does not call OpenAI:

```bash
source .venv/bin/activate
python smoke_test.py
```

## File Map

```text
week-1v2/
├── README.md
├── main.py                         # Final API used by students
├── demo_page.py                    # Streamlit UI for the final API
├── smoke_test.py                   # No-token API startup check
├── requirements.txt
├── .env.example
├── .gitignore
└── stages/
    ├── stage_1_bare_ask.py
    ├── stage_2_structured_output.py
    └── stage_3_guardrails_and_observability.py
```

## Troubleshooting

- `Cannot reach http://127.0.0.1:8000`: start the API server in another terminal.
- `OPENAI_API_KEY` error: make sure `.env` exists and contains a real key.
- `Address already in use`: another server is already using port `8000`; stop it or use a different port.
- Streamlit opens but requests fail: confirm the sidebar API base URL is `http://127.0.0.1:8000`.
