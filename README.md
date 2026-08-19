# Jarvis — a private document assistant that refuses to guess

**Live demo:** https://jarvis-ui-3967.onrender.com · **API health:** https://jarvis-8lpg.onrender.com/health
**Backup demo recording:** _(link TBD)_

Built as the capstone of a 6-week AI engineering bootcamp (RAG → agents → evals → memory), then hardened for Demo Day.

## Problem

I want one assistant for myself and my family businesses that answers questions about **our own documents** — policies, contracts, invoices — and nothing else. When the documents don't contain the answer, it must say *"I don't know"* instead of guessing: a wrong-but-confident answer about a payment deadline is worse than no answer. The corpus is bilingual (Polish + English) and questions come in both languages, so cross-language retrieval is a requirement, not a nice-to-have.

## Architecture

**Ask path (RAG):**

```
Browser → Streamlit UI (jarvis-ui) → FastAPI backend (jarvis)
  → ChromaDB retrieval (top-5)
  → grounded prompt → gpt-4o-mini, structured output (Pydantic Answer)
  → answer + cited document IDs — or an explicit refusal (i_dont_know: true)
```

Every response returns tokens, latency and cost; every run (ask / raw-loop agent / LangGraph agent) is appended to `traces/traces.jsonl` with tool calls and prompt version — that trace log is what the eval suite reads.

**Memory path (separate from RAG):**

```
user turn → rule pre-filter (no LLM) → write gate: gpt-4o-mini → {save, subject, fact}
  → embedding → Neon Postgres + pgvector (external, survives restarts)
recall: 0.7·cosine similarity + 0.3·recency, read straight from the DB in a fresh session
```

**Agent (two engines):** the same retrieval behind a Think → Act → Observe loop, implemented twice — a hand-written raw loop and a LangGraph `StateGraph` — sharing one system prompt. Capped at 6 iterations, fail-closed; the loop reformulates the search query itself when the first search comes back empty.

**Security decision carried through the project:** retrieved document text and tool output are treated as *data, never instructions* (prompt-injection rule from Week 3); a fixed injection question is part of the eval set, and the write gate applies the same rule before anything reaches memory.

## Stack

FastAPI + Docker (backend service) · Streamlit (UI service) · ChromaDB (vector store for the corpus) · `text-embedding-3-small` (one embedding model for corpus **and** memory) · `gpt-4o-mini` (answers, agent, write gate) · LangGraph + raw loop (agent engines) · Neon Postgres + `pgvector` (durable memory) · Render (two services, Oregon) · evals in plain Python (no framework).

## Evals — the TRACE loop

Built from real traces, not imagined cases:

- **20 traces open-coded by hand first** (15 pass / 5 fail), then a **5-category failure taxonomy** ranked by frequency × impact — only then tests.
- **7 code assertions, zero LLM judges**, each tied to a taxonomy category; they score the newest run of each of 20 fixed questions (PL/EN, multi-hop, trap and injection cases) — **108 binary checks** per run.
- **Shipped fix** (in the deployed agent prompt, both engines): two-step fragment assessment — does a fragment answer *directly*, or does the answer *follow from a rule/limit/date in it* — plus a "test before refusing" rule: if your refusal would cite the rule the answer follows from, give the answer instead.
- **Result: 94.4% → 96.3%** (102 → 104 of 108) with the final prompt. Intermediate iterations measured 97.2%; the gap is a single assertion that flips at temp=0 (q09: 5↔6 tool calls), which is why single runs are not treated as verdicts — durable wins were q13 (agent stopped over-searching: 5 → 2–3 tool calls per run) and a shorter prompt (−232 tokens per run vs the first fix).
- **Known gap, stated openly:** q12 — a false refusal on a question whose answer must be *inferred* from a rule. Three targeted prompt iterations missed it; in one, the model wrote the correct inference inside its refusal and still refused. Diagnosis: a capability limit of `gpt-4o-mini` on this route, not a prompt bug — candidate for a stronger model there.
- Whole experiment: 81 traces, **$0.0384** in API cost; a single grounded answer costs ~$0.0002 at 1.5–2.5 s.

The Streamlit **Evals tab** shows the score, a per-assertion before/after table against the committed baseline, and the failures — including the known gap.

## Memory in five answers

- **What do I keep?** 1–3 high-value durable facts (billing currency, main client) as one-sentence statements — never raw chat turns.
- **When do I write?** Through a gate: cheap rules pre-filter the turn, then `gpt-4o-mini` decides *durable fact or not* and extracts `subject` + `fact`; chatter and questions are dropped before any embedding is paid for.
- **Where does it live?** Neon Postgres with `pgvector` (`semantic_memory`: subject, fact, `vector(1536)`, confidence, `source_event_id` for provenance, `verified_by`, `stale_after`) — external to the app host, so it survives redeploys, restarts and free-tier sleep.
- **How do I get it back?** Hybrid query — `0.7 × cosine + 0.3 × recency` — so retrieval works by meaning (a query that never mentions "EUR" still finds the billing fact) and recent decisions outrank old ones.
- **When do I forget?** Softly today: the recency term decays old facts down the ranking. Hard forgetting (`stale_after` expiry, human-verified staging, real deletes, contradiction-replaces-old) has schema columns ready and is deliberately deferred — assignment kept simple, sophistication goes to the capstone.

Proven live: facts written by a **local process** days earlier are recalled by the **deployed app** in a fresh browser session — recall across a process restart *and* an environment change. Code: `write_gate.py`, `memory_store.py`, UI in the "Pamięć (Week 5)" tab.

## Try it (2 minutes)

1. **Ask** the default question (*"How many remote days are allowed?"*) → answer with citation `POL-101`.
2. Ask the trap (*"What is the parental leave policy?"*) → explicit refusal, zero citations — even though retrieval returns topically similar fragments.
3. **Memory tab** → "Odczytaj z pamięci" with *"kto jest moim głównym klientem?"* → a fact written in an earlier session comes back.
4. **Evals tab** → run the suite → score, before/after table, known gaps.

Free-tier notes: a cold start takes up to ~1 min (measured backend wake: 32–41 s). The RAG index lives on the instance's ephemeral disk and is re-seeded before demos (`ingest_corpus.py`); the memory store in Neon persists on its own. The Agent tab computes on the UI instance's local (empty) index — its proof of work is the Week 3 submission; the deployed proof paths are Ask, Memory and Evals.

## Origin: this started as the Week 1 course starter

The repo began as the bootcamp's minimal `/ask` starter and grew week by week: structured output + validation retry (W1), bilingual RAG with citations and refusal (W2), the agent loop ×2 engines (W3), tracing + the eval suite (W4), durable memory + deployment of the full stack (W5). The `stages/` files still show the Week 1 build-up of the endpoint in three steps.

Run locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY (and DATABASE_URL for the memory tab)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload   # terminal 1
streamlit run demo_page.py                               # terminal 2
python smoke_test.py   # no-token startup check
```
