"""Week 1 v2 demo API: one compact `/ask` endpoint for the intro class.

Run:
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

app = FastAPI(title="Week 1 v2 /ask Demo")
_client: OpenAI | None = None

ModelName = Literal["gpt-4o-mini", "gpt-4o", "o3-mini"]
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}

# Domyślny model ze zmiennej środowiskowej, żeby zmiana (np. na PC) nie wymagała
# zmiany kodu; "or" łapie też pustą wartość skopiowaną z .env.example.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or "gpt-4o-mini"
if DEFAULT_MODEL not in MODEL_PRICES_PER_1K:
    raise RuntimeError(
        f"Nieznany DEFAULT_MODEL={DEFAULT_MODEL!r} — dozwolone: {sorted(MODEL_PRICES_PER_1K)}"
    )

SYSTEM_PROMPT = (
    "Jesteś Jarvis — prywatny asystent Olka.\n"
    "Zasady, których zawsze przestrzegasz:\n"
    "1. Odpowiadasz po polsku.\n"
    "2. Opierasz się wyłącznie na tym, co wiesz z pewnością.\n"
    "3. Jeśli nie masz podstaw do rzetelnej odpowiedzi (np. pytanie dotyczy prywatnych "
    "dokumentów, faktur czy spraw, do których nie masz dostępu), piszesz wprost, że nie "
    "wiesz, i ustawiasz i_dont_know=true. Nigdy nie zmyślasz.\n"
    "4. Odpowiadasz zwięźle — najlepiej w 1–3 zdaniach."
)


class Answer(BaseModel):
    """Kształt odpowiedzi, który za każdym razem dostaje klient endpointu."""

    answer: str = Field(
        min_length=1,
        description=(
            "Zwięzła odpowiedź po polsku, najlepiej 1–3 zdania. Gdy nie masz podstaw "
            "do rzetelnej odpowiedzi, napisz wprost „nie wiem” i jednym zdaniem dlaczego."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Jak bardzo jesteś pewny odpowiedzi: liczba od 0.0 (czyste zgadywanie) "
            "do 1.0 (pełna pewność). Przy i_dont_know=true daj wartość bliską 0.0."
        ),
    )
    sources_needed: bool = Field(
        description=(
            "true, gdy rzetelna odpowiedź wymagałaby zajrzenia do zewnętrznych źródeł "
            "lub dokumentów, których nie masz w tej rozmowie; inaczej false."
        )
    )
    i_dont_know: bool = Field(
        description=(
            "true, gdy nie masz podstaw do rzetelnej odpowiedzi i uczciwie to przyznajesz "
            "zamiast zmyślać; false, gdy odpowiadasz merytorycznie."
        )
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    model: ModelName | None = None
    force_bad: bool = False


class AttemptResult(BaseModel):
    attempt: int
    step: str
    ok: bool
    message: str
    raw_output: str | None = None
    validation_error: str | None = None


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    attempts: list[AttemptResult]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_per_1k, output_per_1k = MODEL_PRICES_PER_1K.get(
        model, MODEL_PRICES_PER_1K[DEFAULT_MODEL]
    )
    return (prompt_tokens / 1000 * input_per_1k) + (
        completion_tokens / 1000 * output_per_1k
    )


def usage_counts(completion) -> tuple[int, int, int]:
    usage = completion.usage
    if usage is None:
        return 0, 0, 0
    return usage.total_tokens, usage.prompt_tokens, usage.completion_tokens


def call_structured_model(question: str, model: ModelName) -> tuple[Answer, int, int, int]:
    completion = get_client().chat.completions.parse(
        model=model,
        # Kolejność celowa (prompt caching): stała część pierwsza, zmienna ostatnia.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return parsed, total_tokens, prompt_tokens, completion_tokens


def call_malformed_json_once(question: str, model: ModelName) -> tuple[str, int, int, int]:
    """Demo-only path: force one malformed response so students can see retry."""

    completion = get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reply with ONLY JSON using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' instead of a number."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    total_tokens, prompt_tokens, completion_tokens = usage_counts(completion)
    return raw, total_tokens, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    model = body.model or DEFAULT_MODEL
    last_error: str | None = None
    attempts: list[AttemptResult] = []
    total_tokens_used = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    start = time.perf_counter()

    for attempt in range(2):
        try:
            if body.force_bad and attempt == 0:
                raw, tokens_used, prompt_tokens, completion_tokens = call_malformed_json_once(
                    body.question, model
                )
                total_tokens_used += tokens_used
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

                try:
                    answer = Answer.model_validate_json(raw)
                except ValidationError as exc:
                    last_error = str(exc)
                    attempts.append(
                        AttemptResult(
                            attempt=attempt + 1,
                            step="forced_bad_json",
                            ok=False,
                            message="Validation failed, so the endpoint retries with structured output.",
                            raw_output=raw,
                            validation_error=str(exc),
                        )
                    )
                    continue

                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="forced_bad_json",
                        ok=True,
                        message="Unexpectedly passed validation.",
                        raw_output=raw,
                    )
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_structured_model(
                    body.question, model
                )
                total_tokens_used += tokens_used
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                attempts.append(
                    AttemptResult(
                        attempt=attempt + 1,
                        step="structured_output",
                        ok=True,
                        message="Structured output matched the Answer schema.",
                    )
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(
                model, total_prompt_tokens, total_completion_tokens
            )
            return AskResponse(
                answer=answer,
                tokens_used=total_tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                attempts=attempts,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            attempts.append(
                AttemptResult(
                    attempt=attempt + 1,
                    step="structured_output",
                    ok=False,
                    message="Structured output failed validation.",
                    validation_error=str(exc),
                )
            )

    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
