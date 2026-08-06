"""Week 4, Krok 1 — trwały zapis trace'ów do JSONL (litera T z pętli TRACE).

Jeden przebieg = jedna linia JSON w traces/traces.jsonl. Pola wg przewodnika
zadania W4 (user_input, retrieved_context, tool_calls, assistant_output) plus
metadane: silnik, model, wersja promptu, iteracje, tokeny, koszt, latencja
i pełny ślad kroków.

Katalog traces/ jest POZA gitem: trace'y zawierają TREŚĆ dokumentów korpusu,
a dokumenty nigdy nie idą do repo (decyzja v1 w jarvis-plan.md).
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TRACES_PATH = Path(os.getenv("TRACES_PATH") or THIS_DIR / "traces" / "traces.jsonl")


def prompt_version(prompt_text: str) -> str:
    """Krótki odcisk promptu — trace mówi, KTÓRA wersja promptu go wyprodukowała."""
    return hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:8]


def save_trace(record: dict) -> Path:
    """Dopisuje jeden przebieg jako jedną linię JSONL; dokłada trace_id i czas."""
    record = {
        "trace_id": uuid.uuid4().hex[:12],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **record,
    }
    TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return TRACES_PATH


def save_agent_trace(
    engine: str,
    question: str,
    model: str,
    prompt_ver: str,
    result: dict,
    tool_log: list,
) -> Path:
    """Mapuje wynik `run_agent` (surowa pętla / graf) na pola trace'u z przewodnika."""
    return save_trace(
        {
            "engine": engine,
            "user_input": question,
            "model": model,
            "prompt_version": prompt_ver,
            "tool_calls": tool_log,
            # Pełne fragmenty zwrócone przez narzędzie — to, co model FAKTYCZNIE widział.
            "retrieved_context": [
                t["result"] for t in tool_log if t.get("tool") == "search_documents"
            ],
            "assistant_output": result["answer"],
            "sources": result["sources"],
            "refused": result["refused"],
            "iterations": result["iterations"],
            "tool_calls_count": result["tool_calls"],
            "tokens": result["tokens"],
            "cost_usd": result["cost_usd"],
            "latency_ms": result["latency_ms"],
            "trace_steps": result["trace"],
        }
    )
