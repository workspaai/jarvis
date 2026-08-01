"""Week 3, Krok 3 — TA SAMA pętla agenta, ale zapisana jako LangGraph StateGraph.

W3 L3: „graf ma być czytelniejszym zapisem tego, co już zbudowałem" — dlatego
reużywamy DOKŁADNIE tego samego narzędzia, system promptu, modelu i limitu co
w `agent_raw.py`. Zmienia się MASZYNERIA (kto prowadzi pętlę), nie ZACHOWANIE.

Trzy klocki z lekcji:
  STATE  — typowany słownik: wiadomości (reducer: dopisywanie), liczniki, wynik
  NODES  — `think` (model decyduje) + `act` (moj kod wykonuje narzędzie)
  EDGES  — krawędź WARUNKOWA: wywołanie narzędzia -> `act`, inaczej -> END
           (+ gałąź `fail_closed` przy przekroczeniu limitu iteracji)

Checkpointer (MemorySaver) zapisuje stan między krokami — fundament pod
human-in-the-loop (interrupt) jako ewentualny stretch zadania.

Uruchomienie:
  .venv/bin/python agent_graph.py "Twoje pytanie"
  .venv/bin/python agent_graph.py --diagram    # wypisz diagram Mermaid
"""

import operator
import sys
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# TE SAME klocki co w surowej pętli — dowód, że zmieniła się tylko maszyneria.
from agent_raw import (
    AGENT_SYSTEM_PROMPT,
    MAX_ITERATIONS,
    SEARCH_TOOL,
    _observe_summary,
    log_step,
    parse_sources,
    search_documents,
)
from main import DEFAULT_MODEL, compute_cost_usd, get_client, usage_counts


class AgentState(TypedDict):
    """STATE: obiekt wędrujący po krawędziach; każdy węzeł czyta i aktualizuje."""

    messages: Annotated[list, operator.add]  # reducer: nowe wiadomości dopisywane
    trace: Annotated[list, operator.add]  # ślad Think/Act/Observe dla UI
    iterations: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    answer: str


def think(state: AgentState) -> dict:
    """NODE 1 (Think): model decyduje — zawołać narzędzie czy odpowiedzieć."""
    iteration = state["iterations"] + 1
    completion = get_client().chat.completions.create(
        model=DEFAULT_MODEL,
        messages=state["messages"],
        tools=[SEARCH_TOOL],
        temperature=0,
    )
    _, prompt_tokens, completion_tokens = usage_counts(completion)
    msg = completion.choices[0].message

    step: list[dict] = []
    if msg.tool_calls:
        log_step(step, iteration, "THINK", "model zdecydował: wywołać narzędzie")
        answer = ""
    else:
        log_step(step, iteration, "THINK", "model zdecydował: ODPOWIEDŹ KOŃCOWA")
        answer = (msg.content or "").strip()

    return {
        "messages": [msg],
        "trace": step,
        "iterations": iteration,
        "prompt_tokens": state["prompt_tokens"] + prompt_tokens,
        "completion_tokens": state["completion_tokens"] + completion_tokens,
        "answer": answer,
    }


def act(state: AgentState) -> dict:
    """NODE 2 (Act + Observe): MÓJ kod wykonuje narzędzie, wynik wraca jako obserwacja."""
    last = state["messages"][-1]
    iteration = state["iterations"]
    new_messages = []
    step: list[dict] = []
    executed = 0

    for tc in last.tool_calls:
        if tc.function.name != "search_documents":
            result = f"BŁĄD: nieznane narzędzie '{tc.function.name}'."
            log_step(step, iteration, "ACT", f"ODRZUCONE: {tc.function.name}")
        else:
            import json

            args = json.loads(tc.function.arguments)
            query = args.get("query", "")
            executed += 1
            log_step(step, iteration, "ACT", f'search_documents(query="{query}")')
            result = search_documents(query)
        log_step(step, iteration, "OBSERVE", _observe_summary(result))
        new_messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result}
        )

    return {
        "messages": new_messages,
        "trace": step,
        "tool_calls": state["tool_calls"] + executed,
    }


def fail_closed(state: AgentState) -> dict:
    """Limit iteracji przekroczony — uczciwa odmowa zamiast zmyślonej odpowiedzi."""
    step: list[dict] = []
    log_step(
        step,
        state["iterations"],
        "STOP",
        f"FAIL CLOSED — przekroczony limit {MAX_ITERATIONS} iteracji",
    )
    return {
        "trace": step,
        "answer": (
            "Nie wiem — agent przekroczył limit iteracji "
            f"({MAX_ITERATIONS}) i przerywam zamiast zgadywać. Źródła: [brak]."
        ),
    }


def route(state: AgentState) -> str:
    """EDGE warunkowa: narzędzie -> act, limit -> fail_closed, inaczej -> END."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if state["iterations"] >= MAX_ITERATIONS:
            return "fail_closed"
        return "act"
    return END


def build_graph():
    """Buduje i kompiluje graf z checkpointerem (fundament pod HITL)."""
    builder = StateGraph(AgentState)
    builder.add_node("think", think)
    builder.add_node("act", act)
    builder.add_node("fail_closed", fail_closed)

    builder.add_edge(START, "think")
    builder.add_conditional_edges(
        "think", route, {"act": "act", "fail_closed": "fail_closed", END: END}
    )
    builder.add_edge("act", "think")  # krawędź powrotna = pętla agenta
    builder.add_edge("fail_closed", END)

    return builder.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(question: str, thread_id: str | None = None) -> dict:
    """Uruchamia graf na jednym pytaniu; log i statystyki jak w surowej pętli.

    `thread_id` domyślnie NOWY dla każdego przebiegu. Checkpointer zapisuje stan
    per wątek, a pola `messages` i `trace` mają reducer `operator.add` — użycie
    tego samego wątku dwa razy DOKLEJAŁO stan poprzedniego pytania (podwójny
    ślad, zawyżone tokeny, zanieczyszczona odpowiedź). Stały `thread_id`
    podajemy tylko świadomie: do kontynuacji rozmowy albo wznowienia po HITL.
    """
    if thread_id is None:
        thread_id = f"run-{uuid.uuid4().hex[:8]}"
    started = time.time()
    print(f"\n{'=' * 72}\nPRZEBIEG AGENTA (LangGraph) — pytanie: {question}\n{'=' * 72}")

    final = get_graph().invoke(
        {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "trace": [],
            "iterations": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "answer": "",
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    cost = compute_cost_usd(
        DEFAULT_MODEL, final["prompt_tokens"], final["completion_tokens"]
    )
    tokens = final["prompt_tokens"] + final["completion_tokens"]
    elapsed_ms = int((time.time() - started) * 1000)
    print(f"\nODPOWIEDŹ: {final['answer']}")
    print(
        f"-- statystyki: iteracje={final['iterations']}, "
        f"wywołania narzędzia={final['tool_calls']}, tokeny={tokens}, "
        f"koszt=${cost:.6f}, czas={elapsed_ms / 1000:.1f}s"
    )
    sources, refused = parse_sources(final["answer"])
    return {
        "answer": final["answer"],
        "iterations": final["iterations"],
        "tool_calls": final["tool_calls"],
        "cost_usd": cost,
        "tokens": tokens,
        "latency_ms": elapsed_ms,
        "sources": sources,
        "refused": refused,
        "trace": final["trace"],
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagram":
        print(get_graph().get_graph().draw_mermaid())
    elif len(sys.argv) > 1:
        run_agent(" ".join(sys.argv[1:]))
    else:
        print('Użycie: python agent_graph.py "Twoje pytanie" | --diagram')
        raise SystemExit(1)
