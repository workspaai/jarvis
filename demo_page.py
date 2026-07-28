"""Streamlit UI dla API Jarvisa: /ask (RAG z cytowaniami, Week 2) + /ingest.

UI tylko woła API — cała logika RAG (chunking, retrieval, grounding) mieszka
w FastAPI. Zero sekretów po stronie strony: adres API z paska bocznego.

Run:
  streamlit run demo_page.py
"""

import json

import httpx
import streamlit as st

WORKDIR_CMD = "projekty/jarvis"
MODELS = ["gpt-4o-mini", "gpt-4o", "o3-mini"]


def build_payload(question: str, model: str, force_bad: bool) -> dict:
    return {
        "question": question,
        "model": model,
        "force_bad": force_bad,
    }


def render_curl(base_url: str, payload: dict) -> str:
    body = json.dumps(payload)
    return (
        f'curl -s -X POST {base_url.rstrip("/")}/ask '
        f'-H "Content-Type: application/json" '
        f"-d '{body}'"
    )


def call_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict | str]:
    try:
        if method == "POST":
            response = httpx.post(url, json=payload, timeout=120.0)
        else:
            response = httpx.get(url, timeout=120.0)

        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {url}. Start the API server first."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def render_attempts(data: dict | str) -> None:
    if not isinstance(data, dict):
        return

    attempts = data.get("attempts", [])
    if not attempts:
        return

    st.markdown("### Attempts")
    for attempt in attempts:
        status = "passed" if attempt.get("ok") else "failed"
        title = f"Attempt {attempt.get('attempt')}: {attempt.get('step')} ({status})"
        with st.expander(title, expanded=True):
            st.write(attempt.get("message"))
            if attempt.get("raw_output"):
                st.markdown("**Raw model output**")
                st.code(attempt["raw_output"], language="json")
            if attempt.get("validation_error"):
                st.markdown("**Validation error**")
                st.code(attempt["validation_error"], language="text")


def render_response_summary(data: dict | str) -> None:
    if not isinstance(data, dict) or "error" in data:
        return

    answer = data.get("answer")
    if isinstance(answer, dict):
        st.markdown("### Odpowiedź")
        if answer.get("i_dont_know"):
            # Odmowa ma być czytelna od razu, nie ukryta w polu boolean.
            st.warning(f"**Brak w dokumentach** — {answer.get('answer', '')}")
        else:
            st.success(answer.get("answer", ""))

        source = answer.get("source") or []
        if source:
            st.markdown("**Źródła (cytowane):** " + " ".join(f"`{s}`" for s in source))
        else:
            st.markdown("**Źródła (cytowane):** _brak — odpowiedź nie cytuje żadnego dokumentu_")
        st.caption(
            f"confidence: {answer.get('confidence')} | "
            f"sources_needed: {answer.get('sources_needed')} | "
            f"i_dont_know: {answer.get('i_dont_know')}"
        )

    retrieved = data.get("retrieved_chunks") or []
    if retrieved:
        st.markdown(
            "**Pobrane chunki (retrieval):** " + " ".join(f"`{c}`" for c in retrieved)
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", str(data.get("model", "-")))
    metric_cols[1].metric("Tokens", str(data.get("tokens_used", "-")))
    metric_cols[2].metric("Latency", f"{data.get('latency_ms', '-')} ms")
    metric_cols[3].metric("Cost", f"${data.get('cost_usd', '-')}")


st.set_page_config(page_title="Jarvis — demo RAG (Week 2)", layout="centered")
st.title("Jarvis: pytania do dokumentów (RAG)")
st.caption(
    "Streamlit tylko woła API — cała logika RAG (chunking, retrieval, grounding) "
    "mieszka w FastAPI. Pliki `stages/` pokazują, jak endpoint rósł krok po kroku."
)

base_url = st.sidebar.text_input("API base URL", "https://jarvis-8lpg.onrender.com")
st.sidebar.markdown("### Start the API")
st.sidebar.code(
    f"cd {WORKDIR_CMD}\n"
    "source .venv/bin/activate\n"
    "uvicorn main:app --host 127.0.0.1 --port 8000 --reload",
    language="bash",
)
st.sidebar.markdown("### Start this page")
st.sidebar.code(
    f"cd {WORKDIR_CMD}\nsource .venv/bin/activate\nstreamlit run demo_page.py",
    language="bash",
)

if st.sidebar.button("Sprawdź /health"):
    status, data = call_json("GET", f"{base_url.rstrip('/')}/health")
    st.sidebar.markdown(f"**HTTP {status}**" if status else "**Brak połączenia**")
    st.sidebar.json(data)

tab_ask, tab_ingest = st.tabs(["Zadaj pytanie (/ask)", "Dodaj dokument (/ingest)"])

with tab_ask:
    with st.form("ask_form"):
        question = st.text_area(
            "Pytanie",
            "How many remote days are allowed?",
            height=100,
        )
        model = st.selectbox("Model", MODELS, index=0)
        force_bad = st.checkbox(
            "Wymuś zepsutą pierwszą odpowiedź (demo walidacji + retry z Week 1)",
            value=False,
        )
        submitted = st.form_submit_button("Zapytaj", type="primary")

    payload = build_payload(question, model, force_bad)
    st.markdown("#### Request")
    st.code(render_curl(base_url, payload), language="bash")

    if submitted:
        with st.spinner("Wołam /ask… (uśpiony Render może się budzić do ~1 min)"):
            status, data = call_json("POST", f"{base_url.rstrip('/')}/ask", payload)
        st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
        render_response_summary(data)
        render_attempts(data)
        with st.expander("Pełny JSON odpowiedzi"):
            st.json(data)

with tab_ingest:
    with st.form("ingest_form"):
        doc_text = st.text_area(
            "Treść dokumentu",
            height=220,
            placeholder="Wklej pełny tekst dokumentu do zaindeksowania…",
        )
        document_id = st.text_input("document_id", placeholder="np. POL-101")
        doc_source = st.text_input(
            "source (opcjonalne)", placeholder="np. doc1_handbook.txt"
        )
        ingest_submitted = st.form_submit_button("Zaindeksuj", type="primary")

    if ingest_submitted:
        ingest_payload = {
            "text": doc_text,
            "document_id": document_id,
            "source": doc_source,
        }
        with st.spinner("Wołam /ingest…"):
            status, data = call_json(
                "POST", f"{base_url.rstrip('/')}/ingest", ingest_payload
            )
        st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
        if status == 200 and isinstance(data, dict):
            st.success(
                f"Zaindeksowano **{data.get('document_id')}** → "
                f"{data.get('chunks_indexed')} chunk(ów), status: {data.get('status')}"
            )
        st.json(data)
