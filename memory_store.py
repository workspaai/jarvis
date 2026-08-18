"""Trwały store pamięci SEMANTYCZNEJ Jarvisa (Postgres + pgvector na Neon).

Zadanie Week 5 „Give your capstone memory": fakty przeżywają restart procesu
i dają się odczytać w nowej sesji. Ten moduł to sama warstwa bazy danych —
liczenie embeddingu i bramka zapisu wchodzą w Kroku 2 (ścieżka zapisu).

Wymaga zmiennej środowiskowej DATABASE_URL (connection string z Neona) w .env.
"""

import os

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")  # connection string z Neona (.env)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"
EMBED_DIM = 1536  # text-embedding-3-small

_embeddings: "OpenAIEmbeddings | None" = None


def embed_text(text: str) -> list[float]:
    """Wektor 1536-wymiarowy dla tekstu — TEN SAM model embeddingów co korpus RAG."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings.embed_query(text)


def _require_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError(
            "Brak DATABASE_URL w .env — wklej connection string z Neona "
            "(postgresql://...neon.tech/...?sslmode=require)."
        )
    return DATABASE_URL


def init_schema() -> None:
    """Tworzy rozszerzenie pgvector i tabelę semantic_memory (idempotentnie).

    DDL nie wymaga register_vector — potrzebuje tylko, by extension istniało.
    """
    with psycopg.connect(_require_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id              bigserial     PRIMARY KEY,
                    subject         text          NOT NULL,        -- o KIM/CZYM jest fakt (encja)
                    fact            text          NOT NULL,        -- fakt, jedno zdanie
                    embedding       vector({EMBED_DIM}),           -- wektor treści faktu (3-small)
                    confidence      real          NOT NULL DEFAULT 0.5,
                    source_event_id text,                          -- provenance: id trace'a (traces.jsonl)
                    verified_by     text,                          -- NULL = poczekalnia; "human:olek" = potwierdzone (Path B)
                    stale_after     date,                          -- kiedy fakt stęchły (Path B)
                    created_at      timestamptz   NOT NULL DEFAULT now(),
                    updated_at      timestamptz   NOT NULL DEFAULT now()
                );
                """
            )
        conn.commit()


def _connect() -> "psycopg.Connection":
    """Połączenie z zarejestrowanym typem vector (do insert/select). Wymaga istniejącego extension."""
    conn = psycopg.connect(_require_url())
    register_vector(conn)
    return conn


def insert_fact(
    subject: str,
    fact: str,
    embedding: list[float],
    confidence: float = 0.6,
    source_event_id: str | None = None,
) -> int:
    """Zapisuje jeden fakt. `embedding` = lista floatów długości EMBED_DIM. Zwraca id wpisu."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO semantic_memory (subject, fact, embedding, confidence, source_event_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (subject, fact, embedding, confidence, source_event_id),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
    return new_id


def get_facts(subject: str | None = None) -> list[tuple]:
    """Prosty odczyt faktów (opcjonalnie po subject) — do testu Kroku 1.

    Hybrydowy odczyt wektorowy (podobieństwo + świeżość) dokładamy w Kroku 3.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            if subject:
                cur.execute(
                    "SELECT id, subject, fact, confidence, created_at "
                    "FROM semantic_memory WHERE subject = %s ORDER BY created_at DESC;",
                    (subject,),
                )
            else:
                cur.execute(
                    "SELECT id, subject, fact, confidence, created_at "
                    "FROM semantic_memory ORDER BY created_at DESC;"
                )
            return cur.fetchall()


def search_facts(query_embedding: list[float], limit: int = 5) -> list[tuple]:
    """Hybrydowy odczyt: 0.7 * podobieństwo + 0.3 * świeżość (półokres ~30 dni).

    Bez twardego filtra `subject` (jeden użytkownik, mało faktów). Zwraca krotki
    (id, subject, fact, confidence, score) posortowane malejąco po score.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, subject, fact, confidence,
                       0.7 * (1 - (embedding <=> %s::vector))
                     + 0.3 * exp(-extract(epoch FROM (now() - updated_at)) / 2592000.0) AS score
                FROM semantic_memory
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
                LIMIT %s;
                """,
                (query_embedding, limit),
            )
            return cur.fetchall()


def recall_facts(query_text: str, limit: int = 5) -> list[tuple]:
    """Wygodne: liczy embedding zapytania i zwraca istotne fakty (hybrydowo)."""
    return search_facts(embed_text(query_text), limit)


if __name__ == "__main__":
    # Szybki test ręczny Kroku 1 (po ustawieniu DATABASE_URL):
    #   python memory_store.py
    init_schema()
    print("[OK] init_schema — extension + tabela gotowe")
    test_vec = [0.0] * EMBED_DIM
    fid = insert_fact("test", "To jest testowy fakt Kroku 1.", test_vec, source_event_id="krok1-test")
    print(f"[OK] insert_fact — id={fid}")
    rows = get_facts("test")
    print(f"[OK] get_facts('test') — {len(rows)} wiersz(y): {[(r[0], r[2]) for r in rows]}")
