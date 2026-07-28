"""Batch ingest korpusu do żywego (lub lokalnego) API Jarvisa.

Po każdym deployu na Render baza Chroma jest PUSTA (efemeryczny dysk) —
ten skrypt jedną komendą odtwarza cały korpus:

  .venv/bin/python ingest_corpus.py https://jarvis-8lpg.onrender.com
  .venv/bin/python ingest_corpus.py                # domyślnie lokalny serwer

Katalog korpusu można nadpisać zmienną CORPUS_DIR (domyślnie dane/northwind
w katalogu AI-Bootcamp — POZA repo, zgodnie z decyzją "dokumenty nigdy do gita").
Każdy plik .txt musi zawierać linię "Document ID: <ID>" — stąd bierzemy document_id.
"""

import os
import re
import sys
from pathlib import Path

import httpx

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_DIR = THIS_DIR.parent.parent / "dane" / "northwind"

DOCUMENT_ID_RE = re.compile(r"^Document ID:\s*(\S+)", re.MULTILINE)


def main() -> int:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    corpus_dir = Path(os.getenv("CORPUS_DIR") or DEFAULT_CORPUS_DIR)

    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        print(f"BŁĄD: brak plików .txt w {corpus_dir}")
        return 1

    print(f"Korpus: {corpus_dir} ({len(files)} plików) -> {base_url}/ingest")
    failures = 0
    # Cold start na Renderze (darmowy plan usypia serwis) potrafi trwać ~1 min,
    # stąd długi timeout.
    with httpx.Client(timeout=120) as client:
        for path in files:
            text = path.read_text(encoding="utf-8")
            match = DOCUMENT_ID_RE.search(text)
            if not match:
                print(f"  {path.name}: POMINIĘTY — brak linii 'Document ID:'")
                failures += 1
                continue
            document_id = match.group(1)
            try:
                response = client.post(
                    f"{base_url}/ingest",
                    json={"text": text, "document_id": document_id, "source": path.name},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  {path.name}: BŁĄD — {exc}")
                failures += 1
                continue
            data = response.json()
            print(
                f"  {path.name}: {data['document_id']} -> "
                f"{data['chunks_indexed']} chunk(ów), status={data['status']}"
            )

    if failures:
        print(f"UWAGA: {failures} plik(ów) nie weszło.")
        return 1
    print("Cały korpus zaindeksowany.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
