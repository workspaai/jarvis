"""Stage 1: smallest useful `/ask` endpoint.

Run:
  uvicorn stages.stage_1_bare_ask:app --host 127.0.0.1 --port 8000 --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR.parent / ".env")
load_dotenv(THIS_DIR.parent.parent / ".env")

app = FastAPI(title="Stage 1: Bare Ask")
client = OpenAI()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    tokens_used: int


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": body.question}],
    )

    answer = completion.choices[0].message.content or ""
    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return AskResponse(answer=answer, tokens_used=tokens_used)
