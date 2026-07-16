"""Stage 2: ask for a predictable structured answer.

Run:
  uvicorn stages.stage_2_structured_output:app --host 127.0.0.1 --port 8000 --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR.parent / ".env")
load_dotenv(THIS_DIR.parent.parent / ".env")

app = FastAPI(title="Stage 2: Structured Output")
client = OpenAI()


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": body.question}],
        response_format=Answer,
    )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("Model returned no parseable structured output")

    tokens_used = completion.usage.total_tokens if completion.usage else 0
    return AskResponse(answer=answer, tokens_used=tokens_used)
