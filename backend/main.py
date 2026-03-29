from fastapi import FastAPI
from pydantic import BaseModel
import time
from transformers import pipeline

app = FastAPI(title="qa-local backend", version="1.0")

MODEL_ID = "deepset/bert-base-uncased-squad2"
qa_pipe = None


def get_pipe():
    global qa_pipe
    if qa_pipe is None:
        qa_pipe = pipeline("question-answering", model=MODEL_ID)
    return qa_pipe


class QARequest(BaseModel):
    context: str
    question: str


@app.get("/")
def root():
    return {
        "message": "qa-local backend up",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/qa")
def qa(req: QARequest):
    context = (req.context or "").strip()
    question = (req.question or "").strip()

    if not context:
        return {
            "answer": "Please paste some context/passage first.",
            "meta": "bad_request"
        }

    if not question:
        return {
            "answer": "Please type a question.",
            "meta": "bad_request"
        }

    if len(context) > 8000:
        context = context[:8000] + "..."

    t0 = time.time()

    try:
        pipe = get_pipe()
        out = pipe(question=question, context=context)
    except Exception as e:
        return {
            "answer": f"Local model error: {e}",
            "meta": "local_error"
        }

    dt = time.time() - t0

    if isinstance(out, dict):
        answer = out.get("answer") or "(No answer found in the provided context.)"
        score = out.get("score")
    else:
        answer = str(out) if out else "(No answer found in the provided context.)"
        score = None

    meta = f"Mode: Local | Model: {MODEL_ID} | Time: {dt:.2f}s"
    if score is not None:
        meta += f" | Score: {score:.3f}"

    return {
        "answer": answer,
        "meta": meta
    }
