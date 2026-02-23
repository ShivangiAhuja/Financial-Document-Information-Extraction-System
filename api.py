"""
FastAPI Inference Endpoint
Run:  uvicorn api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ner_pipeline import extract_entities

app = FastAPI(
    title="Financial Document NER API",
    description="Extracts amounts, dates, and organizations from financial text.",
    version="1.0.0",
)


# ── Request / Response schemas ──────────────────────────────────────────────
class DocumentRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": (
                    "Goldman Sachs reported Q2 revenue of $12.4 billion on July 15, 2024. "
                    "The deal with Morgan Stanley closes on August 1, 2024."
                )
            }
        }


class ExtractionResponse(BaseModel):
    amounts:       list[str]
    dates:         list[str]
    organizations: list[str]
    char_count:    int


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResponse)
def extract(req: DocumentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text field cannot be empty")

    entities = extract_entities(req.text)
    return ExtractionResponse(**entities, char_count=len(req.text))


@app.post("/batch_extract")
def batch_extract(requests: list[DocumentRequest]):
    """Process multiple documents in one call."""
    return [extract_entities(r.text) for r in requests]