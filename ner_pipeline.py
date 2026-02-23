"""
Financial Document NER Pipeline
Extracts: amounts, dates, organization names from financial documents
Model: dslim/bert-base-NER (HuggingFace) fine-tuned for financial domain
"""

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re
from typing import List, Dict


# ── 1. Load pretrained NER model from HuggingFace ──────────────────────────
MODEL_NAME = "dslim/bert-base-NER"   # swap with fine-tuned model path later

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",   # merges sub-word tokens automatically
)


# ── 2. Post-processing helpers ──────────────────────────────────────────────
AMOUNT_PATTERN = re.compile(
    r"\$[\d,]+(?:\.\d+)?(?:\s?(?:trillion|billion|million|thousand))?", re.IGNORECASE
)
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Run BERT NER + regex post-processing.
    Returns dict with keys: amounts, dates, organizations.
    """
    ner_results = ner_pipeline(text)

    organizations, dates = [], []

    for ent in ner_results:
        label = ent["entity_group"]
        word  = ent["word"].strip()
        if label == "ORG":
            organizations.append(word)
        elif label in ("DATE", "TIME"):
            dates.append(word)

    # Supplement dates and amounts with regex (model often misses $ amounts)
    regex_amounts = AMOUNT_PATTERN.findall(text)
    regex_dates   = DATE_PATTERN.findall(text)

    # Deduplicate while preserving order
    def dedupe(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    return {
        "amounts":       dedupe(regex_amounts),
        "dates":         dedupe(dates + regex_dates),
        "organizations": dedupe(organizations),
    }


# ── 3. Quick smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "On March 15, 2024, Goldman Sachs reported quarterly revenue of $12.4 billion. "
        "JPMorgan Chase agreed to acquire First Republic Bank for $10.6 billion on 01/05/2023. "
        "BlackRock manages assets worth $9.1 trillion as of December 31, 2023."
    )
    results = extract_entities(sample)
    print("Extracted Entities:")
    for k, v in results.items():
        print(f"  {k:15s}: {v}")