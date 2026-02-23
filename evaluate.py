"""
Model Evaluation — Precision, Recall, F1
Uses a small hand-labelled test set to measure NER accuracy.
Run: python evaluate.py
"""

from ner_pipeline import extract_entities


# ── Hand-labelled ground truth ──────────────────────────────────────────────
TEST_CASES = [
    {
        "text": "Apple Inc. raised $2.5 billion on January 10, 2024.",
        "expected": {
            "amounts":       ["$2.5 billion"],
            "dates":         ["January 10, 2024"],
            "organizations": ["Apple Inc."],
        },
    },
    {
        "text": "On 03/22/2023, Bank of America issued bonds worth $500 million.",
        "expected": {
            "amounts":       ["$500 million"],
            "dates":         ["03/22/2023"],
            "organizations": ["Bank of America"],
        },
    },
    {
        "text": "Microsoft acquired Activision Blizzard for $68.7 billion, closing December 31, 2023.",
        "expected": {
            "amounts":       ["$68.7 billion"],
            "dates":         ["December 31, 2023"],
            "organizations": ["Microsoft", "Activision Blizzard"],
        },
    },
    {
        "text": "On 06/15/2024, Citigroup announced a $1.5 billion share buyback program.",
        "expected": {
            "amounts":       ["$1.5 billion"],
            "dates":         ["06/15/2024"],
            "organizations": ["Citigroup"],
        },
    },
    {
        "text": "Wells Fargo reported net income of $4.9 billion for the quarter ending March 31, 2024.",
        "expected": {
            "amounts":       ["$4.9 billion"],
            "dates":         ["March 31, 2024"],
            "organizations": ["Wells Fargo"],
        },
    },
]


def evaluate(test_cases):
    entity_types = ["amounts", "dates", "organizations"]

    print("=" * 60)
    print(f"{'Entity Type':<16} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print("=" * 60)

    per_type_tp = {t: 0 for t in entity_types}
    per_type_fp = {t: 0 for t in entity_types}
    per_type_fn = {t: 0 for t in entity_types}

    for case in test_cases:
        predicted = extract_entities(case["text"])
        expected  = case["expected"]

        for etype in entity_types:
            pred_set = set(v.lower().strip() for v in predicted.get(etype, []))
            true_set = set(v.lower().strip() for v in expected.get(etype, []))

            per_type_tp[etype] += len(pred_set & true_set)
            per_type_fp[etype] += len(pred_set - true_set)
            per_type_fn[etype] += len(true_set - pred_set)

    overall_tp = overall_fp = overall_fn = 0

    for etype in entity_types:
        tp = per_type_tp[etype]
        fp = per_type_fp[etype]
        fn = per_type_fn[etype]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        print(f"{etype:<16} {precision:>9.2%} {recall:>9.2%} {f1:>9.2%}")
        overall_tp += tp; overall_fp += fp; overall_fn += fn

    print("-" * 60)
    p  = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0
    r  = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print(f"{'OVERALL':<16} {p:>9.2%} {r:>9.2%} {f1:>9.2%}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate(TEST_CASES)