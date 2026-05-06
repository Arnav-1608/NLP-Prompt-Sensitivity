import os
import json
import csv
import sys

from sentence_transformers.cross_encoder import CrossEncoder

SUMMARIES_PATH = ".tmp/raw_summaries.jsonl"
OUTPUT_PATH = ".tmp/scores/factual_scores.csv"
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# id2label for this model: 0=contradiction, 1=entailment, 2=neutral
ENTAILMENT_IDX = 1
BATCH_SIZE = 64


def load_model():
    print(f"Loading {MODEL_NAME} (downloads ~700 MB on first run, cached after)...")
    # Force CPU: long documents exceed GPU memory budget on MPS (Apple Silicon).
    return CrossEncoder(MODEL_NAME, max_length=512, device="cpu")


def main():
    if not os.path.exists(SUMMARIES_PATH):
        raise FileNotFoundError(
            f"{SUMMARIES_PATH} not found. Run run_inference.py first."
        )

    os.makedirs(".tmp/scores", exist_ok=True)

    records = []
    with open(SUMMARIES_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    model = load_model()
    print(f"Scoring {len(records)} rows for factual consistency (local NLI)...")

    pairs = [[r["source_document"], r["generated_summary"]] for r in records]
    scores_out = [None] * len(records)

    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]
        try:
            batch_scores = model.predict(batch, apply_softmax=True)
            for j, s in enumerate(batch_scores):
                scores_out[start + j] = float(s[ENTAILMENT_IDX])
        except Exception as e:
            print(f"  ERROR on batch {start}–{start + len(batch) - 1}: {e}", file=sys.stderr)
        print(f"  Progress: {min(start + BATCH_SIZE, len(records))}/{len(records)}")

    rows = [
        {
            "row_id": r["row_id"],
            "factual_score": round(scores_out[i], 6) if scores_out[i] is not None else "",
        }
        for i, r in enumerate(records)
    ]

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "factual_score"])
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["factual_score"] != ""]
    print(f"Scored {len(valid)}/{len(rows)} rows successfully. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
