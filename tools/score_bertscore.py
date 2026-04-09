import os
import json
import csv

SUMMARIES_PATH = ".tmp/raw_summaries.jsonl"
OUTPUT_PATH = ".tmp/scores/bertscore_scores.csv"


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

    candidates = [r["generated_summary"] for r in records]
    references = [r["reference_summary"] for r in records]

    print("Note: First run will download ~1.3GB model. This is normal and cached after.")
    print(f"Scoring {len(records)} rows with BERTScore (distilbert-base-uncased, CPU)...")

    from bert_score import score as bert_score
    _, _, F1 = bert_score(
        cands=candidates,
        refs=references,
        model_type="distilbert-base-uncased",
        device="cpu",
        verbose=True,
    )

    f1_scores = F1.tolist()

    rows = []
    for record, f1 in zip(records, f1_scores):
        rows.append({
            "row_id": record["row_id"],
            "bertscore_f1": round(f1, 6),
        })

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "bertscore_f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {len(rows)} rows. Output: {OUTPUT_PATH}")
    if rows:
        avg = sum(r["bertscore_f1"] for r in rows) / len(rows)
        print(f"Mean BERTScore F1: {avg:.4f} (expected range 0.8–0.95 for distilbert)")


if __name__ == "__main__":
    main()
