import os
import json
import csv

from rouge_score import rouge_scorer

SUMMARIES_PATH = ".tmp/raw_summaries.jsonl"
OUTPUT_PATH = ".tmp/scores/rouge_scores.csv"


def main():
    if not os.path.exists(SUMMARIES_PATH):
        raise FileNotFoundError(
            f"{SUMMARIES_PATH} not found. Run run_inference.py first."
        )

    os.makedirs(".tmp/scores", exist_ok=True)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    with open(SUMMARIES_PATH) as f:
        for line in f:
            record = json.loads(line)
            scores = scorer.score(
                target=record["reference_summary"],
                prediction=record["generated_summary"],
            )
            rows.append({
                "row_id": record["row_id"],
                "rouge1": round(scores["rouge1"].fmeasure, 6),
                "rouge2": round(scores["rouge2"].fmeasure, 6),
                "rougeL": round(scores["rougeL"].fmeasure, 6),
            })

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "rouge1", "rouge2", "rougeL"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {len(rows)} rows. Output: {OUTPUT_PATH}")
    if rows:
        avg_r1 = sum(r["rouge1"] for r in rows) / len(rows)
        print(f"Mean ROUGE-1: {avg_r1:.4f} (expected range 0.2–0.6 for XSum)")


if __name__ == "__main__":
    main()
