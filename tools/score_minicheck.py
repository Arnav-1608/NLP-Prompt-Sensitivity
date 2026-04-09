import os
import json
import csv
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

SUMMARIES_PATH = ".tmp/raw_summaries.jsonl"
OUTPUT_PATH = ".tmp/scores/factual_scores.csv"

API_URL = "https://api-inference.huggingface.co/models/cross-encoder/nli-deberta-v3-base"
TIMEOUT = 120
COURTESY_SLEEP = 0.5
COLD_START_WAIT = 30


def query_minicheck(document, summary, headers):
    # NLI: premise = source document, hypothesis = generated summary
    payload = {
        "inputs": {
            "premise": document,
            "hypothesis": summary,
        }
    }
    for attempt in range(3):
        response = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
        if response.status_code == 503:
            body = response.json()
            if "loading" in str(body).lower():
                print(f"  Model loading — waiting {COLD_START_WAIT}s...", file=sys.stderr)
                time.sleep(COLD_START_WAIT)
                continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("NLI API failed after 3 attempts (model loading)")


def extract_factual_score(api_response):
    # HF NLI API returns [{"label": "ENTAILMENT", "score": ...}, ...]
    if isinstance(api_response, list) and len(api_response) > 0:
        inner = api_response[0] if isinstance(api_response[0], list) else api_response
        for item in inner:
            if item.get("label", "").upper() == "ENTAILMENT":
                return item["score"]
        # Fallback: return score of whichever label has highest score
        return max(inner, key=lambda x: x["score"])["score"]
    return None


def main():
    if not os.path.exists(SUMMARIES_PATH):
        raise FileNotFoundError(
            f"{SUMMARIES_PATH} not found. Run run_inference.py first."
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set. Add it to your .env file.")

    headers = {"Authorization": f"Bearer {hf_token}"}

    os.makedirs(".tmp/scores", exist_ok=True)

    records = []
    with open(SUMMARIES_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Scoring {len(records)} rows for factual consistency via MiniCheck...")
    print("Note: First call may take ~30s due to API cold start.")

    rows = []
    for i, record in enumerate(records):
        try:
            result = query_minicheck(
                document=record["source_document"],
                summary=record["generated_summary"],
                headers=headers,
            )
            score = extract_factual_score(result)
        except Exception as e:
            print(f"  ERROR on {record['row_id']}: {e}", file=sys.stderr)
            score = None

        rows.append({
            "row_id": record["row_id"],
            "factual_score": round(score, 6) if score is not None else "",
        })

        time.sleep(COURTESY_SLEEP)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(records)}")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "factual_score"])
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["factual_score"] != ""]
    print(f"Scored {len(valid)}/{len(rows)} rows successfully. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
