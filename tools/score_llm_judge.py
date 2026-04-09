import os
import json
import csv
import sys
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SUMMARIES_PATH = ".tmp/raw_summaries.jsonl"
OUTPUT_PATH = ".tmp/scores/llm_judge_scores.csv"

JUDGE_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert evaluator of text summaries. "
    "Score the following summary on four dimensions, each from 1 to 5. "
    "Return ONLY a JSON object with exactly these keys: "
    "faithfulness, informativeness, fluency, conciseness. "
    "No explanation. No markdown. Just the JSON object."
)

STRICT_SYSTEM_PROMPT = (
    "You are an evaluator. Respond with ONLY a raw JSON object. "
    'Example: {"faithfulness": 4, "informativeness": 3, "fluency": 5, "conciseness": 4} '
    "Score each dimension 1-5. No other text."
)

COURTESY_SLEEP = 0.3


def make_user_prompt(source_document, generated_summary, strict=False):
    return (
        f"Article: {source_document[:1000]}\n\n"
        f"Summary: {generated_summary}\n\n"
        f"Scores:"
    )


def parse_judge_response(text):
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`").strip()
    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON object from surrounding text
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def call_judge(client, source_document, generated_summary, system_prompt):
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": make_user_prompt(source_document, generated_summary)},
        ],
        max_tokens=128,
    )
    return response.choices[0].message.content


def score_row(client, record):
    # First attempt
    try:
        text = call_judge(client, record["source_document"], record["generated_summary"], SYSTEM_PROMPT)
        time.sleep(COURTESY_SLEEP)
        result = parse_judge_response(text)
        if result:
            return result
    except Exception as e:
        print(f"  Judge call error on {record['row_id']}: {e}", file=sys.stderr)

    # Retry with stricter prompt
    try:
        text = call_judge(client, record["source_document"], record["generated_summary"], STRICT_SYSTEM_PROMPT)
        time.sleep(COURTESY_SLEEP)
        result = parse_judge_response(text)
        if result:
            return result
        print(f"  JSON parse failed twice for {record['row_id']}. Raw: {text[:100]}", file=sys.stderr)
    except Exception as e:
        print(f"  Retry judge call error on {record['row_id']}: {e}", file=sys.stderr)

    return None


def main():
    if not os.path.exists(SUMMARIES_PATH):
        raise FileNotFoundError(
            f"{SUMMARIES_PATH} not found. Run run_inference.py first."
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=api_key)

    os.makedirs(".tmp/scores", exist_ok=True)

    records = []
    with open(SUMMARIES_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Scoring {len(records)} rows with LLM judge ({JUDGE_MODEL})...")

    rows = []
    parse_failures = 0

    for i, record in enumerate(records):
        scores = score_row(client, record)

        if scores:
            rows.append({
                "row_id": record["row_id"],
                "faithfulness": scores.get("faithfulness"),
                "informativeness": scores.get("informativeness"),
                "fluency": scores.get("fluency"),
                "conciseness": scores.get("conciseness"),
            })
        else:
            parse_failures += 1
            rows.append({
                "row_id": record["row_id"],
                "faithfulness": None,
                "informativeness": None,
                "fluency": None,
                "conciseness": None,
            })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(records)} ({parse_failures} parse failures so far)")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["row_id", "faithfulness", "informativeness", "fluency", "conciseness"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows scored, {parse_failures} parse failures.")
    print(f"Output: {OUTPUT_PATH}")
    if parse_failures > len(records) * 0.05:
        print(
            f"WARNING: {parse_failures}/{len(records)} rows failed JSON parsing "
            f"(>{5:.0f}%). Check stderr output and consider investigating.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
