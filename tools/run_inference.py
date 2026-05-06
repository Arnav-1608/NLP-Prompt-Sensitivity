import os
import json
import sys
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PROMPTS_PATH = ".tmp/all_prompts.jsonl"
OUTPUT_PATH = ".tmp/raw_summaries.jsonl"

MODELS = {
    "qwen3_32b": "qwen/qwen3-32b",
    "llama3_8b": "llama-3.1-8b-instant",
}

MAX_TOKENS = 256
RETRY_COUNT = 3
RETRY_WAIT = 60  # seconds to wait on rate limit
COURTESY_SLEEP = 13  # seconds between calls to stay under 6K TPM


def call_groq(client, model_id, prompt_text, retries=RETRY_COUNT, wait=RETRY_WAIT):
    messages = [{"role": "user", "content": prompt_text}]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                print(f"  Rate limited (attempt {attempt + 1}/{retries}) — waiting {wait}s...",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for model {model_id}")


def load_completed_ids():
    completed = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    completed.add(row["row_id"])
                except json.JSONDecodeError:
                    pass
    return completed


def strip_cot_preamble(text):
    """Strip reasoning preamble from chain-of-thought outputs.

    CoT prompts end with 'Let me identify the key points:' — models continue
    with reasoning steps before writing the actual summary. This strips
    everything up to and including the last summary marker so that only the
    final summary sentence(s) are scored.
    """
    import re
    patterns = [
        r"(?i)final summary\s*:",
        r"(?i)here is (?:a |the )?(?:brief |concise |short )?summary\s*[:\-]?",
        r"(?i)in summary\s*[:\-]",
        r"(?i)summary\s*:",
    ]
    last_match_end = -1
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            if m.end() > last_match_end:
                last_match_end = m.end()
    if last_match_end > 0:
        stripped = text[last_match_end:].strip()
        if stripped:
            return stripped
    return text


def is_refusal(text):
    refusal_markers = [
        "i'm sorry", "i cannot", "i can't", "as an ai", "i apologize",
        "i'm unable", "i am unable", "i won't", "i will not",
    ]
    lower = text.lower()
    return any(marker in lower for marker in refusal_markers)


def main():
    if not os.path.exists(PROMPTS_PATH):
        raise FileNotFoundError(
            f"{PROMPTS_PATH} not found. Run generate_prompts.py first."
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to your .env file.")

    client = Groq(api_key=api_key)

    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            prompts.append(json.loads(line))

    completed_ids = load_completed_ids()
    if completed_ids:
        print(f"Resuming — {len(completed_ids)} rows already complete, skipping them.")

    total = len(prompts) * len(MODELS)
    done = 0
    skipped = 0

    with open(OUTPUT_PATH, "a") as out_f:
        for model_slug, model_id in MODELS.items():
            print(f"\nRunning model: {model_id} (slug: {model_slug})")
            for prompt_row in prompts:
                row_id = f"{model_slug}__{prompt_row['prompt_id']}"

                if row_id in completed_ids:
                    skipped += 1
                    continue

                try:
                    generated = call_groq(client, model_id, prompt_row["prompt_text"])
                    time.sleep(COURTESY_SLEEP)
                except Exception as e:
                    print(f"  ERROR on {row_id}: {e}", file=sys.stderr)
                    continue

                if prompt_row["prompt_style"] == "chain_of_thought":
                    generated = strip_cot_preamble(generated)

                if is_refusal(generated):
                    print(f"  REFUSAL detected: {row_id}", file=sys.stderr)

                record = {
                    "row_id": row_id,
                    "model": model_slug,
                    "prompt_style": prompt_row["prompt_style"],
                    "doc_id": prompt_row["doc_id"],
                    "rep": prompt_row["rep"],
                    "generated_summary": generated,
                    "reference_summary": prompt_row["reference_summary"],
                    "source_document": prompt_row["source_document"],
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                done += 1
                if done % 10 == 0:
                    print(f"  Progress: {done + skipped}/{total} ({skipped} skipped)")

    total_written = done + skipped
    print(f"\nDone. Wrote {done} new rows. Total in file: {total_written}.")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
