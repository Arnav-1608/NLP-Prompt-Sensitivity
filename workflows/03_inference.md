# Workflow 03: LLM Inference

## Objective
Run all prompts through both models (Llama and Qwen via Groq) and save the generated summaries.

## Inputs
`.tmp/all_prompts.jsonl` — produced by Step 2.
`.env` — must contain `GROQ_API_KEY`.

## Command
```bash
python tools/run_inference.py
```

## Expected Output
`.tmp/raw_summaries.jsonl` — **420 rows** at `NUM_DOCS=10` (210 prompts × 2 models).

Each row: `row_id`, `model`, `prompt_style`, `doc_id`, `rep`, `generated_summary`, `reference_summary`, `source_document`.

`row_id` format: `{model_slug}__{prompt_style}__{doc_id}__rep{N}` — e.g. `llama3_70b__zero_shot__xsum_0000__rep1`

Expected wall-clock time: ~10–20 minutes for 420 calls (with 1s courtesy sleep between calls).

## Edge Cases

**Rate limit hit (HTTP 429):**
The retry loop waits 60s and retries up to 3 times. If you still get failures, Groq daily limits may be exhausted — wait until midnight UTC and re-run. The script resumes from where it left off.

**Resume after failure:**
If the script crashes mid-run, re-run it. It reads existing `row_id` values from `.tmp/raw_summaries.jsonl` at startup and skips completed rows. Do not delete the file before re-running.

**Model refusals:**
Refusal text is logged to stderr with the `row_id` but still saved to the output file. Count refusals after the run:
```bash
python -c "
import json, sys
rows = [json.loads(l) for l in open('.tmp/raw_summaries.jsonl')]
refusals = [r for r in rows if any(m in r['generated_summary'].lower() for m in ['i cannot', \"i'm sorry\", 'as an ai'])]
print(f'{len(refusals)} refusals out of {len(rows)} rows')
"
```

**Groq API key not set:**
The script raises `EnvironmentError` immediately. Ensure `.env` exists with `GROQ_API_KEY=your_key`.

**qwen/qwen3-32b model name:**
The slash in the model name is handled internally — the API call uses the full string `"qwen/qwen3-32b"` while the slug `qwen3_32b` is used in row_ids. Do not edit these strings.
