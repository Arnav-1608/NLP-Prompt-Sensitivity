# Workflow 05: Factual Consistency and LLM Judge Scoring

## Objective
Score all summaries for factual consistency (MiniCheck via HuggingFace) and multi-dimensional quality (LLM judge via Groq).

## Inputs
`.tmp/raw_summaries.jsonl` — produced by Step 3.
`.env` — must contain `HF_TOKEN` (for MiniCheck) and `GROQ_API_KEY` (for LLM judge).

## Commands
Run both scripts (order does not matter — they are independent):
```bash
python tools/score_minicheck.py
python tools/score_llm_judge.py
```

## Expected Outputs

`.tmp/scores/factual_scores.csv` — **420 rows**, columns: `row_id, factual_score` (float 0–1)

`.tmp/scores/llm_judge_scores.csv` — **420 rows**, columns: `row_id, faithfulness, informativeness, fluency, conciseness` (integers 1–5 or null on parse failure)

## Edge Cases

**MiniCheck cold start (30s delay on first call):**
The HuggingFace Inference API loads the model on first request. The script retries with a 30s wait on HTTP 503. This is expected — do not cancel the script during this wait.

**MiniCheck response format:**
The API returns `[[{...}]]` (list of lists). The script extracts the score for the label matching `LABEL_1`, `CONSISTENT`, `ENTAILMENT`, or `TRUE`. If the model updates its label naming, inspect raw responses with a test call and update `extract_factual_score()` in `score_minicheck.py`.

**LLM judge JSON parse failures:**
If >5% of rows produce null scores, the script prints a warning to stderr. Investigate the raw responses by adding a `print(text)` before the `json.loads()` call. Common causes: the model wraps JSON in prose, or returns a numbered list instead of an object. The two-stage retry (normal prompt → strict prompt) handles most cases.

**LLM judge wall-clock time:**
420 sequential API calls at ~0.3s courtesy sleep each takes ~5–10 minutes. Do not interrupt mid-run — results are written row-by-row and are not resumable (unlike `run_inference.py`). If interrupted, delete `llm_judge_scores.csv` and re-run from the beginning.

**HF_TOKEN not set:**
`score_minicheck.py` raises `EnvironmentError` immediately. Ensure `.env` exists with `HF_TOKEN=your_token`.
