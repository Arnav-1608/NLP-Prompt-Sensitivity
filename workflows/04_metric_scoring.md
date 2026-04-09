# Workflow 04: ROUGE and BERTScore Scoring

## Objective
Compute n-gram overlap (ROUGE) and semantic similarity (BERTScore) for all generated summaries.

## Inputs
`.tmp/raw_summaries.jsonl` — produced by Step 3.

## Commands
Run both scripts (order does not matter — they are independent):
```bash
python tools/score_rouge.py
python tools/score_bertscore.py
```

## Expected Outputs

`.tmp/scores/rouge_scores.csv` — **420 rows**, columns: `row_id, rouge1, rouge2, rougeL` (F1 scores)
- Expected mean ROUGE-1 for XSum: 0.2–0.6

`.tmp/scores/bertscore_scores.csv` — **420 rows**, columns: `row_id, bertscore_f1`
- Expected mean BERTScore F1 with distilbert-base-uncased: 0.80–0.95

## Edge Cases

**BERTScore first-run download (~1.3GB):**
The `distilbert-base-uncased` model is downloaded once and cached by the `transformers` library at `~/.cache/huggingface`. The script prints a notice before triggering the download. Expected runtime on CPU: 5–15 minutes for 420 rows.

**ROUGE near zero:**
Very low ROUGE scores (below 0.05) likely indicate model refusals or very short outputs. This is expected behaviour — do not filter these rows. The refusal rate is a research finding.

**BERTScore memory:**
On machines with limited RAM, running BERTScore on 420 rows at once may be slow. If it crashes, split the input into batches of 100 rows and concatenate the CSVs manually.

**CSV column names:**
`rougeL` uses a capital L. All downstream scripts expect this exact casing — do not rename.
