# Workflow 06: Variance Analysis

## Objective
Compute mean and standard deviation across the 3 repetitions per (model, prompt_style, doc_id) condition for every metric.

## Inputs
All four score CSVs (must all exist before running):
- `.tmp/scores/rouge_scores.csv`
- `.tmp/scores/bertscore_scores.csv`
- `.tmp/scores/factual_scores.csv`
- `.tmp/scores/llm_judge_scores.csv`

## Command
```bash
python tools/compute_variance.py
```

## Expected Output
`.tmp/scores/variance_table.csv` — long format with columns: `model, prompt_style, doc_id, metric, mean, std_dev, n_reps`

Expected rows: 10 docs × 7 prompt styles × 9 metrics × 2 models = **1,260 rows** (some may be NaN if judge parse failed).

## Edge Cases

**All std_devs are zero:**
This can happen if the models produce identical outputs across all 3 repetitions (fully deterministic). The script prints a warning. This is itself a finding — document it in the Discussion section of the PDF. It does not prevent the pipeline from continuing.

**Missing rows (n_reps < 3):**
If `run_inference.py` failed mid-run and some (model, prompt_style, doc_id) combinations have fewer than 3 reps, `std_dev` will be computed on fewer values. The `n_reps` column in the output tracks this. Rows with `n_reps=1` will have `std_dev=0.0`.

**NaN in judge scores:**
`pd.to_numeric(col, errors='coerce')` converts null/empty strings to NaN before computing std. Groups where all 3 reps failed JSON parsing will show NaN mean and NaN std_dev — expected behaviour.

**Perturbation styles:**
Variance for `perturbation_1`, `perturbation_2`, `perturbation_3` is computed independently (across their own 3 reps). Cross-perturbation variance (how much the 3 paraphrases differ from each other) is a separate metric computed in Step 9 (`aggregate_results.py`).
