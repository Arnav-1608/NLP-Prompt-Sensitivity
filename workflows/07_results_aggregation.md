# Workflow 07: Results Aggregation

## Objective
Merge all score CSVs into one flat master table and compute cross-perturbation sensitivity scores.

## Inputs
All four score CSVs from Steps 4–5 (same as Step 8 — run after all scoring scripts complete).

## Command
```bash
python tools/aggregate_results.py
```

## Expected Output
`.tmp/master_results.csv` — **420 rows** (one per raw_summary row), all metric columns merged, plus:
- `model`, `prompt_style`, `doc_id`, `rep` (parsed from `row_id`)
- `perturbation_sensitivity` (cross-perturbation std averaged across metrics, for perturbation rows)

The script also prints a summary table (mean per model × prompt_style) to stdout.

## Edge Cases

**Outer join on row_id:**
All four CSVs are joined with `outer` merge. If a scoring script produced fewer rows than expected (e.g. MiniCheck failed for some rows), the merged table will have NaN in the `factual_score` column for those rows — this is expected and handled correctly by downstream scripts.

**Cross-perturbation sensitivity calculation:**
For each (doc, model, rep), the std dev across `perturbation_1`, `perturbation_2`, `perturbation_3` is averaged across all metrics. A high value means minor wording changes produced substantially different outputs. If there are fewer than 2 perturbation variants with valid scores, sensitivity will be NaN.

**Reviewing the summary table:**
The printed summary should show non-trivial variation across prompt styles. If all scores are identical, check that the `row_id` join is working correctly — each style should have distinct values.

**Output used by the PDF generator:**
`master_results.csv` is the sole input to Step 10. Ensure it exists and has non-empty `model`, `prompt_style`, `doc_id` columns before running the PDF generator.
