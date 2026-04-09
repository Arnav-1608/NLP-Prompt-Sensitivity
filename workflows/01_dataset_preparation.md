# Workflow 01: Dataset Preparation

## Objective
Load the XSum dataset, stratify by document length, and save a reproducible sample to `.tmp/dataset.jsonl`.

## Inputs
None (Step 1 — no prior `.tmp/` file required).

## Command
```bash
python tools/prepare_dataset.py
```

## Expected Output
`.tmp/dataset.jsonl` — **10 rows** (one per sampled document) with fields:
- `doc_id`: `xsum_0000` through `xsum_0009`
- `document`: truncated to 600 words
- `reference_summary`: original XSum gold summary
- `length_tier`: `short`, `medium`, or `long`

Expected tier distribution: ~3 short, ~4 medium, ~3 long (remainder goes to medium).

## Edge Cases

**HuggingFace download fails:**
The XSum dataset is downloaded automatically on first run (~200MB). If the network fails, re-run — `datasets` caches to `~/.cache/huggingface`. Check with `datasets.load_dataset("xsum", split="train")` in a Python shell.

**Not enough docs in a tier:**
Extremely short or long XSum articles are rare. If you see `ValueError: Not enough <tier> docs`, lower `NUM_DOCS` or widen the tier boundaries in `prepare_dataset.py`.

**Output row count mismatch:**
Verify with `wc -l .tmp/dataset.jsonl` — should equal `NUM_DOCS`. If fewer, check for write errors.

**Reproducibility:**
The script uses `random.seed(42)`. Running it twice produces identical output. Delete `.tmp/dataset.jsonl` before re-running if you want to reset.
