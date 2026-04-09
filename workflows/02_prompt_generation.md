# Workflow 02: Prompt Generation

## Objective
Build all (document × prompt style × repetition) combinations and save them to `.tmp/all_prompts.jsonl`.

## Inputs
`.tmp/dataset.jsonl` — produced by Step 1.

## Command
```bash
python tools/generate_prompts.py
```

## Expected Output
`.tmp/all_prompts.jsonl` — **210 rows** at `NUM_DOCS=10` (10 docs × 7 styles × 3 reps).

Seven effective prompt styles:
`zero_shot`, `role_primed`, `few_shot`, `chain_of_thought`, `perturbation_1`, `perturbation_2`, `perturbation_3`

Each row has fields: `prompt_id`, `doc_id`, `prompt_style`, `rep`, `prompt_text`, `reference_summary`, `source_document`.

`prompt_id` format: `{prompt_style}__{doc_id}__rep{N}` — e.g. `zero_shot__xsum_0000__rep1`

## Edge Cases

**Row count:**
Verify: `python -c "import json; lines=open('.tmp/all_prompts.jsonl').readlines(); print(len(lines))"` should print 210.

**Perturbation review:**
Before running inference, manually check the 3 perturbation prompts for `xsum_0000` to confirm they are semantically equivalent and sensible. Retrieve them with:
```bash
python -c "
import json
rows = [json.loads(l) for l in open('.tmp/all_prompts.jsonl')]
for r in rows:
    if r['doc_id'] == 'xsum_0000' and r['prompt_style'].startswith('perturbation') and r['rep'] == 1:
        print(r['prompt_style'])
        print(r['prompt_text'][:200])
        print('---')
"
```

**Few-shot examples:**
The two hardcoded examples in `generate_prompts.py` are topically neutral news summaries. If they look wrong or biased, edit the `FEW_SHOT_EXAMPLES` list in `tools/generate_prompts.py` before proceeding.

**Missing input:**
If `.tmp/dataset.jsonl` is not found, the script raises a clear `FileNotFoundError`. Run Step 1 first.
