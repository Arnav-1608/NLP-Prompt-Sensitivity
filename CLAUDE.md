# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Purpose

Empirical NLP research study measuring how prompt style variation affects summarization quality and consistency across two open-source LLM families (Llama and Qwen) via the Groq free API.

**Research question:** How sensitive are open-source LLMs to prompt style variation in summarization tasks — and does that sensitivity differ across model families?

**All infrastructure is free:** Groq free API (no credit card), HuggingFace Inference API (free tier), local pip packages, CPU-only.

---

## Setup

```bash
pip install groq datasets rouge-score bert-score reportlab matplotlib pandas huggingface_hub
```

Create `.env` in the project root:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
```

- **Groq:** console.groq.com (no credit card required)
- **HuggingFace:** huggingface.co → Settings → Access Tokens

---

## Pipeline — Run in Order

Each script reads the previous step's output from `.tmp/`. Never skip steps or reorder.

```
Step 1  → python tools/prepare_dataset.py      → .tmp/dataset.jsonl
Step 2  → python tools/generate_prompts.py     → .tmp/all_prompts.jsonl
Step 3  → python tools/run_inference.py        → .tmp/raw_summaries.jsonl
Step 4  → python tools/score_rouge.py          → .tmp/scores/rouge_scores.csv
Step 5  → python tools/score_bertscore.py      → .tmp/scores/bertscore_scores.csv
Step 6  → python tools/score_minicheck.py      → .tmp/scores/factual_scores.csv
Step 7  → python tools/score_llm_judge.py      → .tmp/scores/llm_judge_scores.csv
Step 8  → python tools/compute_variance.py     → .tmp/scores/variance_table.csv
Step 9  → python tools/aggregate_results.py    → .tmp/master_results.csv
Step 10 → python tools/generate_pdf_report.py  → outputs/prompt_sensitivity_report.pdf
```

**To scale the study**, only change one variable in `run_inference.py`:
```python
NUM_DOCS = 10   # Pilot. Change to 30 (small run) or 75 (full study)
```

---

## Architecture

### WAT Framework (Workflows → Agents → Tools)

- **`workflows/`** — Markdown SOPs: read the relevant one before running any tool. Each defines objective, inputs, which script to call, expected outputs, and edge cases.
- **`tools/`** — Plain Python scripts: deterministic, CSV/JSONL outputs, safe to re-run. No GPU needed.
- **`.tmp/`** — All intermediate outputs. Safe to delete and regenerate from any step.

### Experimental Design

- **Models:** `llama-3.3-70b-versatile` and `qwen/qwen3-32b` via Groq
- **Prompt styles (5 per document):** zero-shot, role-primed, few-shot (2 examples), chain-of-thought, prompt perturbation (3 surface paraphrases)
- **Dataset:** XSum via HuggingFace `datasets`, stratified by document length (short/medium/long)
- **Repetitions:** 3 runs per (document × prompt style) to measure output variance

### Row ID Format

Every output row across all scripts uses this join key:
```
{model_slug}__{prompt_style}__{doc_id}__rep{N}
# e.g. qwen3_32b__chain_of_thought__xsum_0007__rep2
```

### Inference Pattern

```python
from groq import Groq
import time, os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def call_groq(model, messages, retries=3, wait=60):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
```

---

## Metrics

| Metric | Script | What it measures |
|--------|--------|-----------------|
| ROUGE-1/2/L | `score_rouge.py` | N-gram overlap with reference |
| BERTScore F1 | `score_bertscore.py` | Semantic similarity to reference |
| Factual consistency | `score_minicheck.py` | Hallucination / faithfulness via MiniCheck (HF API) |
| LLM-as-judge | `score_llm_judge.py` | Faithfulness / informativeness / fluency / conciseness (1–5) via `llama-3.1-8b-instant` |
| Output variance | `compute_variance.py` | Std dev across 3 reps per condition |

**LLM judge expected output format:** `{"faithfulness": 4, "informativeness": 3, "fluency": 5, "conciseness": 4}`
Strip markdown fences before JSON parsing; retry once with a stricter prompt on parse failure.

**Variance is a primary finding.** A prompt style with slightly lower ROUGE but much lower std dev is more reliable in practice — surface this tradeoff explicitly.

---

## PDF Report (`generate_pdf_report.py`)

Uses `reportlab` with `SimpleDocTemplate`, `Paragraph`, `Table`, `Spacer`, and `Image` (matplotlib charts saved as PNG, embedded via `reportlab.platypus.Image`).

**Do not use Unicode subscript/superscript characters** — use ReportLab's `<sub>` and `<super>` XML tags inside `Paragraph` objects.

Sections: Title Page → Executive Summary → Methodology → Results (Quality / Cross-Model / Variance / Perturbation Sensitivity / Factual Consistency) → Discussion → Appendix.

When `NUM_DOCS = 10`, add a visible banner on the title page: *"PILOT RUN — 10 documents. For illustrative purposes only."*

---

## Groq Free Tier Limits

| Model | Req/day | Tokens/day |
|-------|---------|-----------|
| `llama-3.3-70b-versatile` | 1,000 | 100,000 |
| `qwen/qwen3-32b` | 1,000 | 500,000 |
| `llama-3.1-8b-instant` (judge) | 14,400 | 500,000 |

Pilot (10 docs) = 300 inference calls, comfortably within single-day limits.

---

## Known Edge Cases

| Issue | Mitigation |
|-------|-----------|
| Groq 429 rate limit | Retry loop with 60s wait (built into `run_inference.py`) |
| Model returns refusal instead of summary | Log refusal count per prompt style — this is itself a finding |
| LLM judge returns malformed JSON | Strip markdown fences, retry with stricter prompt |
| BERTScore first run slow (~1.3GB download) | Expected, cached after that |
| MiniCheck HF API cold start (30s delay) | Add 120s timeout |
| `.tmp/` file missing when resuming | Each script should check for its input file and raise a clear error |

---

## Failure Recovery

1. Fix the erroring script
2. Test on 2 documents before re-running at full scale
3. Update the relevant `workflows/` SOP with what you learned
4. Resume from the last saved `.tmp/` file — never restart from Step 1

---

## Scaling Checklist (Before Going Beyond 10 Docs)

- [ ] All 10 pilot runs completed without errors
- [ ] PDF generated with real data
- [ ] LLM judge JSON parsed correctly on all 300 outputs
- [ ] No unexpected model refusals
- [ ] Groq rate limits handled cleanly
