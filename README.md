# NLP Prompt Sensitivity Study

An empirical research pipeline measuring how **prompt style variation affects summarization quality and consistency** across two open-source LLM families (Llama and Qwen) via the Groq API.

**Research question:** How sensitive are open-source LLMs to prompt style variation in summarization tasks — and does that sensitivity differ across model families?

---

## Current Status

**50-document run — fully complete. PDF report generated.**

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1 | `prepare_dataset.py` | `.tmp/dataset.jsonl` | Done |
| 2 | `generate_prompts.py` | `.tmp/all_prompts.jsonl` | Done |
| 3 | `run_inference.py` | `.tmp/raw_summaries.jsonl` | Done (2,100 rows) |
| 4 | `score_rouge.py` | `.tmp/scores/rouge_scores.csv` | Done |
| 5 | `score_bertscore.py` | `.tmp/scores/bertscore_scores.csv` | Done |
| 6 | `score_minicheck.py` | `.tmp/scores/factual_scores.csv` | Done |
| 7 | `score_llm_judge.py` | `.tmp/scores/llm_judge_scores.csv` | Done (~87% coverage) |
| 8 | `compute_variance.py` | `.tmp/scores/variance_table.csv` | Done |
| 9 | `aggregate_results.py` | `.tmp/master_results.csv` | Done |
| 10 | `rank_prompts.py` | `.tmp/scores/prompt_rankings.csv` | Done |
| 11 | `generate_pdf_report.py` | `outputs/prompt_sensitivity_report.pdf` | Done |

---

## Models

| Model | Groq ID | Role |
|-------|---------|------|
| Llama 3.1 8B | `llama-3.1-8b-instant` | Summarization inference |
| Qwen3 32B | `qwen/qwen3-32b` | Summarization inference |
| Llama 3.1 8B | `llama-3.1-8b-instant` | LLM-as-judge scoring |

> **Note on model selection:** The original design called for `llama-3.3-70b-versatile`. At 50 documents, the full pipeline (inference + LLM-as-judge scoring) requires over 800,000 tokens — approximately 8× the 70B model's 100,000 token/day free-tier limit. Running 70B at this scale would have taken multiple days across both the inference and judge steps, which exceeded our project deadline. We switched to `llama-3.1-8b-instant` (500,000 tokens/day), which completed the full pipeline within our timeline. This is a meaningful difference: the 8B and 70B models are different capability tiers. Results should not be generalised to 70B-scale behaviour.

---

## Prompt Styles Tested

- **Zero-shot** — plain instruction, no examples
- **Role-primed** — system prompt assigns an expert persona
- **Few-shot** — 2 in-context examples provided
- **Chain-of-thought** — model prompted to reason before summarizing
- **Prompt perturbation** — 3 surface paraphrases of the same instruction

---

## Setup

### Prerequisites

- Python 3.9+
- Free API key from [Groq](https://console.groq.com) (no credit card required)

### Install dependencies

```bash
pip install groq datasets rouge-score bert-score reportlab matplotlib pandas scipy sentence-transformers huggingface_hub python-dotenv
```

### Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

---

## Running the Pipeline

Run steps in order. Each script reads from `.tmp/` and writes its output back to `.tmp/`.

```bash
python tools/prepare_dataset.py       # Step 1  — fetch & stratify XSum docs
python tools/generate_prompts.py      # Step 2  — build all prompt variants
python tools/run_inference.py         # Step 3  — call Groq for summaries
python tools/score_rouge.py           # Step 4  — ROUGE-1/2/L scoring
python tools/score_bertscore.py       # Step 5  — BERTScore F1
python tools/score_minicheck.py       # Step 6  — factual consistency (local NLI)
python tools/score_llm_judge.py       # Step 7  — LLM-as-judge (faithfulness/fluency/etc.)
python tools/compute_variance.py      # Step 8  — std dev across 3 repetitions
python tools/aggregate_results.py     # Step 9  — merge all scores
python tools/rank_prompts.py          # Step 10 — composite prompt ranking
python tools/generate_pdf_report.py   # Step 11 — generate PDF report
```

The final report is written to `outputs/prompt_sensitivity_report.pdf`.

---

## Scaling

The 50-document run is complete. To scale further, change one line in `tools/run_inference.py`:

```python
NUM_DOCS = 50   # current run (complete)
NUM_DOCS = 75   # full study
```

Nothing else changes. All downstream scripts read from `.tmp/` automatically.

**Before scaling further, confirm:**
- All current pipeline steps completed without errors
- PDF generated with real data
- LLM judge JSON parsed correctly on all outputs
- No unexpected model refusals
- Groq rate limits handled cleanly

---

## Metrics

| Metric | Script | What it measures |
|--------|--------|-----------------|
| ROUGE-1/2/L | `score_rouge.py` | N-gram overlap with reference summary |
| BERTScore F1 | `score_bertscore.py` | Semantic similarity to reference |
| Factual consistency | `score_minicheck.py` | NLI entailment score (local `cross-encoder/nli-deberta-v3-base`) |
| LLM-as-judge | `score_llm_judge.py` | Faithfulness, informativeness, fluency, conciseness (1–5) |
| Output variance | `compute_variance.py` | Std dev across 3 repetitions per condition |
| Composite ranking | `rank_prompts.py` | Weighted composite score across all metrics under 4 schemes |

### Factual Consistency Scorer

`score_minicheck.py` runs `cross-encoder/nli-deberta-v3-base` **locally** via `sentence-transformers`. It treats the source document as the NLI premise and the generated summary as the hypothesis. The **entailment probability** (0–1) is the factual consistency score — a score close to 1 means the document supports the summary; a low score indicates potential hallucination.

The model (~700 MB) downloads automatically on first run and is cached. No API key or internet connection needed for subsequent runs.

### Composite Prompt Ranking

`rank_prompts.py` scores each prompt style across six dimensions (ROUGE-1, ROUGE-L, BERTScore, factual consistency, LLM judge composite, reliability) under four weighting schemes:

| Scheme | Description |
|--------|-------------|
| Equal | All dimensions weighted equally |
| Faithfulness-Heavy | Factual + judge weighted 2×; ROUGE halved |
| ROUGE-Heavy | ROUGE-1 + ROUGE-L weighted 2×; factual + judge halved |
| Quality Only | Reliability weight = 0; pure quality score |

A **rank stability check** reports whether the top-ranked prompt is the same across all four schemes (STABLE) or differs (VARIES). A stable winner is a robust recommendation regardless of which metrics are prioritised.

---

## Future Work

- **Scale to `llama-3.3-70b-versatile`** — The primary planned extension is replicating this study using the 70B model when there is no time constraint. Comparing 8B and 70B results directly would reveal whether prompt sensitivity is a model-size phenomenon or a model-family phenomenon.
- **Increase document count** — Scaling from 50 to 75+ documents would improve statistical power for significance tests.
- **Additional prompt styles** — Tree-of-thought, self-consistency, and instruction-tuned formats are natural extensions.
- **Cross-dataset validation** — Replicating on XSum or NewsRoom (beyond CNN/DailyMail) would test whether findings generalise across summarization domains.

---

## Project Structure

```
.
├── tools/                     # Pipeline scripts (run in order)
│   ├── prepare_dataset.py
│   ├── generate_prompts.py
│   ├── run_inference.py
│   ├── score_rouge.py
│   ├── score_bertscore.py
│   ├── score_minicheck.py     # local NLI scorer (cross-encoder/nli-deberta-v3-base)
│   ├── score_llm_judge.py
│   ├── compute_variance.py
│   ├── aggregate_results.py
│   ├── rank_prompts.py        # composite weighted ranking (new)
│   └── generate_pdf_report.py
├── workflows/                 # SOPs for each pipeline step
├── .tmp/                      # Intermediate outputs (gitignored, regeneratable)
├── outputs/                   # Final PDF report (gitignored)
├── CLAUDE.md                  # AI assistant instructions
└── .env                       # API keys (gitignored — never commit)
```

---

## Groq Free Tier Limits

| Model | Requests/day | Tokens/day |
|-------|-------------|-----------|
| `llama-3.3-70b-versatile` | 1,000 | 100,000 |
| `qwen/qwen3-32b` | 1,000 | 500,000 |
| `llama-3.1-8b-instant` | 14,400 | 500,000 |

---

## Row ID Format

All output files use a consistent join key:

```
{model_slug}__{prompt_style}__{doc_id}__rep{N}
# e.g. qwen3_32b__chain_of_thought__xsum_0007__rep2
```

---

## Failure Recovery

1. Fix the erroring script
2. Test on 2 documents before re-running at full scale
3. Resume from the last saved `.tmp/` file — never restart from Step 1
