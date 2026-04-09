# NLP Prompt Sensitivity Study

An empirical research pipeline measuring how **prompt style variation affects summarization quality and consistency** across two open-source LLM families (Llama and Qwen) via the Groq API.

**Research question:** How sensitive are open-source LLMs to prompt style variation in summarization tasks — and does that sensitivity differ across model families?

---

## Current Status

**Pilot run (10 documents) — fully complete.**

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1 | `prepare_dataset.py` | `.tmp/dataset.jsonl` | Done |
| 2 | `generate_prompts.py` | `.tmp/all_prompts.jsonl` | Done |
| 3 | `run_inference.py` | `.tmp/raw_summaries.jsonl` | Done |
| 4 | `score_rouge.py` | `.tmp/scores/rouge_scores.csv` | Done |
| 5 | `score_bertscore.py` | `.tmp/scores/bertscore_scores.csv` | Done |
| 6 | `score_minicheck.py` | `.tmp/scores/factual_scores.csv` | Done |
| 7 | `score_llm_judge.py` | `.tmp/scores/llm_judge_scores.csv` | Done |
| 8 | `compute_variance.py` | `.tmp/scores/variance_table.csv` | Done |
| 9 | `aggregate_results.py` | `.tmp/master_results.csv` | Done |
| 10 | `generate_pdf_report.py` | `outputs/prompt_sensitivity_report.pdf` | Done |

Next: scale to 30 or 75 documents for the full study (see [Scaling](#scaling)).

---

## Models

| Model | Groq ID | Role |
|-------|---------|------|
| Llama 3.3 70B | `llama-3.3-70b-versatile` | Summarization inference |
| Qwen3 32B | `qwen/qwen3-32b` | Summarization inference |
| Llama 3.1 8B | `llama-3.1-8b-instant` | LLM-as-judge scoring |

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
- Free API keys from [Groq](https://console.groq.com) (no credit card) and [HuggingFace](https://huggingface.co/settings/tokens)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
```

---

## Running the Pipeline

Run steps in order. Each script reads from `.tmp/` and writes its output back to `.tmp/`.

```bash
python tools/prepare_dataset.py       # Step 1 — fetch & stratify XSum docs
python tools/generate_prompts.py      # Step 2 — build all prompt variants
python tools/run_inference.py         # Step 3 — call Groq for summaries
python tools/score_rouge.py           # Step 4 — ROUGE-1/2/L scoring
python tools/score_bertscore.py       # Step 5 — BERTScore F1
python tools/score_minicheck.py       # Step 6 — factual consistency (MiniCheck)
python tools/score_llm_judge.py       # Step 7 — LLM-as-judge (faithfulness/fluency/etc.)
python tools/compute_variance.py      # Step 8 — std dev across 3 repetitions
python tools/aggregate_results.py     # Step 9 — merge all scores
python tools/generate_pdf_report.py   # Step 10 — generate PDF report
```

The final report is written to `outputs/prompt_sensitivity_report.pdf`.

---

## Scaling

To scale beyond the 10-document pilot, change one line in `tools/run_inference.py`:

```python
NUM_DOCS = 10   # pilot
NUM_DOCS = 30   # small run
NUM_DOCS = 75   # full study
```

Nothing else changes. All downstream scripts read from `.tmp/` automatically.

**Before scaling, confirm:**
- All 10 pilot steps completed without errors
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
| Factual consistency | `score_minicheck.py` | Hallucination / faithfulness via MiniCheck |
| LLM-as-judge | `score_llm_judge.py` | Faithfulness, informativeness, fluency, conciseness (1–5) |
| Output variance | `compute_variance.py` | Std dev across 3 repetitions per condition |

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
│   ├── score_minicheck.py
│   ├── score_llm_judge.py
│   ├── compute_variance.py
│   ├── aggregate_results.py
│   └── generate_pdf_report.py
├── workflows/                 # SOPs for each pipeline step
├── .tmp/                      # Intermediate outputs (gitignored, regeneratable)
├── outputs/                   # Final PDF report (gitignored)
├── requirements.txt
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

The 10-document pilot requires ~300 inference calls — well within single-day limits.

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
