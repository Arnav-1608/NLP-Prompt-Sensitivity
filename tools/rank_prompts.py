"""
Composite prompt ranking across all five metric dimensions.

Reads:  .tmp/master_results.csv
        .tmp/scores/variance_table.csv
Writes: .tmp/scores/prompt_rankings.csv

Each prompt style is scored under four weighting schemes. All input metrics are
min-max normalised to [0, 1] per model before weighting so dimensions are
comparable. Reliability = 1 - normalised(mean_std_dev); lower variance → higher
reliability score.
"""

import os
import sys
import pandas as pd

MASTER_PATH = ".tmp/master_results.csv"
VARIANCE_PATH = ".tmp/scores/variance_table.csv"
OUTPUT_PATH = ".tmp/scores/prompt_rankings.csv"

LLM_JUDGE_COLS = ["faithfulness", "informativeness", "fluency", "conciseness"]

# Each scheme maps dimension → relative weight (unnormalised; script normalises).
WEIGHT_SCHEMES = {
    "equal": {
        "rouge1": 1, "rougeL": 1, "bertscore_f1": 1,
        "factual_score": 1, "llm_judge_mean": 1, "reliability": 1,
    },
    "faithfulness_heavy": {
        "rouge1": 0.5, "rougeL": 0.5, "bertscore_f1": 1,
        "factual_score": 2, "llm_judge_mean": 2, "reliability": 1,
    },
    "rouge_heavy": {
        "rouge1": 2, "rougeL": 2, "bertscore_f1": 1,
        "factual_score": 0.5, "llm_judge_mean": 0.5, "reliability": 1,
    },
    "quality_only": {
        "rouge1": 1, "rougeL": 1, "bertscore_f1": 1,
        "factual_score": 1, "llm_judge_mean": 1, "reliability": 0,
    },
}

QUALITY_DIMS = ["rouge1", "rougeL", "bertscore_f1", "factual_score", "llm_judge_mean"]


def minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def main():
    for path in (MASTER_PATH, VARIANCE_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run previous pipeline steps first.")

    master = pd.read_csv(MASTER_PATH)
    variance = pd.read_csv(VARIANCE_PATH)

    for col in ["rouge1", "rougeL", "bertscore_f1", "factual_score"] + LLM_JUDGE_COLS:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    # LLM judge composite: mean of four sub-dimensions (each scored 1–5), scaled to [0, 1].
    judge_available = [c for c in LLM_JUDGE_COLS if c in master.columns]
    if judge_available:
        master["llm_judge_mean"] = (master[judge_available].mean(axis=1) - 1) / 4
    else:
        master["llm_judge_mean"] = float("nan")

    # Mean per (model, prompt_style) across all docs and reps.
    quality_cols = [c for c in QUALITY_DIMS if c in master.columns]
    agg = (
        master.groupby(["model", "prompt_style"])[quality_cols]
        .mean()
        .reset_index()
    )

    # Mean std_dev per (model, prompt_style) collapsed across all metrics and docs.
    var_agg = (
        variance.groupby(["model", "prompt_style"])["std_dev"]
        .mean()
        .reset_index(name="mean_std_dev")
    )
    agg = agg.merge(var_agg, on=["model", "prompt_style"], how="left")
    agg["mean_std_dev"] = agg["mean_std_dev"].fillna(0.0)

    # Normalise within each model so schemes are comparable across model families.
    for model_name, group_idx in agg.groupby("model").groups.items():
        group = agg.loc[group_idx]
        for col in quality_cols:
            if col in agg.columns:
                agg.loc[group_idx, f"norm_{col}"] = minmax_norm(group[col]).values
        # Reliability: invert normalised std_dev (lower variance → higher score).
        agg.loc[group_idx, "norm_reliability"] = (1 - minmax_norm(group["mean_std_dev"])).values

    # Compute composite score for each scheme.
    for scheme_name, weights in WEIGHT_SCHEMES.items():
        total_weight = sum(weights.values())
        score = pd.Series(0.0, index=agg.index)
        for dim, w in weights.items():
            norm_col = f"norm_{dim}" if dim != "reliability" else "norm_reliability"
            if norm_col in agg.columns:
                score += w * agg[norm_col].fillna(0.0)
        agg[f"score_{scheme_name}"] = (score / total_weight).round(6)

    # Rank within each model (rank 1 = best).
    for scheme_name in WEIGHT_SCHEMES:
        score_col = f"score_{scheme_name}"
        agg[f"rank_{scheme_name}"] = (
            agg.groupby("model")[score_col]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    agg.to_csv(OUTPUT_PATH, index=False)
    print(f"Prompt rankings written to {OUTPUT_PATH}\n")

    # Print readable summary per model.
    rank_cols = [f"rank_{s}" for s in WEIGHT_SCHEMES]
    score_cols = [f"score_{s}" for s in WEIGHT_SCHEMES]
    display_cols = ["prompt_style"] + score_cols + rank_cols

    for model_name, group in agg.groupby("model"):
        print(f"=== {model_name} ===")
        row = group[display_cols].sort_values("score_equal", ascending=False)
        print(row.to_string(index=False))
        print()

    # Highlight any prompt whose top rank changes across schemes.
    print("--- Rank stability check (top-ranked prompt per scheme per model) ---")
    for model_name, group in agg.groupby("model"):
        winners = {s: group.loc[group[f"rank_{s}"] == 1, "prompt_style"].values for s in WEIGHT_SCHEMES}
        unique_winners = {s: v[0] if len(v) else "N/A" for s, v in winners.items()}
        all_same = len(set(unique_winners.values())) == 1
        status = "STABLE" if all_same else "VARIES"
        print(f"  {model_name}: {status}")
        for s, w in unique_winners.items():
            print(f"    {s:<22} → {w}")
        print()


if __name__ == "__main__":
    main()
