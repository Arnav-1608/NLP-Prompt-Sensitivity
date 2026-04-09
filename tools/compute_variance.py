import os
import pandas as pd

SCORES_DIR = ".tmp/scores"
OUTPUT_PATH = ".tmp/scores/variance_table.csv"

SCORE_FILES = {
    "rouge": f"{SCORES_DIR}/rouge_scores.csv",
    "bertscore": f"{SCORES_DIR}/bertscore_scores.csv",
    "factual": f"{SCORES_DIR}/factual_scores.csv",
    "llm_judge": f"{SCORES_DIR}/llm_judge_scores.csv",
}

METRIC_COLS = ["rouge1", "rouge2", "rougeL", "bertscore_f1",
               "factual_score", "faithfulness", "informativeness", "fluency", "conciseness"]


def parse_row_id(row_id):
    # Format: {model_slug}__{prompt_style}__{doc_id}__rep{N}
    # e.g. llama3_70b__zero_shot__xsum_0000__rep1
    parts = row_id.split("__")
    if len(parts) != 4:
        return None, None, None, None
    model, prompt_style, doc_id, rep_str = parts
    rep = int(rep_str.replace("rep", ""))
    return model, prompt_style, doc_id, rep


def main():
    for name, path in SCORE_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run score_{name.replace('_', '')}.py first."
            )

    # Load and merge all score CSVs on row_id
    dfs = []
    for path in SCORE_FILES.values():
        df = pd.read_csv(path)
        dfs.append(df.set_index("row_id"))

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")
    merged = merged.reset_index()

    # Convert all metric columns to numeric (coerce None/empty to NaN)
    for col in METRIC_COLS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Parse row_id into components
    parsed = merged["row_id"].apply(parse_row_id)
    merged["model"] = [p[0] for p in parsed]
    merged["prompt_style"] = [p[1] for p in parsed]
    merged["doc_id"] = [p[2] for p in parsed]
    merged["rep"] = [p[3] for p in parsed]

    # Drop rows where parsing failed
    merged = merged.dropna(subset=["model", "prompt_style", "doc_id", "rep"])

    # Compute mean and std dev across reps per (model, prompt_style, doc_id)
    group_cols = ["model", "prompt_style", "doc_id"]
    available_metrics = [c for c in METRIC_COLS if c in merged.columns]

    records = []
    for (model, prompt_style, doc_id), group in merged.groupby(group_cols):
        for metric in available_metrics:
            values = group[metric].dropna()
            records.append({
                "model": model,
                "prompt_style": prompt_style,
                "doc_id": doc_id,
                "metric": metric,
                "mean": round(values.mean(), 6) if len(values) > 0 else None,
                "std_dev": round(values.std(), 6) if len(values) > 1 else 0.0,
                "n_reps": len(values),
            })

    variance_df = pd.DataFrame(records)
    variance_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Variance table: {len(variance_df)} rows. Output: {OUTPUT_PATH}")
    if len(variance_df) > 0:
        nonzero_std = (variance_df["std_dev"] > 0).sum()
        print(f"Rows with std_dev > 0: {nonzero_std}/{len(variance_df)}")
        if nonzero_std == 0:
            print("WARNING: All std_devs are 0. Models may be returning deterministic outputs — "
                  "this itself is a finding worth documenting.")


if __name__ == "__main__":
    main()
