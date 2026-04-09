import os
import pandas as pd

SCORES_DIR = ".tmp/scores"
OUTPUT_PATH = ".tmp/master_results.csv"

SCORE_FILES = {
    "rouge": f"{SCORES_DIR}/rouge_scores.csv",
    "bertscore": f"{SCORES_DIR}/bertscore_scores.csv",
    "factual": f"{SCORES_DIR}/factual_scores.csv",
    "llm_judge": f"{SCORES_DIR}/llm_judge_scores.csv",
}

METRIC_COLS = ["rouge1", "rouge2", "rougeL", "bertscore_f1",
               "factual_score", "faithfulness", "informativeness", "fluency", "conciseness"]

PERTURBATION_STYLES = ["perturbation_1", "perturbation_2", "perturbation_3"]


def parse_row_id(row_id):
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
                f"{path} not found. Run the corresponding scoring script first."
            )

    # Load and merge all score CSVs
    dfs = []
    for path in SCORE_FILES.values():
        df = pd.read_csv(path)
        dfs.append(df.set_index("row_id"))

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")
    merged = merged.reset_index()

    # Convert metric columns to numeric
    for col in METRIC_COLS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Parse row_id into components
    parsed = merged["row_id"].apply(parse_row_id)
    merged["model"] = [p[0] for p in parsed]
    merged["prompt_style"] = [p[1] for p in parsed]
    merged["doc_id"] = [p[2] for p in parsed]
    merged["rep"] = [p[3] for p in parsed]

    # Cross-perturbation sensitivity: std across perturbation_1/2/3 per (doc, model, rep)
    perturb_df = merged[merged["prompt_style"].isin(PERTURBATION_STYLES)].copy()
    available_metrics = [c for c in METRIC_COLS if c in merged.columns]

    if not perturb_df.empty:
        perturb_sensitivity = (
            perturb_df.groupby(["model", "doc_id", "rep"])[available_metrics]
            .std()
            .mean(axis=1)
            .reset_index(name="perturbation_sensitivity")
        )
        merged = merged.merge(perturb_sensitivity, on=["model", "doc_id", "rep"], how="left")
    else:
        merged["perturbation_sensitivity"] = None

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Master results: {len(merged)} rows. Output: {OUTPUT_PATH}")

    # Print summary statistics (mean per model x prompt_style)
    print("\nSummary — mean scores per (model, prompt_style):")
    summary = merged.groupby(["model", "prompt_style"])[available_metrics].mean().round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
