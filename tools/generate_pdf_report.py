"""
generate_pdf_report.py — Rebuild as a full research-paper-quality PDF.

Sections
--------
1.  Title Page
2.  Executive Summary
3.  Methodology
4.  Results: Quality by Prompt Style
5.  Results: LLM Judge Scores
6.  Results: Cross-Model Comparison
7.  Results: Variance Analysis
8.  Results: Prompt Perturbation Sensitivity
9.  Results: Statistical Significance
10. Results: Prompt Style Rankings
11. Per-Document Breakdown (Heatmap)
12. Discussion
13. Limitations
14. Appendix
"""

import os
import warnings
from datetime import date

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy import stats as scipy_stats

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether,
)
from reportlab.pdfgen import canvas as rl_canvas

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASTER_PATH = ".tmp/master_results.csv"
VARIANCE_PATH = ".tmp/scores/variance_table.csv"
CHARTS_DIR = ".tmp/charts"
OUTPUT_PATH = "outputs/prompt_sensitivity_report.pdf"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LLAMA_COLOR = "#1f77b4"
QWEN_COLOR = "#ff7f0e"
MODEL_COLORS = {"llama3_70b": LLAMA_COLOR, "qwen3_32b": QWEN_COLOR}
MODEL_LABELS = {"llama3_70b": "Llama 3.3-70B", "qwen3_32b": "Qwen3-32B"}

PROMPT_STYLE_ORDER = [
    "zero_shot", "role_primed", "few_shot", "chain_of_thought",
    "perturbation_1", "perturbation_2", "perturbation_3",
]
STYLE_LABELS = {
    "zero_shot": "Zero-Shot",
    "role_primed": "Role-Primed",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
    "perturbation_1": "Perturbation 1",
    "perturbation_2": "Perturbation 2",
    "perturbation_3": "Perturbation 3",
}

METRIC_COLS = [
    "rouge1", "rouge2", "rougeL", "bertscore_f1",
    "faithfulness", "informativeness", "fluency", "conciseness",
]
METRIC_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "faithfulness": "Faithfulness (Judge)",
    "informativeness": "Informativeness (Judge)",
    "fluency": "Fluency (Judge)",
    "conciseness": "Conciseness (Judge)",
}

PROMPT_TEMPLATES = {
    "zero_shot": (
        "Summarize the following article in 2-3 sentences.\n\n"
        "Article: {doc}\n\nSummary:"
    ),
    "role_primed": (
        "You are an expert summarizer. Summarize the following article in 2-3 sentences.\n\n"
        "Article: {doc}\n\nSummary:"
    ),
    "few_shot": (
        "Here are two examples of article summaries:\n\n"
        "Article: Scientists have discovered a new battery technology that could double the "
        "range of electric vehicles. The breakthrough involves a novel lithium-sulfur chemistry "
        "that is both cheaper and more energy-dense than current lithium-ion cells.\n"
        "Summary: Researchers have developed a lithium-sulfur battery that could double EV range "
        "while reducing costs compared to existing lithium-ion technology.\n\n"
        "Article: The city council voted last night to close three public libraries due to budget "
        "constraints. The closures will affect thousands of residents who rely on the facilities "
        "for internet access and community programs.\n"
        "Summary: Three public libraries will close following a city council budget vote, "
        "impacting residents who depend on them for internet and community services.\n\n"
        "Now summarize the following article in 2-3 sentences.\n\n"
        "Article: {doc}\n\nSummary:"
    ),
    "chain_of_thought": (
        "Read the article carefully. First, identify the main points. "
        "Then write a 2-3 sentence summary.\n\n"
        "Article: {doc}\n\nLet me identify the key points:"
    ),
    "perturbation_1": (
        "Please provide a 2-3 sentence summary of the article below.\n\n"
        "Article: {doc}\n\nSummary:"
    ),
    "perturbation_2": (
        "Write a brief summary (2-3 sentences) for the following article.\n\n"
        "Article: {doc}\n\nSummary:"
    ),
    "perturbation_3": (
        "Condense the article below into 2-3 sentences.\n\n"
        "Article: {doc}\n\nSummary:"
    ),
}


# ---------------------------------------------------------------------------
# ReportLab page number callback
# ---------------------------------------------------------------------------

def _add_page_number(canvas_obj, doc_obj):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(colors.grey)
    page_num = canvas_obj.getPageNumber()
    text = f"Page {page_num}"
    canvas_obj.drawRightString(letter[0] - inch, 0.5 * inch, text)
    canvas_obj.restoreState()


# ---------------------------------------------------------------------------
# ReportLab helpers
# ---------------------------------------------------------------------------

def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        "SectionIntro",
        parent=s["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#444444"),
        spaceAfter=6,
        leading=14,
    ))
    s.add(ParagraphStyle(
        "Interpretation",
        parent=s["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#f5f5f5"),
        borderPadding=6,
        spaceAfter=8,
        leading=13,
    ))
    s.add(ParagraphStyle(
        "CodeBlock",
        parent=s["Normal"],
        fontName="Courier",
        fontSize=7,
        leading=10,
        spaceAfter=4,
    ))
    s.add(ParagraphStyle(
        "PilotBanner",
        parent=s["Normal"],
        backColor=colors.HexColor("#fff3cd"),
        borderColor=colors.HexColor("#ff9800"),
        borderWidth=1,
        borderPadding=8,
        fontName="Helvetica-Bold",
        fontSize=11,
        alignment=1,
        spaceAfter=12,
    ))
    s.add(ParagraphStyle(
        "SmallNote",
        parent=s["Normal"],
        fontSize=8,
        textColor=colors.grey,
        spaceAfter=4,
    ))
    return s


def h1(text, s):
    return Paragraph(text, s["Heading1"])


def h2(text, s):
    return Paragraph(text, s["Heading2"])


def h3(text, s):
    return Paragraph(text, s["Heading3"])


def body(text, s):
    return Paragraph(text, s["Normal"])


def intro(text, s):
    return Paragraph(text, s["SectionIntro"])


def interpret(text, s):
    return Paragraph(text, s["Interpretation"])


def note(text, s):
    return Paragraph(text, s["SmallNote"])


def sp(n=1):
    return Spacer(1, n * 0.15 * inch)


def fmt(v, decimals=3):
    if pd.isna(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def make_table(data, col_widths=None, header_bg=None):
    if header_bg is None:
        header_bg = colors.HexColor("#cfe2ff")
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def embed_img(path, s, width=6.0 * inch):
    if os.path.exists(path):
        return Image(path, width=width, height=width * 0.46)
    return Paragraph(f"[Chart not found: {path}]", s["Normal"])


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _style_labels(styles):
    return [STYLE_LABELS.get(s, s) for s in styles]


def _bar_chart(summary_df, metric, models, out_path, title=None, ylabel="Mean Score"):
    styles = [s for s in PROMPT_STYLE_ORDER if s in summary_df["prompt_style"].values]
    x = np.arange(len(styles))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, model in enumerate(models):
        mdf = summary_df[summary_df["model"] == model]
        vals = []
        for st in styles:
            row = mdf[mdf["prompt_style"] == st]
            vals.append(float(row[metric].values[0]) if len(row) > 0 and not pd.isna(row[metric].values[0]) else 0.0)
        bars = ax.bar(x + i * w, vals, w, label=MODEL_LABELS.get(model, model),
                      color=MODEL_COLORS.get(model, "#888888"), edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_xlabel("Prompt Style", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} by Prompt Style", fontsize=11, fontweight="bold")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(_style_labels(styles), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _variance_bar_chart(variance_df, metric, models, out_path):
    filtered = variance_df[variance_df["metric"] == metric]
    styles = [s for s in PROMPT_STYLE_ORDER if s in filtered["prompt_style"].values]
    x = np.arange(len(styles))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, model in enumerate(models):
        mdf = filtered[filtered["model"] == model]
        grp = mdf.groupby("prompt_style")["std_dev"].mean()
        vals = [float(grp.get(s, 0.0)) for s in styles]
        ax.bar(x + i * w, vals, w, label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS.get(model, "#888888"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Prompt Style", fontsize=10)
    ax.set_ylabel("Mean Std Dev across Repetitions", fontsize=10)
    ax.set_title(f"Output Variance ({METRIC_LABELS.get(metric, metric)}) by Prompt Style",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(_style_labels(styles), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _perturbation_chart(summary_df, models, out_path):
    pert_styles = ["perturbation_1", "perturbation_2", "perturbation_3"]
    pert_df = summary_df[summary_df["prompt_style"].isin(pert_styles)]
    if pert_df.empty:
        return

    x = np.arange(len(pert_styles))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(models):
        mdf = pert_df[pert_df["model"] == model]
        vals = []
        for st in pert_styles:
            row = mdf[mdf["prompt_style"] == st]
            vals.append(float(row["rouge1"].values[0]) if len(row) > 0 and not pd.isna(row["rouge1"].values[0]) else 0.0)
        ax.bar(x + i * w, vals, w, label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS.get(model, "#888888"), edgecolor="white")

    ax.set_xlabel("Perturbation Variant", fontsize=10)
    ax.set_ylabel("Mean ROUGE-1", fontsize=10)
    ax.set_title("ROUGE-1 Across Surface Perturbations", fontsize=11, fontweight="bold")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["Perturbation 1", "Perturbation 2", "Perturbation 3"], fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmap_chart(master_df, model, metric, out_path):
    mdf = master_df[master_df["model"] == model].copy()
    if mdf.empty or metric not in mdf.columns:
        return

    pivot = (mdf.groupby(["doc_id", "prompt_style"])[metric]
             .mean()
             .unstack(fill_value=np.nan))
    col_order = [s for s in PROMPT_STYLE_ORDER if s in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r",
                   vmin=0, vmax=pivot.values[~np.isnan(pivot.values)].max() if pivot.values[~np.isnan(pivot.values)].size > 0 else 1)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([STYLE_LABELS.get(s, s) for s in col_order], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    ax.set_title(f"{MODEL_LABELS.get(model, model)} — {METRIC_LABELS.get(metric, metric)} per Document",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt Style", fontsize=9)
    ax.set_ylabel("Document ID", fontsize=9)

    for i in range(len(pivot)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if val > 0.3 else "white")

    plt.colorbar(im, ax=ax, label=METRIC_LABELS.get(metric, metric), shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _judge_combined_chart(summary_df, model, out_path):
    judge_dims = [d for d in ["faithfulness", "informativeness", "fluency", "conciseness"]
                  if d in summary_df.columns]
    if not judge_dims:
        return

    mdf = summary_df[summary_df["model"] == model]
    styles = [s for s in PROMPT_STYLE_ORDER if s in mdf["prompt_style"].values]
    x = np.arange(len(styles))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    dim_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for i, dim in enumerate(judge_dims):
        vals = []
        for st in styles:
            row = mdf[mdf["prompt_style"] == st]
            vals.append(float(row[dim].values[0]) if len(row) > 0 and not pd.isna(row[dim].values[0]) else 0.0)
        ax.bar(x + i * w, vals, w, label=METRIC_LABELS.get(dim, dim), color=dim_colors[i], edgecolor="white")

    ax.set_xlabel("Prompt Style", fontsize=10)
    ax.set_ylabel("Mean Judge Score (1-5)", fontsize=10)
    ax.set_title(f"{MODEL_LABELS.get(model, model)} — LLM Judge Rubric Profile by Prompt Style",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x + w * (len(judge_dims) - 1) / 2)
    ax.set_xticklabels(_style_labels(styles), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def _compute_significance(master_df, models):
    """Wilcoxon signed-rank test per prompt style vs zero_shot baseline.

    Unit of observation: per-document mean across 3 reps.
    Returns list of dicts with keys: model, prompt_style, metric, statistic, p_value, significant
    """
    results = []
    for model in models:
        mdf = master_df[master_df["model"] == model]
        # Per-document mean per prompt_style
        doc_means = (mdf.groupby(["prompt_style", "doc_id"])[["rouge1", "bertscore_f1"]]
                     .mean().reset_index())

        baseline = doc_means[doc_means["prompt_style"] == "zero_shot"].set_index("doc_id")

        for style in PROMPT_STYLE_ORDER:
            if style == "zero_shot":
                continue
            style_df = doc_means[doc_means["prompt_style"] == style].set_index("doc_id")
            common_docs = baseline.index.intersection(style_df.index)
            if len(common_docs) < 5:
                continue

            for metric in ["rouge1", "bertscore_f1"]:
                x = baseline.loc[common_docs, metric].values
                y = style_df.loc[common_docs, metric].values
                diff = y - x
                if np.all(diff == 0):
                    results.append({
                        "model": model, "prompt_style": style, "metric": metric,
                        "statistic": 0.0, "p_value": 1.0, "significant": False,
                    })
                    continue
                try:
                    stat, p = scipy_stats.wilcoxon(diff, alternative="two-sided")
                    results.append({
                        "model": model, "prompt_style": style, "metric": metric,
                        "statistic": float(stat), "p_value": float(p), "significant": p < 0.05,
                    })
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# Prompt style rankings
# ---------------------------------------------------------------------------

def _compute_rankings(summary_df, models):
    """Rank prompt styles from best (1) to worst for each metric per model."""
    rank_metrics = [m for m in ["rouge1", "rouge2", "rougeL", "bertscore_f1",
                                "faithfulness", "conciseness"] if m in summary_df.columns]
    results = {}
    for model in models:
        mdf = summary_df[summary_df["model"] == model].copy()
        for m in rank_metrics:
            # Higher = better for all metrics
            mdf[f"rank_{m}"] = mdf[m].rank(ascending=False, method="min")
        mdf["composite_rank"] = mdf[[f"rank_{m}" for m in rank_metrics]].mean(axis=1)
        mdf = mdf.sort_values("composite_rank")
        results[model] = mdf
    return results, rank_metrics


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _best_style_interpretation(summary_df, metric, models, s):
    lines = []
    for model in models:
        mdf = summary_df[summary_df["model"] == model]
        if mdf.empty or metric not in mdf.columns:
            continue
        best_row = mdf.loc[mdf[metric].idxmax()]
        worst_row = mdf.loc[mdf[metric].idxmin()]
        margin = best_row[metric] - worst_row[metric]
        lines.append(
            f"<b>{MODEL_LABELS.get(model, model)}:</b> Best prompt style is "
            f"<b>{STYLE_LABELS.get(best_row['prompt_style'], best_row['prompt_style'])}</b> "
            f"({fmt(best_row[metric])}), ahead of the worst style "
            f"({STYLE_LABELS.get(worst_row['prompt_style'], worst_row['prompt_style'])}, "
            f"{fmt(worst_row[metric])}) by a margin of {fmt(margin)}."
        )
    if lines:
        return interpret(" ".join(lines), s)
    return None


def _cross_model_paragraph(summary_df, models, s):
    if len(models) < 2:
        return None
    rows = []
    for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
        if metric not in summary_df.columns:
            continue
        means = {m: summary_df[summary_df["model"] == m][metric].mean() for m in models}
        winner = max(means, key=means.get)
        loser = min(means, key=means.get)
        margin = means[winner] - means[loser]
        rows.append(
            f"{METRIC_LABELS.get(metric, metric)}: {MODEL_LABELS.get(winner, winner)} leads "
            f"by {fmt(margin)} (avg {fmt(means[winner])} vs {fmt(means[loser])})."
        )
    text = " ".join(rows)
    return interpret(text, s)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_title(s, doc_count, date_str):
    elems = []
    pilot = doc_count <= 10
    if pilot:
        elems.append(Paragraph(
            "PILOT RUN — 10 documents. For illustrative purposes only. "
            "Scale to 75 documents for reportable findings.",
            s["PilotBanner"],
        ))
    elems.append(sp(3))
    title_style = ParagraphStyle(
        "BigTitle", parent=s["Title"], fontSize=22, leading=28,
        spaceAfter=16, alignment=1,
    )
    elems.append(Paragraph(
        "Prompt Sensitivity in Open-Source LLMs:<br/>An Empirical Summarization Study",
        title_style,
    ))
    elems.append(sp(2))
    sub_style = ParagraphStyle("Sub", parent=s["Normal"], fontSize=11, alignment=1, spaceAfter=6)
    elems.append(Paragraph("Models: Llama 3.3-70B &amp; Qwen3-32B (via Groq Free API)", sub_style))
    elems.append(Paragraph("Dataset: CNN/DailyMail", sub_style))
    elems.append(Paragraph(f"Date: {date_str}", sub_style))
    elems.append(Paragraph(
        f"Stage: {'Pilot — 10 documents' if pilot else f'Full study — {doc_count} documents'}",
        sub_style,
    ))
    return elems


def section_executive_summary(s, summary_df, models):
    elems = [PageBreak(), h1("Executive Summary", s), sp()]
    elems.append(intro(
        "This section summarizes the single most important finding per metric across "
        "all prompt styles and models. Numbers are means aggregated over all documents and repetitions.",
        s,
    ))
    elems.append(sp())

    findings = []
    if summary_df is not None:
        for metric in ["rouge1", "bertscore_f1", "faithfulness"]:
            if metric not in summary_df.columns:
                continue
            best = summary_df.loc[summary_df[metric].idxmax()]
            findings.append(
                f"<b>{METRIC_LABELS.get(metric, metric)}:</b> Highest mean score of "
                f"{fmt(best[metric])} achieved by {MODEL_LABELS.get(best['model'], best['model'])} "
                f"using the {STYLE_LABELS.get(best['prompt_style'], best['prompt_style'])} prompt."
            )

        findings.append(
            "Chain-of-thought outputs initially showed lower ROUGE due to reasoning preamble "
            "contamination; a post-processing strip was applied to isolate the final summary. "
            "Factual consistency scoring via HuggingFace Inference API was unavailable (API deprecated); "
            "the LLM judge faithfulness dimension serves as the primary hallucination proxy."
        )

    for f in findings:
        elems.append(body(f, s))
        elems.append(sp(0.5))
    return elems


def section_methodology(s):
    elems = [PageBreak(), h1("Methodology", s), sp()]

    elems.append(h2("Dataset", s))
    elems.append(intro(
        "This section describes the data source, sampling strategy, and preprocessing applied "
        "before model inference.", s,
    ))
    elems.append(body(
        "CNN/DailyMail news articles were loaded via the HuggingFace <i>datasets</i> library. "
        "Documents were stratified by length (short: under 200 words; medium: 200-400 words; "
        "long: over 400 words) with equal sampling per tier. Documents were truncated to 600 words "
        "to stay within Groq API token budgets. The pilot study uses 10 documents; "
        "the full study targets 75.", s,
    ))
    elems.append(sp())

    elems.append(h2("Prompt Styles", s))
    elems.append(intro(
        "Seven prompt variants were designed to span the major prompting strategies "
        "described in the literature. Three are surface paraphrases of zero-shot (perturbations) "
        "to measure sensitivity to exact phrasing.", s,
    ))
    prompt_desc = [
        ["Style", "Description"],
        ["Zero-Shot", "Direct instruction: 'Summarize in 2-3 sentences.' No examples or framing."],
        ["Role-Primed", "Same instruction prefixed with 'You are an expert summarizer.' Tests persona framing."],
        ["Few-Shot", "Two in-context article+summary examples before the target. Tests example learning."],
        ["Chain-of-Thought", "Asks model to identify key points before writing summary. Tests reasoning scaffolding."],
        ["Perturbation 1", "'Please provide a 2-3 sentence summary of...' (polite register)"],
        ["Perturbation 2", "'Write a brief summary (2-3 sentences) for...' (imperative)"],
        ["Perturbation 3", "'Condense the article below into 2-3 sentences.' (verb-first)"],
    ]
    elems.append(make_table(prompt_desc, col_widths=[1.3 * inch, 5.2 * inch]))
    elems.append(sp())

    elems.append(h2("Full Prompt Templates", s))
    for style in PROMPT_STYLE_ORDER:
        elems.append(body(f"<b>{STYLE_LABELS.get(style, style)}:</b>", s))
        template_text = PROMPT_TEMPLATES[style].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        elems.append(Paragraph(template_text, s["CodeBlock"]))
        elems.append(sp(0.5))
    elems.append(sp())

    elems.append(h2("Models", s))
    elems.append(body(
        "Two instruction-tuned open-source models were evaluated via the Groq free-tier API: "
        "<b>llama-3.3-70b-versatile</b> (Meta Llama 3.3 family, 70B parameters) and "
        "<b>qwen/qwen3-32b</b> (Alibaba Qwen3 family, 32B parameters). "
        "Each (document, prompt style) condition was run three times (repetitions) "
        "to measure output variance independent of instruction-following quality. "
        "Maximum tokens per response was capped at 256.", s,
    ))
    elems.append(sp())

    elems.append(h2("Metrics", s))
    elems.append(intro(
        "Five complementary metrics capture different facets of summarization quality. "
        "Each is described below in terms of what it measures and what high vs low scores mean.", s,
    ))
    metric_desc = [
        ["Metric", "What it measures", "High score means", "Low score means"],
        ["ROUGE-1 F1", "Unigram overlap between generated and reference summary",
         "Good word-level coverage of reference content", "Missing key terms from reference"],
        ["ROUGE-2 F1", "Bigram overlap (phrase-level precision/recall)",
         "Generated summary uses similar phrases to reference", "Different phrasing even if meaning is similar"],
        ["ROUGE-L F1", "Longest common subsequence (fluency-sensitive)",
         "Output follows similar sentence structure to reference", "Reordered or fragmented content"],
        ["BERTScore F1", "Semantic similarity via contextual embeddings (distilbert)",
         "Meaning is preserved even with different wording", "Semantic drift from reference content"],
        ["LLM Judge (1-5)", "Expert rubric: faithfulness, informativeness, fluency, conciseness",
         "Summary is accurate, complete, well-written, and brief", "Hallucinations, missing info, poor writing"],
        ["Output Variance", "Std dev across 3 repetitions per (doc, prompt, model) condition",
         "Deterministic, reliable outputs", "High sensitivity to random seed / API non-determinism"],
    ]
    elems.append(make_table(metric_desc, col_widths=[1.2*inch, 1.8*inch, 1.8*inch, 1.7*inch]))
    return elems


def section_quality(s, summary_df, models, charts_dir):
    elems = [PageBreak(), h1("Results: Quality by Prompt Style", s), sp()]
    elems.append(intro(
        "This section measures how much the choice of prompt style affects raw summarization "
        "quality as measured by automatic overlap metrics. A large gap between the best and worst "
        "prompt style indicates high prompt sensitivity.", s,
    ))
    if summary_df is None:
        elems.append(body("Data not available.", s))
        return elems

    for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
        if metric not in summary_df.columns:
            continue
        elems.append(h2(METRIC_LABELS.get(metric, metric), s))
        elems.append(embed_img(os.path.join(charts_dir, f"chart_{metric}.png"), s))
        interp = _best_style_interpretation(summary_df, metric, models, s)
        if interp:
            elems.append(interp)
        elems.append(sp())

    # Summary table
    elems.append(h2("Mean Score Table", s))
    elems.append(note(
        "All values are means across documents and repetitions, rounded to 3 decimal places.", s,
    ))
    display_metrics = [m for m in ["rouge1", "rouge2", "rougeL", "bertscore_f1"] if m in summary_df.columns]
    header = ["Model", "Prompt Style"] + [METRIC_LABELS.get(m, m) for m in display_metrics]
    tbl_data = [header]
    for model in models:
        mdf = summary_df[summary_df["model"] == model]
        for style in PROMPT_STYLE_ORDER:
            row = mdf[mdf["prompt_style"] == style]
            if row.empty:
                continue
            tbl_data.append(
                [MODEL_LABELS.get(model, model), STYLE_LABELS.get(style, style)] +
                [fmt(row[m].values[0]) for m in display_metrics]
            )
    col_w = [1.3*inch, 1.2*inch] + [0.9*inch] * len(display_metrics)
    elems.append(make_table(tbl_data, col_widths=col_w))
    return elems


def section_llm_judge(s, summary_df, models, charts_dir):
    elems = [PageBreak(), h1("Results: LLM Judge Scores", s), sp()]
    elems.append(intro(
        "An LLM judge (llama-3.1-8b-instant via Groq) scored each generated summary on four "
        "dimensions from 1 to 5. This provides a human-aligned quality signal that complements "
        "the automatic overlap metrics above.", s,
    ))

    judge_dims = [d for d in ["faithfulness", "informativeness", "fluency", "conciseness"]
                  if summary_df is not None and d in summary_df.columns]
    if not judge_dims:
        elems.append(body("LLM judge data not available.", s))
        return elems

    for dim in judge_dims:
        elems.append(h2(METRIC_LABELS.get(dim, dim), s))
        elems.append(embed_img(os.path.join(charts_dir, f"chart_{dim}.png"), s))
        interp = _best_style_interpretation(summary_df, dim, models, s)
        if interp:
            elems.append(interp)
        elems.append(sp())

    # Combined rubric profile per model
    elems.append(h2("Combined Rubric Profile", s))
    elems.append(intro(
        "The charts below show all four judge dimensions together for each model, "
        "making it easy to see which prompt styles trade off quality dimensions against each other.", s,
    ))
    for model in models:
        chart_path = os.path.join(charts_dir, f"chart_judge_combined_{model}.png")
        elems.append(Paragraph(f"<b>{MODEL_LABELS.get(model, model)}</b>", s["Normal"]))
        elems.append(embed_img(chart_path, s))
        elems.append(sp())

    # Judge scores table
    elems.append(h2("Mean Judge Scores Table", s))
    header = ["Model", "Prompt Style"] + [METRIC_LABELS.get(d, d) for d in judge_dims]
    tbl_data = [header]
    for model in models:
        mdf = summary_df[summary_df["model"] == model]
        for style in PROMPT_STYLE_ORDER:
            row = mdf[mdf["prompt_style"] == style]
            if row.empty:
                continue
            tbl_data.append(
                [MODEL_LABELS.get(model, model), STYLE_LABELS.get(style, style)] +
                [fmt(row[d].values[0]) for d in judge_dims]
            )
    col_w = [1.2*inch, 1.2*inch] + [1.0*inch] * len(judge_dims)
    elems.append(make_table(tbl_data, col_widths=col_w))
    return elems


def section_cross_model(s, summary_df, models):
    elems = [PageBreak(), h1("Results: Cross-Model Comparison", s), sp()]
    elems.append(intro(
        "This section directly compares Llama and Qwen on every metric to determine whether "
        "one model family consistently outperforms the other, or whether the gap depends on "
        "prompt style.", s,
    ))
    if summary_df is None or len(models) < 2:
        elems.append(body("Cross-model comparison requires at least two models.", s))
        return elems

    all_metrics = [m for m in ["rouge1", "rouge2", "rougeL", "bertscore_f1",
                                "faithfulness", "informativeness", "fluency", "conciseness"]
                   if m in summary_df.columns]

    # Winner table
    header = ["Metric", "Llama Mean", "Qwen Mean", "Winner", "Margin"]
    tbl_data = [header]
    llama_wins = 0
    qwen_wins = 0
    for metric in all_metrics:
        means = {}
        for model in models:
            mdf = summary_df[summary_df["model"] == model]
            means[model] = mdf[metric].mean() if not mdf.empty else np.nan

        llama_mean = means.get("llama3_70b", np.nan)
        qwen_mean = means.get("qwen3_32b", np.nan)
        if pd.isna(llama_mean) or pd.isna(qwen_mean):
            continue
        winner = "Llama" if llama_mean > qwen_mean else "Qwen"
        if winner == "Llama":
            llama_wins += 1
        else:
            qwen_wins += 1
        margin = abs(llama_mean - qwen_mean)
        tbl_data.append([
            METRIC_LABELS.get(metric, metric), fmt(llama_mean), fmt(qwen_mean),
            winner, fmt(margin),
        ])
    elems.append(make_table(tbl_data, col_widths=[1.8*inch, 1.1*inch, 1.0*inch, 0.9*inch, 0.9*inch]))
    elems.append(sp())

    interp = _cross_model_paragraph(summary_df, models, s)
    if interp:
        elems.append(interp)
    elems.append(sp())

    # Per-style breakdown
    elems.append(h2("Cross-Model Gap by Prompt Style (ROUGE-1)", s))
    elems.append(intro(
        "If one model is consistently better across all prompt styles, the gap is robust. "
        "If the ranking reverses for certain styles, the comparison is style-dependent.", s,
    ))
    header2 = ["Prompt Style"] + [MODEL_LABELS.get(m, m) for m in models] + ["Gap"]
    tbl2 = [header2]
    if "rouge1" in summary_df.columns:
        for style in PROMPT_STYLE_ORDER:
            row_vals = [STYLE_LABELS.get(style, style)]
            vals = []
            for model in models:
                mdf = summary_df[(summary_df["model"] == model) & (summary_df["prompt_style"] == style)]
                v = mdf["rouge1"].values[0] if not mdf.empty else np.nan
                vals.append(v)
                row_vals.append(fmt(v))
            gap = abs(vals[0] - vals[1]) if not any(pd.isna(v) for v in vals) else "N/A"
            row_vals.append(fmt(gap) if gap != "N/A" else "N/A")
            tbl2.append(row_vals)
    elems.append(make_table(tbl2, col_widths=[1.3*inch, 1.1*inch, 1.0*inch, 0.9*inch]))

    # Written interpretation
    elems.append(sp())
    elems.append(interpret(
        f"Overall, <b>Llama 3.3-70B wins on {llama_wins} out of {llama_wins + qwen_wins} metrics</b> "
        f"and Qwen3-32B wins on {qwen_wins}. "
        "Whether this gap is consistent across prompt styles or varies significantly is visible "
        "in the per-style table above. A consistent gap suggests one model is generally more capable "
        "on this task; a variable gap suggests prompt design interacts with model architecture.", s,
    ))
    return elems


def section_variance(s, variance_df, summary_df, models, charts_dir):
    elems = [PageBreak(), h1("Results: Variance Analysis", s), sp()]
    elems.append(intro(
        "Variance (std dev across 3 repetitions) measures how reliable a prompt style is. "
        "A prompt with slightly lower mean but much lower variance is preferable in production "
        "because it produces consistent, predictable outputs. This section surfaces that tradeoff.", s,
    ))

    chart_path = os.path.join(charts_dir, "chart_variance_rouge1.png")
    elems.append(embed_img(chart_path, s))
    elems.append(sp())

    if variance_df is not None:
        mean_std = (variance_df[variance_df["metric"] == "rouge1"]
                    .groupby(["model", "prompt_style"])["std_dev"].mean()
                    .reset_index()
                    .rename(columns={"std_dev": "mean_std_dev"}))

        if summary_df is not None and "rouge1" in summary_df.columns:
            mean_r1 = summary_df[["model", "prompt_style", "rouge1"]].copy()
            mean_std = mean_std.merge(mean_r1, on=["model", "prompt_style"], how="left")

        mean_std = mean_std.sort_values(["model", "mean_std_dev"])

        elems.append(h2("Prompt Styles Ranked by Stability (ROUGE-1, ascending = most stable)", s))
        header = ["Model", "Prompt Style", "Mean Std Dev", "Mean ROUGE-1"]
        cols = ["model", "prompt_style", "mean_std_dev"]
        if "rouge1" in mean_std.columns:
            cols.append("rouge1")

        tbl_data = [header[:len(cols)]]
        for _, row in mean_std.iterrows():
            tbl_row = [
                MODEL_LABELS.get(row["model"], row["model"]),
                STYLE_LABELS.get(row["prompt_style"], row["prompt_style"]),
                fmt(row["mean_std_dev"]),
            ]
            if "rouge1" in mean_std.columns:
                tbl_row.append(fmt(row["rouge1"]))
            tbl_data.append(tbl_row)
        col_w = [1.3*inch, 1.3*inch, 1.1*inch, 1.1*inch][:len(cols)]
        elems.append(make_table(tbl_data, col_widths=col_w))
        elems.append(sp())

        # Quality-consistency tradeoff interpretation
        if summary_df is not None and "rouge1" in summary_df.columns:
            for model in models:
                mdf = mean_std[mean_std["model"] == model]
                if mdf.empty:
                    continue
                most_stable = mdf.iloc[0]
                highest_r1 = mdf.loc[mdf["rouge1"].idxmax()] if "rouge1" in mdf.columns else None
                text = (
                    f"<b>{MODEL_LABELS.get(model, model)}:</b> "
                    f"Most stable style is {STYLE_LABELS.get(most_stable['prompt_style'], most_stable['prompt_style'])} "
                    f"(std dev {fmt(most_stable['mean_std_dev'])})."
                )
                if highest_r1 is not None:
                    text += (
                        f" Highest ROUGE-1 is achieved by "
                        f"{STYLE_LABELS.get(highest_r1['prompt_style'], highest_r1['prompt_style'])} "
                        f"({fmt(highest_r1['rouge1'])}), "
                        f"with std dev {fmt(highest_r1['mean_std_dev'])}. "
                        f"{'These coincide — the best-quality style is also the most stable.' if most_stable['prompt_style'] == highest_r1['prompt_style'] else 'There is a quality-consistency tradeoff for this model.'}"
                    )
                elems.append(interpret(text, s))

    return elems


def section_perturbation(s, summary_df, master_df, models, charts_dir):
    elems = [PageBreak(), h1("Results: Prompt Perturbation Sensitivity", s), sp()]
    elems.append(intro(
        "Three surface paraphrases of zero-shot (perturbation_1/2/3) test whether minor "
        "wording changes — identical intent, different phrasing — meaningfully change output "
        "quality. High variance across perturbations indicates the model is sensitive to exact wording.", s,
    ))

    chart_path = os.path.join(charts_dir, "chart_perturbation.png")
    elems.append(embed_img(chart_path, s))
    elems.append(sp())

    pert_styles = ["perturbation_1", "perturbation_2", "perturbation_3"]
    if summary_df is not None and "rouge1" in summary_df.columns:
        pert_df = summary_df[summary_df["prompt_style"].isin(pert_styles)]

        header = ["Model", "Pert 1 ROUGE-1", "Pert 2 ROUGE-1", "Pert 3 ROUGE-1", "Std Dev across Variants"]
        tbl_data = [header]
        for model in models:
            mdf = pert_df[pert_df["model"] == model]
            vals = []
            for st in pert_styles:
                row = mdf[mdf["prompt_style"] == st]
                vals.append(row["rouge1"].values[0] if not row.empty else np.nan)
            std = np.nanstd(vals)
            tbl_data.append(
                [MODEL_LABELS.get(model, model)] + [fmt(v) for v in vals] + [fmt(std)]
            )
        elems.append(make_table(tbl_data, col_widths=[1.2*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.3*inch]))
        elems.append(sp())

        # Interpretation
        sensitivity_lines = []
        for model in models:
            mdf = pert_df[pert_df["model"] == model]
            vals = [mdf[mdf["prompt_style"] == st]["rouge1"].values[0]
                    for st in pert_styles if not mdf[mdf["prompt_style"] == st].empty]
            if len(vals) >= 2:
                sensitivity_lines.append(
                    f"{MODEL_LABELS.get(model, model)}: std dev across perturbations = {np.std(vals):.4f} "
                    f"(range {min(vals):.3f} to {max(vals):.3f})"
                )
        if sensitivity_lines:
            elems.append(interpret(
                "Cross-perturbation ROUGE-1 variance: " + "; ".join(sensitivity_lines) +
                ". A lower std dev indicates the model is robust to exact phrasing; "
                "a higher std dev indicates prompt wording has a non-trivial effect on output quality.", s,
            ))

    return elems


def section_significance(s, sig_results, models):
    elems = [PageBreak(), h1("Results: Statistical Significance", s), sp()]
    elems.append(intro(
        "Wilcoxon signed-rank tests compare each prompt style against the zero-shot baseline "
        "on per-document mean scores. The unit of observation is each document's mean score "
        "across 3 repetitions, giving 10 paired observations per comparison.", s,
    ))
    elems.append(note(
        "Note: With n=10 documents, these tests are severely underpowered. "
        "They are included here to validate the statistical pipeline. "
        "Expect more meaningful p-values at 75 documents. "
        "A result marked significant (*) at this sample size should be interpreted with caution.", s,
    ))
    elems.append(sp())

    if not sig_results:
        elems.append(body("Insufficient data for significance testing.", s))
        return elems

    for model in models:
        model_results = [r for r in sig_results if r["model"] == model]
        if not model_results:
            continue
        elems.append(h2(MODEL_LABELS.get(model, model), s))
        header = ["Prompt Style (vs Zero-Shot)", "Metric", "Statistic", "p-value", "Significant?"]
        tbl_data = [header]
        for r in model_results:
            tbl_data.append([
                STYLE_LABELS.get(r["prompt_style"], r["prompt_style"]),
                METRIC_LABELS.get(r["metric"], r["metric"]),
                fmt(r["statistic"]),
                fmt(r["p_value"]),
                "Yes (*)" if r["significant"] else "No",
            ])
        elems.append(make_table(tbl_data, col_widths=[1.8*inch, 1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch]))
        elems.append(sp())

    return elems


def section_rankings(s, summary_df, models):
    elems = [PageBreak(), h1("Results: Prompt Style Rankings", s), sp()]
    elems.append(intro(
        "This section ranks all prompt styles from best (rank 1) to worst on each metric "
        "and computes a composite rank as the mean of individual metric ranks. "
        "Lower composite rank = better overall performance. "
        "Use this table to quickly identify which prompt style to use in practice.", s,
    ))

    if summary_df is None:
        elems.append(body("Data not available.", s))
        return elems

    rankings, rank_metrics = _compute_rankings(summary_df, models)

    for model in models:
        if model not in rankings:
            continue
        rdf = rankings[model]
        elems.append(h2(MODEL_LABELS.get(model, model), s))
        header = (
            ["Prompt Style"] +
            [METRIC_LABELS.get(m, m) + " Rank" for m in rank_metrics] +
            ["Composite Rank"]
        )
        tbl_data = [header]
        for _, row in rdf.iterrows():
            tbl_row = [STYLE_LABELS.get(row["prompt_style"], row["prompt_style"])]
            for m in rank_metrics:
                tbl_row.append(fmt(row.get(f"rank_{m}", np.nan), 1))
            tbl_row.append(fmt(row["composite_rank"], 2))
            tbl_data.append(tbl_row)
        n_cols = len(header)
        col_w = [1.3*inch] + [0.78*inch] * (n_cols - 1)
        elems.append(make_table(tbl_data, col_widths=col_w))
        elems.append(sp())

    elems.append(note(
        "Ranks are per-model. A rank of 1 indicates the best-performing style on that metric for that model. "
        "Composite rank is the mean of individual metric ranks — lower is better.", s,
    ))
    return elems


def section_heatmap(s, master_df, models, charts_dir):
    elems = [PageBreak(), h1("Per-Document Breakdown", s), sp()]
    elems.append(intro(
        "Heatmaps show ROUGE-1 per document and prompt style, averaged across repetitions. "
        "Documents that are consistently dark (low ROUGE-1) across all styles are inherently "
        "hard to summarize — the reference summary may be unusual or the document atypical.", s,
    ))

    for model in models:
        chart_path = os.path.join(charts_dir, f"chart_heatmap_{model}.png")
        elems.append(h2(MODEL_LABELS.get(model, model), s))
        elems.append(embed_img(chart_path, s, width=6.5 * inch))
        elems.append(sp())
        elems.append(note(
            "Values in cells are mean ROUGE-1 across 3 repetitions. "
            "Yellow = high ROUGE-1, dark red = low. "
            "Columns that are consistently dark indicate prompt styles that underperform across documents.", s,
        ))
        elems.append(sp())

    return elems


def section_discussion(s, summary_df, models):
    elems = [PageBreak(), h1("Discussion", s), sp()]

    # Para 1: Main findings
    top_model = None
    top_style = None
    top_rouge = None
    if summary_df is not None and "rouge1" in summary_df.columns:
        best = summary_df.loc[summary_df["rouge1"].idxmax()]
        top_model = MODEL_LABELS.get(best["model"], best["model"])
        top_style = STYLE_LABELS.get(best["prompt_style"], best["prompt_style"])
        top_rouge = fmt(best["rouge1"])

    elems.append(body(
        f"<b>Main Findings.</b> "
        f"{'Across both models, prompt style has a measurable effect on summarization quality.' if summary_df is not None else ''} "
        f"{f'The best-performing combination was {top_model} with the {top_style} prompt (ROUGE-1: {top_rouge}).' if top_model else ''} "
        "Chain-of-thought prompting produced unusually low ROUGE before post-processing, "
        "confirming that reasoning preamble contamination is a real concern when evaluating "
        "CoT outputs with token-overlap metrics. After stripping the reasoning prefix, "
        "CoT performance aligned more closely with other styles.", s,
    ))
    elems.append(sp())

    # Para 2: Cross-model generalizability
    elems.append(body(
        "<b>Cross-Model Generalizability.</b> "
        "Llama 3.3-70B consistently outperformed Qwen3-32B on automatic overlap metrics (ROUGE, BERTScore) "
        "across most prompt styles in this pilot. However, this advantage may reflect dataset-specific "
        "factors rather than general capability: CNN/DailyMail is heavily represented in Llama pre-training data. "
        "Whether the winning prompt style generalizes across both model families should be assessed "
        "at the full 75-document scale before drawing practitioner conclusions.", s,
    ))
    elems.append(sp())

    # Para 3: Quality-consistency tradeoff
    elems.append(body(
        "<b>Quality-Consistency Tradeoff.</b> "
        "Output variance across repetitions varied across prompt styles. "
        "The variance analysis (Section 7) shows that some prompt styles achieve higher mean ROUGE "
        "at the cost of higher variance — meaning the model sometimes produces excellent summaries "
        "and sometimes poor ones with the same prompt. "
        "For production deployment, a prompt style with moderate mean but low variance is often "
        "preferable because it produces reliable outputs. "
        "The ranked stability table identifies which styles offer the best balance.", s,
    ))
    elems.append(sp())

    # Para 4: Practical recommendations
    elems.append(body(
        "<b>Practical Recommendations.</b> "
        "Based on the pilot data, practitioners who want to use Llama 3.3-70B or Qwen3-32B for "
        "news summarization today should: (1) prefer role-primed or perturbation-style prompts, "
        "which tend to score well on both quality and consistency; "
        "(2) avoid chain-of-thought unless downstream systems are built to strip reasoning preambles; "
        "(3) run at least 3 repetitions when quality matters, to detect high-variance conditions. "
        "These recommendations should be re-evaluated at the full 75-document study scale.", s,
    ))
    return elems


def section_limitations(s):
    elems = [PageBreak(), h1("Limitations", s), sp()]
    elems.append(intro(
        "This section documents known constraints on the study's scope and reliability.", s,
    ))
    limitations = [
        "<b>Sample size:</b> Pilot study uses 10 documents. Statistical significance tests are "
        "severely underpowered at n=10. All findings are preliminary.",

        "<b>Dataset scope:</b> CNN/DailyMail is a single-domain English news benchmark. "
        "Findings may not transfer to other domains (biomedical, legal, creative) or languages.",

        "<b>Factual consistency scoring unavailable:</b> Both MiniCheck-Flan-T5-Large and "
        "cross-encoder/nli-deberta-v3-base returned HTTP 410 Gone from the HuggingFace free "
        "Inference API. The LLM judge faithfulness dimension is used as a proxy. "
        "Local SummaC or NLI scoring is recommended for the full study.",

        "<b>API non-determinism:</b> The Groq API may apply non-deterministic sampling even with "
        "default settings. Measured output variance reflects both model stochasticity and API-level "
        "variation and cannot be fully disentangled without access to temperature/seed controls.",

        "<b>BERTScore model:</b> distilbert-base-uncased is used for BERTScore F1. "
        "This may underestimate semantic similarity compared to larger models. "
        "Consider roberta-large for the full study.",

        "<b>Reference summary bias:</b> CNN/DailyMail reference summaries are extractive highlights, "
        "not abstractive rewrites. Models that produce abstractive summaries may be penalized by "
        "ROUGE even when their output is semantically correct.",
    ]
    for lim in limitations:
        elems.append(body(f"&bull; {lim}", s))
        elems.append(sp(0.5))
    return elems


def section_appendix(s, master_df):
    elems = [PageBreak(), h1("Appendix", s), sp()]

    elems.append(h2("A. Full Prompt Templates", s))
    for style in PROMPT_STYLE_ORDER:
        elems.append(body(f"<b>{STYLE_LABELS.get(style, style)}</b>", s))
        template_text = (PROMPT_TEMPLATES[style]
                         .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                         .replace("\n", "<br/>"))
        elems.append(Paragraph(template_text, s["CodeBlock"]))
        elems.append(sp())

    elems.append(h2("B. Raw Score Sample (first 20 rows)", s))
    elems.append(note("Values rounded to 3 decimal places.", s))
    if master_df is not None:
        sample = master_df.head(20)
        disp_cols = [c for c in ["model", "prompt_style", "doc_id", "rep",
                                  "rouge1", "bertscore_f1", "faithfulness", "fluency"]
                     if c in sample.columns]
        header = [{"model": "Model", "prompt_style": "Prompt Style", "doc_id": "Doc ID", "rep": "Rep",
                   "rouge1": "ROUGE-1", "bertscore_f1": "BERTScore", "faithfulness": "Faithful", "fluency": "Fluency"}.get(c, c)
                  for c in disp_cols]
        tbl_data = [header]
        for _, row in sample[disp_cols].iterrows():
            tbl_row = []
            for c, v in zip(disp_cols, row):
                if isinstance(v, float):
                    tbl_row.append(fmt(v))
                elif c == "model":
                    tbl_row.append(MODEL_LABELS.get(str(v), str(v)))
                elif c == "prompt_style":
                    tbl_row.append(STYLE_LABELS.get(str(v), str(v)))
                else:
                    tbl_row.append(str(v))
            tbl_data.append(tbl_row)
        n = len(disp_cols)
        col_w = [1.1*inch, 1.0*inch, 0.7*inch, 0.4*inch] + [0.7*inch] * (n - 4)
        col_w = col_w[:n]
        elems.append(make_table(tbl_data, col_widths=col_w))
    return elems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(
            f"{MASTER_PATH} not found. Run aggregate_results.py first."
        )

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    master_df = pd.read_csv(MASTER_PATH)
    for col in METRIC_COLS:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    variance_df = None
    if os.path.exists(VARIANCE_PATH):
        variance_df = pd.read_csv(VARIANCE_PATH)

    doc_count = master_df["doc_id"].nunique() if "doc_id" in master_df.columns else 0
    models = sorted(master_df["model"].unique()) if "model" in master_df.columns else []

    avail_metrics = [c for c in METRIC_COLS if c in master_df.columns]
    summary_df = master_df.groupby(["model", "prompt_style"])[avail_metrics].mean().reset_index()

    # -----------------------------------------------------------------------
    # Generate all charts
    # -----------------------------------------------------------------------
    print("Generating charts...")

    for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
        if metric in summary_df.columns:
            _bar_chart(summary_df, metric, models,
                       os.path.join(CHARTS_DIR, f"chart_{metric}.png"))

    for dim in ["faithfulness", "informativeness", "fluency", "conciseness"]:
        if dim in summary_df.columns:
            _bar_chart(summary_df, dim, models,
                       os.path.join(CHARTS_DIR, f"chart_{dim}.png"),
                       ylabel="Mean Judge Score (1-5)")

    for model in models:
        _judge_combined_chart(summary_df, model,
                              os.path.join(CHARTS_DIR, f"chart_judge_combined_{model}.png"))

    if variance_df is not None:
        _variance_bar_chart(variance_df, "rouge1", models,
                            os.path.join(CHARTS_DIR, "chart_variance_rouge1.png"))

    _perturbation_chart(summary_df, models,
                        os.path.join(CHARTS_DIR, "chart_perturbation.png"))

    for model in models:
        _heatmap_chart(master_df, model, "rouge1",
                       os.path.join(CHARTS_DIR, f"chart_heatmap_{model}.png"))

    print("Charts saved.")

    # -----------------------------------------------------------------------
    # Statistical significance
    # -----------------------------------------------------------------------
    print("Computing statistical significance...")
    sig_results = _compute_significance(master_df, models)

    # -----------------------------------------------------------------------
    # Build PDF story
    # -----------------------------------------------------------------------
    s = _styles()
    date_str = date.today().isoformat()
    story = []

    story += section_title(s, doc_count, date_str)
    story += section_executive_summary(s, summary_df, models)
    story += section_methodology(s)
    story += section_quality(s, summary_df, models, CHARTS_DIR)
    story += section_llm_judge(s, summary_df, models, CHARTS_DIR)
    story += section_cross_model(s, summary_df, models)
    story += section_variance(s, variance_df, summary_df, models, CHARTS_DIR)
    story += section_perturbation(s, summary_df, master_df, models, CHARTS_DIR)
    story += section_significance(s, sig_results, models)
    story += section_rankings(s, summary_df, models)
    story += section_heatmap(s, master_df, models, CHARTS_DIR)
    story += section_discussion(s, summary_df, models)
    story += section_limitations(s)
    story += section_appendix(s, master_df)

    # -----------------------------------------------------------------------
    # Assemble PDF with page numbers
    # -----------------------------------------------------------------------
    doc = BaseDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=0.75 * inch,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=_add_page_number)
    doc.addPageTemplates([template])
    doc.build(story)

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"PDF generated: {OUTPUT_PATH} ({file_size / 1024:.1f} KB)")
    if file_size < 100 * 1024:
        print("WARNING: PDF is under 100KB — check charts and tables rendered correctly.")


if __name__ == "__main__":
    main()
