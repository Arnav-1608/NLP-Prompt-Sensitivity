# Workflow 08: PDF Report Generation

## Objective
Generate the final PDF research report with charts, tables, and analysis from the master results table.

## Inputs
- `.tmp/master_results.csv` — produced by Step 9
- `.tmp/scores/variance_table.csv` — produced by Step 8

## Command
```bash
python tools/generate_pdf_report.py
```

## Expected Output
`outputs/prompt_sensitivity_report.pdf` — multi-section research report.

Expected file size: >50KB (the script warns if smaller). Expect ~200–600KB depending on chart resolution and number of tables.

Sections:
1. Title Page (with PILOT RUN banner if doc_count ≤ 10)
2. Executive Summary
3. Methodology
4. Results: Quality by Prompt Style
5. Results: Cross-Model Comparison
6. Results: Variance Analysis
7. Results: Prompt Perturbation Sensitivity
8. Results: Factual Consistency
9. Discussion
10. Appendix

## Edge Cases

**Charts not rendering (blank images in PDF):**
Charts are saved as PNG files in `.tmp/` before the PDF is built. If a chart is missing, the PDF embeds the text `[Chart not found: ...]`. Regenerate only the charts by running `generate_pdf_report.py` again — it regenerates all charts before building the PDF.

**ReportLab Unicode error:**
Do not use Unicode subscript/superscript characters (², ₁, etc.) in any string passed to `Paragraph`. Use ReportLab XML tags: `F<sub>1</sub>`. If you add new text and see a `UnicodeEncodeError`, find and replace the Unicode character.

**Table data type error:**
ReportLab `Table` requires all cells to be strings or `Paragraph` objects. If you add a new table and see `TypeError`, ensure all float values are wrapped with `str(round(..., 4))`.

**PILOT RUN banner:**
Automatically added when `master_results.csv` contains ≤10 unique `doc_id` values. No manual configuration needed.

**Regenerating without re-running scoring:**
If you only need to update the PDF layout (not recompute scores), you can re-run `generate_pdf_report.py` directly — it reads from the existing CSVs and regenerates charts. Delete `.tmp/chart_*.png` first to force chart regeneration.
