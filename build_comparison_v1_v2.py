"""
build_comparison_v1_v2.py

Builds the v1-vs-v2 side-by-side comparison artifacts required by
BRIEFING_BUG_FIX.md §6 ("Summary of Changes", Old/New/Delta per metric).

Reads:
  results/table{1,2,3,4}_*.csv                   (v1 artifacts, untouched)
  results/reanalysis_v2/table*_v2.csv            (v2 artifacts from analyze_results_v2.py)
Writes:
  results/reanalysis_v2/comparison_v1_vs_v2.csv  (machine-readable)
  results/reanalysis_v2/comparison_v1_vs_v2.md   (human-readable)

Conventions:
  - delta = v2 - v1 (in the v1 unit; percentages stay percentages).
  - Cells where v1 has no directly-comparable column (e.g. squad_f1, ece_non_abstention)
    are marked as "new in v2".
"""

import csv
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(ROOT, "results")
V2_DIR = os.path.join(ROOT, "results", "reanalysis_v2")
OUT_CSV = os.path.join(V2_DIR, "comparison_v1_vs_v2.csv")
OUT_MD = os.path.join(V2_DIR, "comparison_v1_vs_v2.md")

PROMPT_TYPES = ["constrained", "unconstrained"]
CONDITIONS = ["full", "partial", "none"]


def _load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _to_float(x):
    if x is None or x == "" or x == "nan":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def _delta(v2, v1, nd=2):
    if v2 is None or v1 is None:
        return "—"
    d = v2 - v1
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{nd}f}"


def main():
    os.makedirs(V2_DIR, exist_ok=True)

    v1_t1 = {(r["Prompt"], r["Condition"]): r for r in _load_csv(os.path.join(V1_DIR, "table1_hallucination_rate.csv"))}
    v1_t2 = {r["Condition"]: r for r in _load_csv(os.path.join(V1_DIR, "table2_auroc.csv"))}
    v1_t3 = {(r["Prompt"], r["Condition"]): r for r in _load_csv(os.path.join(V1_DIR, "table3_calibration.csv"))}
    v1_t4 = {(r["Prompt"], r["Condition"]): r for r in _load_csv(os.path.join(V1_DIR, "table4_agreement.csv"))}

    v2_t1 = {(r["prompt_type"], r["condition"]): r for r in _load_csv(os.path.join(V2_DIR, "table1_hallucination_v2.csv"))}
    v2_t2 = {(r["prompt_type"], r["condition"]): r for r in _load_csv(os.path.join(V2_DIR, "table2_auroc_v2.csv"))}
    v2_t3 = {(r["prompt_type"], r["condition"]): r for r in _load_csv(os.path.join(V2_DIR, "table3_calibration_v2.csv"))}
    v2_t4 = {(r["prompt_type"], r["condition"]): r for r in _load_csv(os.path.join(V2_DIR, "table4_agreement_v2.csv"))}

    rows = []

    def add(metric, prompt_type, condition, v1_val, v2_val, nd=2, note=""):
        rows.append({
            "metric": metric,
            "prompt_type": prompt_type,
            "condition": condition,
            "v1": _fmt(v1_val, nd),
            "v2": _fmt(v2_val, nd),
            "delta": _delta(v2_val, v1_val, nd),
            "note": note,
        })

    # Table 1 metrics
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            r1 = v1_t1[(pt, cond)]
            r2 = v2_t1[(pt, cond)]
            add("EM rate (%)", pt, cond, _to_float(r1["EM (%)"]), _to_float(r2["em_rate_pct"]))
            add("Hallucination EM (%)", pt, cond, _to_float(r1["Halluc. EM (%)"]), _to_float(r2["hallucination_em_pct"]))
            add("Hallucination Judge (%)", pt, cond, _to_float(r1["Halluc. Judge (%)"]), _to_float(r2["hallucination_judge_pct"]))
            add("Abstention (%)", pt, cond, _to_float(r1["Abstention (%)"]), _to_float(r2["abstention_pct"]))
            add("Category sum (%)", pt, cond,
                _to_float(r1["EM (%)"]) + _to_float(r1["Halluc. EM (%)"]) + _to_float(r1["Abstention (%)"]),
                _to_float(r2["category_sum_pct"]),
                note="v1 >100% evidences Bug #4" if (_to_float(r1["EM (%)"]) + _to_float(r1["Halluc. EM (%)"]) + _to_float(r1["Abstention (%)"])) > 100.0 else "")
            add("SQuAD F1 (mean)", pt, cond, None, _to_float(r2["squad_f1_mean"]), nd=4, note="new in v2")

    # Table 2: v1 only has per-condition rows (marginalised across prompt_type) and an "Overall" row.
    # Compare the per-condition row against the mean of the two v2 cells for each condition.
    for cond in CONDITIONS:
        r1 = v1_t2[cond]
        def avg(pt_rows, key):
            vals = [_to_float(r[key]) for r in pt_rows if _to_float(r[key]) is not None]
            return sum(vals) / len(vals) if vals else None
        pt_rows = [v2_t2[(pt, cond)] for pt in PROMPT_TYPES]
        add("AUROC VC (EM)",        "avg_over_prompts", cond, _to_float(r1["AUROC VC (EM)"]),        avg(pt_rows, "auroc_vc_em"),        nd=4)
        add("AUROC VC (Judge)",     "avg_over_prompts", cond, _to_float(r1["AUROC VC (Judge)"]),     avg(pt_rows, "auroc_vc_judge"),     nd=4)
        add("AUROC SC (EM)",        "avg_over_prompts", cond, _to_float(r1["AUROC SelfCheck (EM)"]), avg(pt_rows, "auroc_sc_em"),        nd=4)
        add("AUROC SC (Judge)",     "avg_over_prompts", cond, _to_float(r1["AUROC SelfCheck (Judge)"]), avg(pt_rows, "auroc_sc_judge"),  nd=4)
        add("AUROC Judge (EM)",     "avg_over_prompts", cond, _to_float(r1["AUROC Judge (EM only)"]), avg(pt_rows, "auroc_judge_em"),    nd=4)

    # Per-cell AUROC new in v2 (v1 did not split by prompt_type)
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            r2 = v2_t2[(pt, cond)]
            add("AUROC VC (EM, per cell)", pt, cond, None, _to_float(r2["auroc_vc_em"]), nd=4, note="new in v2")
            add("AUROC SC (EM, per cell)", pt, cond, None, _to_float(r2["auroc_sc_em"]), nd=4, note="new in v2")
            add("AUROC VC (EM, non-abst only)", pt, cond, None, _to_float(r2["auroc_vc_em_non_abstention"]), nd=4, note="new in v2")
            add("AUROC SC (EM, non-abst only)", pt, cond, None, _to_float(r2["auroc_sc_em_non_abstention"]), nd=4, note="new in v2")

    # Table 3
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            r1 = v1_t3[(pt, cond)]
            r2 = v2_t3[(pt, cond)]
            add("Mean Confidence (%)", pt, cond, _to_float(r1["Mean Conf. (%)"]), _to_float(r2["mean_verbalized_confidence_parsed"]))
            add("Accuracy (%)", pt, cond, _to_float(r1["Accuracy (%)"]), _to_float(r2["em_accuracy_pct"]))
            add("ECE", pt, cond, _to_float(r1["ECE"]), _to_float(r2["ece_all"]), nd=4)
            add("ECE non-abstention", pt, cond, None, _to_float(r2["ece_non_abstention"]), nd=4, note="new in v2 (Bug #6)")
            add("Overconf. Gap", pt, cond, _to_float(r1["Overconf. Gap"]), _to_float(r2["overconfidence_gap"]), nd=4)
            add("Mean Conf non-abstention (%)", pt, cond, None, _to_float(r2["mean_verbalized_confidence_non_abstention"]), note="new in v2 (Bug #2)")

    # Table 4
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            r1 = v1_t4[(pt, cond)]
            r2 = v2_t4[(pt, cond)]
            add("Cohen's Kappa", pt, cond, _to_float(r1["Cohen's Kappa"]), _to_float(r2["cohens_kappa"]), nd=4)
            add("Agreement (%)", pt, cond, _to_float(r1["Agreement (%)"]), _to_float(r2["agreement_pct"]))
            add("n disagreements", pt, cond, _to_float(r1["n Disagreements"]), _to_float(r2["n_disagreements"]), nd=0)

    # CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "prompt_type", "condition", "v1", "v2", "delta", "note"])
        w.writeheader()
        w.writerows(rows)

    # Markdown: one section per table, with the per-cell rows
    md = []
    md.append("# v1 vs v2 Comparison — Headline Metrics\n")
    md.append(
        "This document pairs every headline number from the original analysis pipeline "
        "(`results/table{1,2,3,4}_*.csv`) against the reanalysis pipeline "
        "(`results/reanalysis_v2/table*_v2.csv`). `delta = v2 − v1`. "
        "Rows marked _new in v2_ have no v1 counterpart and are reported for completeness.\n"
    )
    md.append("Note: v1 uses n=200 per cell; v2 uses n=198 after filtering the two degenerate "
              "SQuAD items (Q61 gt='H', Q74 question='k').\n")

    current_metric = None
    for r in rows:
        if r["metric"] != current_metric:
            md.append(f"\n## {r['metric']}\n")
            md.append("| prompt_type | condition | v1 | v2 | Δ | note |")
            md.append("|---|---|---|---|---|---|")
            current_metric = r["metric"]
        md.append(f"| {r['prompt_type']} | {r['condition']} | {r['v1']} | {r['v2']} | {r['delta']} | {r['note']} |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"  rows: {len(rows)}")


if __name__ == "__main__":
    main()
