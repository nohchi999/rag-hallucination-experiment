"""
analyze_results.py

Loads results/raw_results.json and produces:
  - Table 1: Hallucination Rate per Prompt Type × Condition (CSV + print)
  - Table 2: AUROC per UE Method (Unconstrained, against EM label) (CSV + print)
  - Table 3: Calibration per Prompt Type × Condition (CSV + print)
  - Table 4: Inter-Label Agreement (Cohen's Kappa) (CSV + print)
  - fig1_hallucination_rate.png
  - fig2_auroc_comparison.png
  - fig3_calibration.png
  - fig4_confidence_distribution.png

Usage:
    python analyze_results.py
"""

import json
import os
import sys
from collections import defaultdict

import pandas as pd

import config
from src.metrics import compute_all_metrics
from src.visualize import (
    fig1_hallucination_rate,
    fig2_auroc_comparison,
    fig3_calibration,
    fig4_confidence_distribution,
)


def load_results(path: str = config.RAW_RESULTS_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_cell(results: list[dict]) -> dict[tuple, list[dict]]:
    """Group results by (prompt_type, condition) cell."""
    grouped = defaultdict(list)
    for r in results:
        key = (r.get("prompt_type", "constrained"), r["condition"])
        grouped[key].append(r)
    return dict(grouped)


def print_table(title: str, df: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)
    print(df.to_string(index=False))


def main():
    if not os.path.exists(config.RAW_RESULTS_FILE):
        print(f"ERROR: {config.RAW_RESULTS_FILE} not found. Run run_experiment.py first.")
        sys.exit(1)

    print(f"Loading results from {config.RAW_RESULTS_FILE}...")
    results = load_results()
    print(f"Loaded {len(results)} result entries.")

    grouped = group_by_cell(results)

    # Compute metrics per cell
    metrics_by_cell = {}
    for key in grouped:
        metrics_by_cell[key] = compute_all_metrics(grouped[key])

    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    cells = [(pt, c) for pt in config.PROMPT_TYPES for c in config.CONDITIONS]
    cells = [c for c in cells if c in grouped]

    # ── Table 1: Hallucination Rate per Cell ──────────────────────────────
    rows1 = []
    for (pt, cond) in cells:
        m = metrics_by_cell[(pt, cond)]
        hal_em = m["hallucination_rate_em"]
        hal_judge = m["hallucination_rate_judge"]
        ci_lo = m["hallucination_rate_em_ci_lower"]
        ci_hi = m["hallucination_rate_em_ci_upper"]
        rows1.append({
            "Prompt": pt,
            "Condition": cond,
            "n": m["n"],
            "EM (%)": round(m["exact_match_rate"] * 100, 1),
            "Halluc. EM (%)": round(hal_em * 100, 1),
            "Halluc. Judge (%)": round(hal_judge * 100, 1),
            "Abstention (%)": round(m["abstention_rate"] * 100, 1),
            "EM 95% CI": f"[{round(ci_lo*100,1)}, {round(ci_hi*100,1)}]",
        })
    df1 = pd.DataFrame(rows1)
    print_table("Table 1: Hallucination Rate and Abstention (2×3 Design)", df1)
    df1.to_csv(os.path.join(config.RESULTS_PATH, "table1_hallucination_rate.csv"), index=False)

    # ── Table 2: AUROC per UE Method (primary: Unconstrained, EM label) ──
    rows2 = []
    for cond in config.CONDITIONS:
        key = ("unconstrained", cond)
        if key not in metrics_by_cell:
            continue
        m = metrics_by_cell[key]
        rows2.append({
            "Condition": cond,
            "AUROC VC (EM)": m.get("auroc_vc_em"),
            "AUROC VC (Judge)": m.get("auroc_vc_judge"),
            "AUROC SelfCheck (EM)": m.get("auroc_sc_em"),
            "AUROC SelfCheck (Judge)": m.get("auroc_sc_judge"),
            "AUROC Judge (EM only)": m.get("auroc_judge_em"),
        })
    # Also compute overall for unconstrained
    unconstrained_results = [r for r in results if r.get("prompt_type") == "unconstrained"]
    if unconstrained_results:
        m_all = compute_all_metrics(unconstrained_results)
        rows2.append({
            "Condition": "Overall",
            "AUROC VC (EM)": m_all.get("auroc_vc_em"),
            "AUROC VC (Judge)": m_all.get("auroc_vc_judge"),
            "AUROC SelfCheck (EM)": m_all.get("auroc_sc_em"),
            "AUROC SelfCheck (Judge)": m_all.get("auroc_sc_judge"),
            "AUROC Judge (EM only)": m_all.get("auroc_judge_em"),
        })
    df2 = pd.DataFrame(rows2)
    print_table("Table 2: AUROC per UE Method — Unconstrained Condition", df2)
    df2.to_csv(os.path.join(config.RESULTS_PATH, "table2_auroc.csv"), index=False)

    # ── Table 3: Calibration (2×3) ────────────────────────────────────────
    rows3 = []
    for (pt, cond) in cells:
        m = metrics_by_cell[(pt, cond)]
        rows3.append({
            "Prompt": pt,
            "Condition": cond,
            "Mean Conf. (%)": m["mean_verbalized_confidence"],
            "Accuracy (%)": round((1 - m["hallucination_rate_em"]) * 100, 1),
            "ECE": m["ece"],
            "Overconf. Gap": round(m["overconfidence_gap"], 4),
        })
    df3 = pd.DataFrame(rows3)
    print_table("Table 3: Calibration (2×3 Design)", df3)
    df3.to_csv(os.path.join(config.RESULTS_PATH, "table3_calibration.csv"), index=False)

    # ── Table 4: Inter-Label Agreement (Cohen's Kappa) ────────────────────
    rows4 = []
    for (pt, cond) in cells:
        m = metrics_by_cell[(pt, cond)]
        rows4.append({
            "Prompt": pt,
            "Condition": cond,
            "Cohen's Kappa": m.get("cohens_kappa"),
            "Agreement (%)": round((1 - m["label_disagreements"] / m["n"]) * 100, 1) if m["n"] > 0 else None,
            "n Disagreements": m["label_disagreements"],
        })
    df4 = pd.DataFrame(rows4)
    print_table("Table 4: Inter-Label Agreement (EM vs. Judge)", df4)
    df4.to_csv(os.path.join(config.RESULTS_PATH, "table4_agreement.csv"), index=False)

    # ── Figures ──────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig1_hallucination_rate(metrics_by_cell, config.RESULTS_PATH)
    fig2_auroc_comparison(metrics_by_cell, config.RESULTS_PATH)
    fig3_calibration(grouped, config.RESULTS_PATH)
    fig4_confidence_distribution(grouped, config.RESULTS_PATH)

    print(f"\nAll outputs saved to: {config.RESULTS_PATH}/")


if __name__ == "__main__":
    main()
