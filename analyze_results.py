"""
analyze_results.py

Loads results/raw_results.json and produces:
  - Table 1: Hallucination Rate per Condition (CSV + print)
  - Table 2: AUROC per UE Method per Condition (CSV + print)
  - Table 3: Calibration per Condition (CSV + print)
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


def group_by_condition(results: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for r in results:
        grouped[r["condition"]].append(r)
    return dict(grouped)


def print_table(title: str, df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    print(df.to_string(index=False))


def main():
    if not os.path.exists(config.RAW_RESULTS_FILE):
        print(f"ERROR: {config.RAW_RESULTS_FILE} not found. Run run_experiment.py first.")
        sys.exit(1)

    print(f"Loading results from {config.RAW_RESULTS_FILE}...")
    results = load_results()
    print(f"Loaded {len(results)} result entries.")

    grouped = group_by_condition(results)
    conditions = [c for c in config.CONDITIONS if c in grouped]

    # Compute metrics per condition
    metrics_by_condition = {}
    for cond in conditions:
        metrics_by_condition[cond] = compute_all_metrics(grouped[cond])

    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    # ── Table 1: Hallucination Rate per Condition ──────────────────────
    rows1 = []
    for cond in conditions:
        m = metrics_by_condition[cond]
        rows1.append({
            "Condition": cond,
            "n": m["n"],
            "Exact Match (%)": round(m["exact_match_rate"] * 100, 1),
            "Hallucination Rate (%)": round(m["hallucination_rate"] * 100, 1),
            "Abstention Rate (%)": round(m["abstention_rate"] * 100, 1),
        })
    df1 = pd.DataFrame(rows1)
    print_table("Table 1: Hallucination Rate per Condition", df1)
    df1.to_csv(os.path.join(config.RESULTS_PATH, "table1_hallucination_rate.csv"), index=False)

    # ── Table 2: AUROC per UE Method per Condition ─────────────────────
    rows2 = []
    for cond in conditions:
        m = metrics_by_condition[cond]
        rows2.append({
            "Condition": cond,
            "AUROC Verbalized Conf.": m.get("auroc_verbalized_confidence"),
            "AUROC SelfCheckGPT": m.get("auroc_selfcheck"),
        })
    # Add Overall row
    all_metrics = compute_all_metrics(results)
    rows2.append({
        "Condition": "Overall",
        "AUROC Verbalized Conf.": all_metrics.get("auroc_verbalized_confidence"),
        "AUROC SelfCheckGPT": all_metrics.get("auroc_selfcheck"),
    })
    df2 = pd.DataFrame(rows2)
    print_table("Table 2: AUROC per UE Method per Condition", df2)
    df2.to_csv(os.path.join(config.RESULTS_PATH, "table2_auroc.csv"), index=False)

    # ── Table 3: Calibration per Condition ────────────────────────────
    rows3 = []
    for cond in conditions:
        m = metrics_by_condition[cond]
        rows3.append({
            "Condition": cond,
            "Mean Confidence (%)": m["mean_verbalized_confidence"],
            "Accuracy (%)": round((1 - m["hallucination_rate"]) * 100, 1),
            "ECE": m["ece"],
            "Overconfidence Gap": round(m["overconfidence_gap"], 4),
        })
    df3 = pd.DataFrame(rows3)
    print_table("Table 3: Calibration per Condition", df3)
    df3.to_csv(os.path.join(config.RESULTS_PATH, "table3_calibration.csv"), index=False)

    # ── Figures ────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig1_hallucination_rate(metrics_by_condition, config.RESULTS_PATH)
    fig2_auroc_comparison(metrics_by_condition, config.RESULTS_PATH)
    fig3_calibration(grouped, config.RESULTS_PATH)
    fig4_confidence_distribution(grouped, config.RESULTS_PATH)

    print(f"\nAll outputs saved to: {config.RESULTS_PATH}/")


if __name__ == "__main__":
    main()
