"""
analyze_results_v2.py

Reanalysis pipeline. Reads results/raw_results.json, applies the v2 fixes
(re-parse confidence, extended abstention detection, disjoint classification,
SQuAD-F1, degenerate-datapoint filter, ECE variants), and writes the full set
of v2 artifacts into results/reanalysis_v2/.

The original analyze_results.py is kept unchanged for auditability.
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.metrics_v2 import (
    EXTENDED_ABSTENTION_MARKERS,
    classify_response,
    compute_all_metrics_v2,
    detect_abstention_v2,
    is_valid_datapoint,
    reparse_confidence,
    squad_f1,
)


INPUT_PATH = "results/raw_results.json"
OUTPUT_DIR = "results/reanalysis_v2"
CLEANED_PATH = os.path.join(OUTPUT_DIR, "raw_results_cleaned.json")
BUG_REPORT_PATH = os.path.join(OUTPUT_DIR, "bug_impact_report.md")

PROMPT_TYPES = ["constrained", "unconstrained"]
CONDITIONS = ["full", "partial", "none"]


def enrich(results):
    """Add v2 fields to every result in-place and return (enriched, diagnostics)."""
    diagnostics = {
        "n_total": len(results),
        "n_conf_reparse_changed": 0,
        "n_conf_reparse_none": 0,
        "n_conf_v1_eq_50_v2_parsed": 0,
        "n_conf_v1_eq_50_v2_none": 0,
        "n_abstention_v1_true_v2_true": 0,
        "n_abstention_v1_true_v2_false": 0,
        "n_abstention_v1_false_v2_true": 0,
        "n_abstention_v1_false_v2_false": 0,
        "n_reclassified_hall_to_abs_by_prompt": Counter(),
        "n_invalid_datapoints": 0,
        "invalid_datapoints": [],
        "category_counts": Counter(),
    }

    for r in results:
        raw_text = (r.get("full_api_response") or {}).get("raw_text", "")
        conf_v2 = reparse_confidence(raw_text)
        conf_v1 = r.get("verbalized_confidence")

        r["verbalized_confidence_v2"] = conf_v2
        if conf_v2 is None:
            diagnostics["n_conf_reparse_none"] += 1
            if conf_v1 == 50:
                diagnostics["n_conf_v1_eq_50_v2_none"] += 1
        else:
            if conf_v1 != conf_v2:
                diagnostics["n_conf_reparse_changed"] += 1
            if conf_v1 == 50 and conf_v2 != 50:
                diagnostics["n_conf_v1_eq_50_v2_parsed"] += 1

        is_abs_v2 = detect_abstention_v2(r.get("answer", ""))
        r["is_abstention_v2"] = is_abs_v2
        is_abs_v1 = bool(r.get("is_abstention", False))
        key = (
            "n_abstention_v1_" + ("true" if is_abs_v1 else "false")
            + "_v2_" + ("true" if is_abs_v2 else "false")
        )
        diagnostics[key] += 1
        if is_abs_v2 and not is_abs_v1:
            diagnostics["n_reclassified_hall_to_abs_by_prompt"][r.get("prompt_type", "unknown")] += 1

        category = classify_response(r)
        r["response_category_v2"] = category
        r["is_hallucinated_em_v2"] = 1 if category == "hallucinated" else 0
        diagnostics["category_counts"][(r.get("prompt_type"), r.get("condition"), category)] += 1

        r["squad_f1_v2"] = squad_f1(r.get("answer", ""), r.get("ground_truth", ""))

        valid, reason = is_valid_datapoint(r)
        r["is_valid_v2"] = valid
        if not valid:
            diagnostics["n_invalid_datapoints"] += 1
            diagnostics["invalid_datapoints"].append({
                "question_id": r.get("question_id"),
                "prompt_type": r.get("prompt_type"),
                "condition": r.get("condition"),
                "reason": reason,
            })

    return results, diagnostics


def cell_key(r):
    return (r.get("prompt_type"), r.get("condition"))


def compute_cells(results, filter_invalid=True):
    """Group by (prompt_type, condition) and compute v2 metrics per cell."""
    grouped = defaultdict(list)
    for r in results:
        if filter_invalid and not r.get("is_valid_v2", True):
            continue
        grouped[cell_key(r)].append(r)
    out = {}
    for key, group in grouped.items():
        out[key] = compute_all_metrics_v2(group)
    return out


def write_table1(cells, path):
    """Table 1: Hallucination & abstention rates per cell. Sum must be 100%."""
    cols = [
        "prompt_type", "condition", "n",
        "em_rate_pct", "hallucination_em_pct", "abstention_pct",
        "category_sum_pct",
        "hallucination_judge_pct",
        "hal_em_ci_lower_pct", "hal_em_ci_upper_pct",
        "hal_judge_ci_lower_pct", "hal_judge_ci_upper_pct",
        "squad_f1_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pt in PROMPT_TYPES:
            for cond in CONDITIONS:
                m = cells.get((pt, cond), {})
                if not m:
                    continue
                em_pct = m["em_rate"] * 100
                hal_pct = m["hallucination_rate_em"] * 100
                abs_pct = m["abstention_rate"] * 100
                cat_sum = em_pct + hal_pct + abs_pct
                w.writerow([
                    pt, cond, m["n"],
                    round(em_pct, 2), round(hal_pct, 2), round(abs_pct, 2),
                    round(cat_sum, 2),
                    round(m["hallucination_rate_judge"] * 100, 2),
                    round(m["hallucination_rate_em_ci_lower"] * 100, 2),
                    round(m["hallucination_rate_em_ci_upper"] * 100, 2),
                    round(m["hallucination_rate_judge_ci_lower"] * 100, 2),
                    round(m["hallucination_rate_judge_ci_upper"] * 100, 2),
                    m.get("squad_f1_mean"),
                ])


def write_table2(cells, path):
    """Table 2: AUROC per UE method per cell, EM label primary + Judge label + non-abstention secondary."""
    cols = [
        "prompt_type", "condition",
        "auroc_vc_em", "auroc_vc_em_ci_lower", "auroc_vc_em_ci_upper", "auroc_vc_em_n",
        "auroc_sc_em", "auroc_sc_em_ci_lower", "auroc_sc_em_ci_upper", "auroc_sc_em_n",
        "auroc_judge_em", "auroc_judge_em_ci_lower", "auroc_judge_em_ci_upper", "auroc_judge_em_n",
        "auroc_vc_judge", "auroc_vc_judge_ci_lower", "auroc_vc_judge_ci_upper", "auroc_vc_judge_n",
        "auroc_sc_judge", "auroc_sc_judge_ci_lower", "auroc_sc_judge_ci_upper", "auroc_sc_judge_n",
        "auroc_vc_em_non_abstention", "auroc_vc_em_non_abstention_ci_lower",
        "auroc_vc_em_non_abstention_ci_upper", "auroc_vc_em_non_abstention_n",
        "auroc_sc_em_non_abstention", "auroc_sc_em_non_abstention_ci_lower",
        "auroc_sc_em_non_abstention_ci_upper", "auroc_sc_em_non_abstention_n",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pt in PROMPT_TYPES:
            for cond in CONDITIONS:
                m = cells.get((pt, cond), {})
                if not m:
                    continue
                w.writerow([pt, cond] + [m.get(c) for c in cols[2:]])


def write_table3(cells, path):
    """Table 3: Calibration with three ECE variants per cell."""
    cols = [
        "prompt_type", "condition",
        "mean_verbalized_confidence_parsed",
        "mean_verbalized_confidence_non_abstention",
        "em_accuracy_pct",
        "ece_all", "ece_non_abstention", "ece_parsed_only",
        "ece_n_all", "ece_n_non_abstention", "ece_n_unparsed",
        "overconfidence_gap",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pt in PROMPT_TYPES:
            for cond in CONDITIONS:
                m = cells.get((pt, cond), {})
                if not m:
                    continue
                em_acc_pct = round((1 - m["hallucination_rate_em"]) * 100, 2)
                w.writerow([
                    pt, cond,
                    m.get("mean_verbalized_confidence_parsed"),
                    m.get("mean_verbalized_confidence_non_abstention"),
                    em_acc_pct,
                    m.get("ece_all"), m.get("ece_non_abstention"), m.get("ece_parsed_only"),
                    m.get("ece_n_all"), m.get("ece_n_non_abstention"), m.get("ece_n_unparsed"),
                    m.get("overconfidence_gap"),
                ])


def write_table4(cells, path):
    """Table 4: EM v2 vs Judge label agreement per cell."""
    cols = [
        "prompt_type", "condition", "n",
        "cohens_kappa", "n_disagreements", "agreement_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pt in PROMPT_TYPES:
            for cond in CONDITIONS:
                m = cells.get((pt, cond), {})
                if not m:
                    continue
                agreement_pct = round((1 - m["label_disagreements"] / m["n"]) * 100, 2)
                w.writerow([
                    pt, cond, m["n"],
                    m.get("cohens_kappa"),
                    m.get("label_disagreements"),
                    agreement_pct,
                ])


def _abstention_confidence_bins(results):
    """For Bug #2 reporting: confidence distribution among abstentions."""
    bins_by_cell = {}
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            subset = [r for r in results
                      if r.get("prompt_type") == pt and r.get("condition") == cond
                      and r.get("is_abstention_v2")
                      and r.get("is_valid_v2", True)
                      and r.get("verbalized_confidence_v2") is not None]
            n = len(subset)
            if n == 0:
                bins_by_cell[(pt, cond)] = {"n": 0}
                continue
            high = sum(1 for r in subset if r["verbalized_confidence_v2"] >= 90)
            low = sum(1 for r in subset if r["verbalized_confidence_v2"] <= 10)
            middle = sum(1 for r in subset if 40 < r["verbalized_confidence_v2"] < 60)
            bins_by_cell[(pt, cond)] = {
                "n": n,
                "pct_high": round(high / n * 100, 1),
                "pct_low": round(low / n * 100, 1),
                "pct_middle": round(middle / n * 100, 1),
            }
    return bins_by_cell


def write_bug_report(results, diagnostics, cells, path):
    """Render results/reanalysis_v2/bug_impact_report.md."""
    abs_bins = _abstention_confidence_bins(results)
    reclassified_samples = [
        r for r in results
        if r.get("is_abstention_v2") and not r.get("is_abstention", False)
        and r.get("is_valid_v2", True)
    ][:10]

    lines = []
    lines.append("# Bug Impact Report — v2 Reanalysis\n")
    lines.append("This report quantifies the effect of the six bug fixes defined in "
                 "`BRIEFING_BUG_FIX.md`. The v1 pipeline (`src/metrics.py`, "
                 "`analyze_results.py`) is kept unchanged for auditability.\n")

    lines.append("\n## Summary\n")
    lines.append(f"- Total datapoints: {diagnostics['n_total']}")
    lines.append(f"- Invalid datapoints filtered (Bug #5): {diagnostics['n_invalid_datapoints']}")
    lines.append(f"- Confidence values reparsed differently (Bug #1): "
                 f"{diagnostics['n_conf_reparse_changed']} changed, "
                 f"{diagnostics['n_conf_reparse_none']} now None (vs v1 default 50)")
    lines.append(f"- Abstentions reclassified from v1-false to v2-true (Bug #3): "
                 f"{diagnostics['n_abstention_v1_false_v2_true']}")
    lines.append(f"- Abstentions reclassified from v1-true to v2-false (Bug #3/4): "
                 f"{diagnostics['n_abstention_v1_true_v2_false']}\n")

    lines.append("\n## Bug #1 — Parser Default Contamination\n")
    lines.append(f"- v1 had `verbalized_confidence = 50` on {sum(1 for r in results if r.get('verbalized_confidence') == 50)} rows")
    lines.append(f"- Of those, v2 recovered a real Confidence value on "
                 f"{diagnostics['n_conf_v1_eq_50_v2_parsed']} rows")
    lines.append(f"- Of those, v2 determined no Confidence was parseable on "
                 f"{diagnostics['n_conf_v1_eq_50_v2_none']} rows (now stored as None, not 50)")

    lines.append("\n## Bug #2 — Abstention Confidence Semantics\n")
    lines.append("Confidence distribution among v2 abstentions (where confidence is parseable):\n")
    lines.append("| prompt_type | condition | n | pct_high (≥90) | pct_low (≤10) | pct_middle (40<c<60) |")
    lines.append("|---|---|---|---|---|---|")
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            b = abs_bins[(pt, cond)]
            if b["n"] == 0:
                lines.append(f"| {pt} | {cond} | 0 | — | — | — |")
            else:
                lines.append(f"| {pt} | {cond} | {b['n']} | {b['pct_high']} | {b['pct_low']} | {b['pct_middle']} |")

    lines.append("\n## Bug #3 — Abstention Detector Gaps\n")
    reclass = diagnostics["n_reclassified_hall_to_abs_by_prompt"]
    lines.append(f"- Reclassified v1-hallucination → v2-abstention: "
                 f"constrained={reclass.get('constrained', 0)}, "
                 f"unconstrained={reclass.get('unconstrained', 0)}")
    lines.append("\nSample of up to 10 reclassifications:\n")
    lines.append("| question_id | prompt_type | condition | answer excerpt |")
    lines.append("|---|---|---|---|")
    for r in reclassified_samples:
        excerpt = (r.get("answer") or "").replace("\n", " ").replace("|", "/")[:140]
        lines.append(f"| {r.get('question_id')} | {r.get('prompt_type')} | {r.get('condition')} | {excerpt} |")

    lines.append("\n## Bug #4 — EM / Abstention Disjoint Check\n")
    lines.append("Every cell's three category percentages must sum to exactly 100.00%.\n")
    lines.append("| prompt_type | condition | em_pct | hall_em_pct | abst_pct | sum |")
    lines.append("|---|---|---|---|---|---|")
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            m = cells.get((pt, cond))
            if not m:
                continue
            em = m["em_rate"] * 100
            hal = m["hallucination_rate_em"] * 100
            abst = m["abstention_rate"] * 100
            lines.append(f"| {pt} | {cond} | {round(em,2)} | {round(hal,2)} | "
                         f"{round(abst,2)} | {round(em+hal+abst,2)} |")

    lines.append("\n## Bug #5 — Degenerate Datapoints\n")
    lines.append(f"Filtered out {diagnostics['n_invalid_datapoints']} datapoints:\n")
    if diagnostics["invalid_datapoints"]:
        lines.append("| question_id | prompt_type | condition | reason |")
        lines.append("|---|---|---|---|")
        for d in diagnostics["invalid_datapoints"]:
            lines.append(f"| {d['question_id']} | {d['prompt_type']} | {d['condition']} | {d['reason']} |")
    else:
        lines.append("_None._")

    lines.append("\n## Bug #6 — ECE Variants\n")
    lines.append("| prompt_type | condition | ECE_all | ECE_non_abstention | ECE_parsed_only | N_all | N_non_abs | N_unparsed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for pt in PROMPT_TYPES:
        for cond in CONDITIONS:
            m = cells.get((pt, cond))
            if not m:
                continue
            lines.append(f"| {pt} | {cond} | {m.get('ece_all')} | {m.get('ece_non_abstention')} | "
                         f"{m.get('ece_parsed_only')} | {m.get('ece_n_all')} | "
                         f"{m.get('ece_n_non_abstention')} | {m.get('ece_n_unparsed')} |")

    lines.append("\n## Impact on Thesis Findings\n")
    lines.append("| Finding | v1 claim | v2 status (to be filled after review) |")
    lines.append("|---|---|---|")
    lines.append("| Prompt reduces hallucination (Constr vs Unconstr / Partial) | Δ = 24.5% → 9% | _see table 1_ |")
    lines.append("| SelfCheckGPT outperforms VC | AUROC_SC > AUROC_VC | _see table 2_ |")
    lines.append("| AUROC VC < 0.5 at Partial | confidence inversion | _see table 2_ |")
    lines.append("| Under-confidence contradicts Xiong | mean_conf < accuracy | _see table 3_ |")
    lines.append("| Partial Evidence is most dangerous | highest hall_rate under Partial | _see table 1_ |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    results, diagnostics = enrich(results)

    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    cells = compute_cells(results, filter_invalid=True)

    write_table1(cells, os.path.join(OUTPUT_DIR, "table1_hallucination_v2.csv"))
    write_table2(cells, os.path.join(OUTPUT_DIR, "table2_auroc_v2.csv"))
    write_table3(cells, os.path.join(OUTPUT_DIR, "table3_calibration_v2.csv"))
    write_table4(cells, os.path.join(OUTPUT_DIR, "table4_agreement_v2.csv"))
    write_bug_report(results, diagnostics, cells, BUG_REPORT_PATH)

    # Print a compact summary to stdout
    print(f"Wrote {CLEANED_PATH}")
    for tbl in ["table1_hallucination_v2.csv", "table2_auroc_v2.csv",
                "table3_calibration_v2.csv", "table4_agreement_v2.csv",
                "bug_impact_report.md"]:
        print(f"Wrote {os.path.join(OUTPUT_DIR, tbl)}")

    print()
    print("Diagnostics:")
    print(f"  total: {diagnostics['n_total']}")
    print(f"  invalid: {diagnostics['n_invalid_datapoints']}")
    print(f"  conf_reparse_changed: {diagnostics['n_conf_reparse_changed']}")
    print(f"  conf_reparse_none: {diagnostics['n_conf_reparse_none']}")
    print(f"  conf_v1=50 -> v2 parsed: {diagnostics['n_conf_v1_eq_50_v2_parsed']}")
    print(f"  conf_v1=50 -> v2 None: {diagnostics['n_conf_v1_eq_50_v2_none']}")
    print(f"  abstention v1=F v2=T: {diagnostics['n_abstention_v1_false_v2_true']}")
    print(f"  abstention v1=T v2=F: {diagnostics['n_abstention_v1_true_v2_false']}")


if __name__ == "__main__":
    main()
