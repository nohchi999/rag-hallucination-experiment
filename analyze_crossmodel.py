"""
analyze_crossmodel.py  (Tasks F + G, label-independent parts; added 2026-06-30)

Runs the VALIDATED v2 pipeline (analyze_results_v2.enrich + compute_cells +
write_table1..4) unchanged for EACH generator model, using the SAME external
judge for all three. The external judge labels (judge_verdict_ext /
is_hallucinated_judge_ext) are remapped onto the canonical judge fields so the
existing pipeline consumes them with zero logic changes — keeping the
anti-circularity rule (judge-UE AUROC vs EM only) intact.

This covers everything in Task F EXCEPT the human-vs-judge / human-vs-EM kappa,
which needs the hand-labeled validation_sheet.csv (see compute_human_kappa.py).

Outputs:
  results/cross_model/analysis/<model>/table1..4.csv
  results/cross_model/analysis/cross_model_summary.{csv,md}   (Task G synthesis)

Run:  python analyze_crossmodel.py
"""

import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_results_v2 import (
    enrich, compute_cells, write_table1, write_table2, write_table3, write_table4,
)

MODELS = {
    "claude-haiku-4.5": "results/raw_results_judged.json",
    "gpt-4.1-mini":     "results/cross_model/raw_results_openai_judged.json",
    "gemini-2.5-flash": "results/cross_model/raw_results_google_judged.json",
}
OUT_ROOT = "results/cross_model/analysis"
PROMPT_TYPES = ["constrained", "unconstrained"]
CONDITIONS = ["full", "partial", "none"]


def _load(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remap_external_judge(results):
    """Point the canonical judge fields at the external judge so the validated
    pipeline scores every model with the SAME judge."""
    n = 0
    for r in results:
        if r.get("is_hallucinated_judge_ext") is not None:
            r["is_hallucinated_judge"] = r["is_hallucinated_judge_ext"]
            r["judge_verdict"] = r.get("judge_verdict_ext")
            n += 1
    return n


def analyze_model(name, path):
    results = _load(path)
    remapped = remap_external_judge(results)
    enriched, _diag = enrich(results)
    cells = compute_cells(enriched, filter_invalid=True)
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    write_table1(cells, os.path.join(out_dir, "table1_hallucination.csv"))
    write_table2(cells, os.path.join(out_dir, "table2_auroc.csv"))
    write_table3(cells, os.path.join(out_dir, "table3_calibration.csv"))
    write_table4(cells, os.path.join(out_dir, "table4_agreement.csv"))
    return cells, remapped


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    per_model = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: missing {path}")
            continue
        cells, remapped = analyze_model(name, path)
        per_model[name] = cells
        print(f"[{name}] analyzed ({remapped} ext-judge labels). tables -> {OUT_ROOT}/{name}/")

    # --- Task G synthesis: F1 (prompt mitigation under partial evidence, EM) ---
    # F1 = constrained/partial vs unconstrained/partial EM hallucination rate.
    rows = []
    for name, cells in per_model.items():
        c_part = cells.get(("constrained", "partial"), {})
        u_part = cells.get(("unconstrained", "partial"), {})
        c_hal = c_part.get("hallucination_rate_em")
        u_hal = u_part.get("hallucination_rate_em")
        if c_hal is None or u_hal is None:
            status = "n/a"
        else:
            # replicated if constrained meaningfully lower than unconstrained (same direction as Haiku)
            drop = u_hal - c_hal
            if drop > 0.05:
                status = "replicated"
            elif drop > 0.0:
                status = "partial"
            else:
                status = "divergent"
        rows.append({
            "model": name,
            "uncon_partial_hal_em_pct": None if u_hal is None else round(u_hal * 100, 2),
            "con_partial_hal_em_pct": None if c_hal is None else round(c_hal * 100, 2),
            "F1_prompt_mitigation": status,
        })

    csv_path = os.path.join(OUT_ROOT, "cross_model_summary.csv")
    with io.open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "uncon_partial_hal_em_pct",
                                          "con_partial_hal_em_pct", "F1_prompt_mitigation"])
        w.writeheader(); w.writerows(rows)

    md = ["# Cross-Model Synthesis (Task G) — label-independent\n",
          "## F1 — Prompt mitigation under partial evidence (EM-based hallucination)\n",
          "| Model | unconstrained/partial | constrained/partial | F1 status |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['uncon_partial_hal_em_pct']}% | "
                  f"{r['con_partial_hal_em_pct']}% | **{r['F1_prompt_mitigation']}** |")
    md.append("\n*EM-based; the validated pipeline (disjoint classify_response) and the SAME "
              "external judge (qwen3-235b-a22b-instruct-2507) were used for all models. "
              "AUROC/ECE/judge-rate tables per model in ./<model>/. Human-vs-judge kappa pending labels.*\n")
    with io.open(os.path.join(OUT_ROOT, "cross_model_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n=== F1 cross-model (EM hallucination under partial) ===")
    for r in rows:
        print(f"  {r['model']:18s} uncon/partial={r['uncon_partial_hal_em_pct']}%  "
              f"con/partial={r['con_partial_hal_em_pct']}%  -> {r['F1_prompt_mitigation']}")
    print(f"\nWrote {OUT_ROOT}/cross_model_summary.(csv|md)")


if __name__ == "__main__":
    main()
