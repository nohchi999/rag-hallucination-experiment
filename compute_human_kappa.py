"""
compute_human_kappa.py  (Task F — judge validation; added 2026-06-30)

After the human fills validation_sheet.csv's `human_label` column, this computes
Cohen's kappa between the hand label and (a) the external judge label and
(b) the EM label — the measurable "the judge agrees with human judgement at
kappa = 0.8X" argument the briefing wants.

Join: each sheet row -> the matching judged record by (model, question_id,
prompt_type, condition), to read judge_verdict_ext and exact_match.

Verdict -> hallucination binary (same mapping as the pipeline):
    supported -> 0 ;  partially_supported / not_supported -> 1
EM -> hallucination binary: exact_match==1 -> 0 (not hallucinated) else 1.

Outputs: results/cross_model/analysis/human_kappa.{json,md}

Run AFTER labeling:  python compute_human_kappa.py
Does nothing destructive; refuses to run if human_label is empty.
"""

import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.metrics_v2 import cohens_kappa

SHEET = "results/cross_model/validation_sheet.csv"
JUDGED = {
    "claude-haiku-4.5": "results/raw_results_judged.json",
    "gpt-4.1-mini":     "results/cross_model/raw_results_openai_judged.json",
    "gemini-2.5-flash": "results/cross_model/raw_results_google_judged.json",
}
OUT_JSON = "results/cross_model/analysis/human_kappa.json"
OUT_MD = "results/cross_model/analysis/human_kappa.md"

VALID_LABELS = {"supported", "partially_supported", "not_supported"}


def verdict_to_hal(v):
    return 0 if v == "supported" else 1


def _load(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = list(csv.DictReader(io.open(SHEET, encoding="utf-8")))
    labeled = [r for r in rows if (r.get("human_label") or "").strip()]
    if not labeled:
        print(f"No human_label filled in {SHEET}. Fill it first (one of: {sorted(VALID_LABELS)}).")
        sys.exit(1)

    bad = [r["row_id"] for r in labeled if r["human_label"].strip() not in VALID_LABELS]
    if bad:
        print(f"Invalid human_label values in rows {bad}. Use only {sorted(VALID_LABELS)}.")
        sys.exit(1)

    # index judged records by (model, qid, pt, cond)
    index = {}
    for model, path in JUDGED.items():
        for r in _load(path):
            index[(model, r["question_id"], r["prompt_type"], r["condition"])] = r

    human_hal, judge_hal, em_hal = [], [], []
    human_verdict, judge_verdict = [], []
    missing = 0
    for r in labeled:
        key = (r["model"], int(r["question_id"]), r["prompt_type"], r["condition"])
        rec = index.get(key)
        if rec is None:
            missing += 1
            continue
        hv = r["human_label"].strip()
        jv = rec.get("judge_verdict_ext")
        human_verdict.append(hv)
        judge_verdict.append(jv)
        human_hal.append(verdict_to_hal(hv))
        judge_hal.append(verdict_to_hal(jv))
        em_hal.append(0 if rec.get("exact_match") == 1 else 1)

    n = len(human_hal)
    k_hj_bin = cohens_kappa(human_hal, judge_hal)        # human vs judge (hallucination binary)
    k_he_bin = cohens_kappa(human_hal, em_hal)           # human vs EM (hallucination binary)
    k_hj_3 = cohens_kappa(human_verdict, judge_verdict)  # human vs judge (3-level verdict)
    agree_hj = sum(1 for a, b in zip(human_hal, judge_hal) if a == b) / n
    agree_he = sum(1 for a, b in zip(human_hal, em_hal) if a == b) / n

    out = {
        "n_labeled": len(labeled), "n_matched": n, "n_missing_join": missing,
        "kappa_human_vs_judge_hallucination_binary": k_hj_bin,
        "kappa_human_vs_em_hallucination_binary": k_he_bin,
        "kappa_human_vs_judge_3level_verdict": k_hj_3,
        "agreement_human_vs_judge_pct": round(agree_hj * 100, 2),
        "agreement_human_vs_em_pct": round(agree_he * 100, 2),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Judge Validation — Cohen's kappa vs human labels (Task F)\n",
        f"- Labeled rows: **{len(labeled)}** (matched to judged data: {n}; unmatched: {missing})\n",
        "| Comparison | Cohen's kappa | Agreement |",
        "|---|---|---|",
        f"| Human vs external Judge (hallucination binary) | **{k_hj_bin}** | {out['agreement_human_vs_judge_pct']}% |",
        f"| Human vs EM (hallucination binary) | **{k_he_bin}** | {out['agreement_human_vs_em_pct']}% |",
        f"| Human vs external Judge (3-level verdict) | **{k_hj_3}** | — |",
        "\n*Judge = qwen3-235b-a22b-instruct-2507. supported->0, partially/not_supported->1; "
        "EM: exact_match==1 -> 0 else 1.*\n",
    ]
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"n_labeled={len(labeled)} matched={n} missing={missing}")
    print(f"kappa human-vs-judge (binary) = {k_hj_bin} | human-vs-EM (binary) = {k_he_bin}")
    print(f"kappa human-vs-judge (3-level) = {k_hj_3}")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
