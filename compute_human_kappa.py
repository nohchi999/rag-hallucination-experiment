"""
compute_human_kappa.py  (Task F — judge validation; added 2026-06-30)

Computes Cohen's kappa between the human gold labels and the pipeline / judge / EM
labels, to validate the automated classification and the external judge.

Reads the hand-labeled sheet (validation_sheet_labeled.xlsx preferred, else the
csv). Column names and the label scheme are auto-detected:
  * correctness scheme:  abstention / correct / incorrect      (what the human used)
  * groundedness scheme: supported / partially_supported / not_supported

Pipeline labels use the EXTENDED abstention detector (src.abstention_ext), the same
instrument used in analyze_crossmodel.py and applied uniformly to all models
(Haiku-safe). Joins each labeled row to its judged record by
(model, question_id, prompt_type, condition).

Outputs: results/cross_model/analysis/human_kappa.{json,md}

Run:  python compute_human_kappa.py
Refuses to run if no labels are present (never fabricates labels).
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import cohen_kappa_score

from src.abstention_ext import classify_response_ext
from src.judge import verdict_to_hallucinated

SHEET_XLSX = "results/cross_model/validation_sheet_labeled.xlsx"
SHEET_CSV = "results/cross_model/validation_sheet.csv"
JUDGED = {
    "claude-haiku-4.5": "results/raw_results_judged.json",
    "gpt-4.1-mini":     "results/cross_model/raw_results_openai_judged.json",
    "gemini-2.5-flash": "results/cross_model/raw_results_google_judged.json",
}
OUT_JSON = "results/cross_model/analysis/human_kappa.json"
OUT_MD = "results/cross_model/analysis/human_kappa.md"

# human label -> (3-way category, hallucination binary)
LABEL_MAP = {
    # correctness scheme (what the human used)
    "abstention": ("abstention", 0),
    "correct": ("em_correct", 0),
    "incorrect": ("hallucinated", 1),
    # groundedness scheme (sheet's original request)
    "supported": ("abstention_or_supported", 0),
    "partially_supported": ("hallucinated", 1),
    "not_supported": ("hallucinated", 1),
}


def _read_rows():
    import csv
    if os.path.exists(SHEET_XLSX):
        import pandas as pd
        df = pd.read_excel(SHEET_XLSX)
        return [dict(r) for _, r in df.iterrows()]
    with io.open(SHEET_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get(row, *names):
    for n in names:
        if n in row and str(row[n]).strip() not in ("", "nan", "None"):
            return row[n]
    return None


def _load(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = _read_rows()
    # build judged index
    idx = {}
    for model, path in JUDGED.items():
        for r in _load(path):
            idx[(model, r["question_id"], r["prompt_type"], r["condition"])] = r

    human_cat, pipe_cat, human_hal, judge_hal, em_hal = [], [], [], [], []
    n_labeled = n_match = n_bad = 0
    for row in rows:
        lab = _get(row, "★ HUMAN LABEL", "human_label", "HUMAN LABEL")
        if lab is None:
            continue
        n_labeled += 1
        lab = str(lab).strip().lower()
        if lab not in LABEL_MAP:
            n_bad += 1
            continue
        model = _get(row, "Model", "model")
        qid = _get(row, "Q-ID", "question_id")
        pt = _get(row, "Prompt type", "prompt_type")
        cond = _get(row, "Context cond.", "condition")
        rec = idx.get((model, int(qid), pt, cond))
        if rec is None:
            continue
        n_match += 1
        hcat, hbin = LABEL_MAP[lab]
        cat = classify_response_ext(rec)
        human_cat.append(hcat)
        pipe_cat.append(cat)
        human_hal.append(hbin)
        em_hal.append(1 if cat == "hallucinated" else 0)
        judge_hal.append(verdict_to_hallucinated(rec.get("judge_verdict_ext")))

    if n_labeled == 0:
        print(f"No labels found in the sheet. Fill the human-label column first.")
        sys.exit(1)

    def k(a, b):
        return round(float(cohen_kappa_score(a, b)), 3)

    def agree(a, b):
        return round(sum(x == y for x, y in zip(a, b)) / len(a) * 100, 1)

    out = {
        "n_labeled": n_labeled, "n_matched": n_match, "n_unmapped_label": n_bad,
        "kappa_human_vs_pipeline_category_3way": k(human_cat, pipe_cat),
        "agreement_human_vs_pipeline_category_pct": agree(human_cat, pipe_cat),
        "kappa_human_vs_external_judge_hal": k(human_hal, judge_hal),
        "agreement_human_vs_judge_pct": agree(human_hal, judge_hal),
        "kappa_human_vs_em_hal": k(human_hal, em_hal),
        "agreement_human_vs_em_pct": agree(human_hal, em_hal),
        "note": ("Pipeline labels use the extended abstention detector (Haiku-safe). "
                 "Human scheme auto-detected: correctness (abstention/correct/incorrect)."),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Judge / pipeline validation — Cohen's kappa vs human labels (Task F)\n",
        f"Hand-labeled rows: **{n_labeled}** (matched to data: {n_match}).\n",
        "| Comparison | Cohen's kappa | Agreement |",
        "|---|---|---|",
        f"| Human vs pipeline response-category (3-way) | **{out['kappa_human_vs_pipeline_category_3way']}** | {out['agreement_human_vs_pipeline_category_pct']}% |",
        f"| Human vs external Judge (hallucination) | **{out['kappa_human_vs_external_judge_hal']}** | {out['agreement_human_vs_judge_pct']}% |",
        f"| Human vs EM (hallucination) | **{out['kappa_human_vs_em_hal']}** | {out['agreement_human_vs_em_pct']}% |",
        "\n*Human labels: correctness scheme (abstention/correct/incorrect). Pipeline uses the "
        "extended, Haiku-safe abstention detector. Judge = qwen3-235b-a22b-instruct-2507.*\n",
    ]
    with io.open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"labeled={n_labeled} matched={n_match}")
    print(f"kappa human-vs-pipeline (3-way) = {out['kappa_human_vs_pipeline_category_3way']} "
          f"(agree {out['agreement_human_vs_pipeline_category_pct']}%)")
    print(f"kappa human-vs-judge = {out['kappa_human_vs_external_judge_hal']} "
          f"(agree {out['agreement_human_vs_judge_pct']}%)")
    print(f"kappa human-vs-EM = {out['kappa_human_vs_em_hal']} "
          f"(agree {out['agreement_human_vs_em_pct']}%)")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
