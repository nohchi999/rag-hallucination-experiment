"""
build_validation_sheet.py  (Task E, added 2026-06-29)

Produce a stratified human-labeling sheet (~120-150 rows) for judge validation.
Sampling is stratified across model x prompt_type x condition, and within each
stratum balanced between abstentions and attempted answers, so the later
Cohen's-kappa estimate covers all regimes (incl. the small attempted-answer cells).

Only VALID datapoints (metrics_v2.is_valid_datapoint) are eligible — no degenerate
questions. Deterministic (seeded) so the sheet is reproducible.

Output: results/cross_model/validation_sheet.csv
Columns: row_id, model, prompt_type, condition, question_id, question,
         ground_truth, context_chunks, model_answer, model_is_abstention,
         model_exact_match, human_label, notes
The human_label column is EMPTY — the human fills it. NO auto-labeling.

>>> SCOPE BOUNDARY: this script STOPS after writing the sheet. <<<

Run AFTER generation completes:  python build_validation_sheet.py
"""

import csv
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src import metrics_v2

SEED = 42
TARGET_TOTAL = 144          # 3 models x 2 prompt x 3 cond = 18 strata
PER_STRATUM = TARGET_TOTAL // 18   # 8 per stratum, split ~ abstention/attempted

SOURCES = {
    "claude-haiku-4.5": "results/raw_results.json",
    "gpt-4.1-mini":     "results/cross_model/raw_results_openai.json",
    "gemini-2.5-flash": "results/cross_model/raw_results_google.json",
}
OUT_CSV = "results/cross_model/validation_sheet.csv"


def _load(path):
    if not os.path.exists(path):
        return None
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _eligible(r):
    ok, _ = metrics_v2.is_valid_datapoint(r)
    if not ok:
        return False
    return r.get("answer") not in (None, "", "ERROR")


def main():
    rng = random.Random(SEED)
    rows = []
    row_id = 0
    missing = []

    for model_name, path in SOURCES.items():
        data = _load(path)
        if data is None:
            missing.append((model_name, path))
            continue

        for pt in config.PROMPT_TYPES:
            for cond in config.CONDITIONS:
                pool = [r for r in data
                        if r.get("prompt_type") == pt and r.get("condition") == cond and _eligible(r)]
                abst = [r for r in pool if metrics_v2.classify_response(r) == "abstention"]
                attp = [r for r in pool if metrics_v2.classify_response(r) != "abstention"]
                rng.shuffle(abst); rng.shuffle(attp)

                # balanced split, fall back to whichever bucket has data
                half = PER_STRATUM // 2
                picked = abst[:half] + attp[:PER_STRATUM - half]
                if len(picked) < PER_STRATUM:
                    remainder = (abst[half:] + attp[PER_STRATUM - half:])
                    picked += remainder[:PER_STRATUM - len(picked)]

                for r in picked:
                    row_id += 1
                    rows.append({
                        "row_id": row_id,
                        "model": model_name,
                        "prompt_type": pt,
                        "condition": cond,
                        "question_id": r.get("question_id"),
                        "question": r.get("question", ""),
                        "ground_truth": r.get("ground_truth", ""),
                        "context_chunks": "\n---\n".join(r.get("retrieved_chunks", [])),
                        "model_answer": r.get("answer", ""),
                        "model_is_abstention": metrics_v2.classify_response(r) == "abstention",
                        "model_exact_match": metrics_v2.exact_match(r.get("answer", ""), r.get("ground_truth", "")),
                        "human_label": "",   # supported / partially_supported / not_supported  (HUMAN fills)
                        "notes": "",
                    })

    rng.shuffle(rows)  # de-cluster so the labeler isn't biased by stratum order
    # reassign row_id after shuffle for stable reference
    for i, row in enumerate(rows, 1):
        row["row_id"] = i

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["row_id", "model", "prompt_type", "condition", "question_id",
                  "question", "ground_truth", "context_chunks", "model_answer",
                  "model_is_abstention", "model_exact_match", "human_label", "notes"]
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows.")
    if missing:
        print("WARNING — these sources were missing (generation not finished?):")
        for m, p in missing:
            print(f"  {m}: {p}")
        print("Re-run after all three raw files exist for a fully cross-model sheet.")
    print("\nLABELING INSTRUCTIONS (for the human):")
    print("  Fill 'human_label' with one of: supported / partially_supported / not_supported")
    print("  (honest abstention when context lacks the answer = 'supported').")
    print("  >>> Claude STOPS here. No automatic labels are produced. <<<")


if __name__ == "__main__":
    main()
