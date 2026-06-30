"""
run_experiment_multi.py  (Task D, added 2026-06-29)

Full 2x3 generation for a NEW generator model (OpenAI or Google) ONLY. Haiku is
NOT regenerated. The model is chosen by config.GENERATOR_PROVIDER /
config.GENERATOR_MODEL (or --provider).

Invariant: evidence is the SINGLE varying factor's complement. To guarantee the
new model sees BITWISE-IDENTICAL evidence to Haiku, retrieved_chunks (and the
question/ground_truth/cell) are REUSED from results/raw_results.json rather than
re-querying ChromaDB. This isolates the generator as the only changed factor and
avoids any retrieval/embedding drift.

Per item (~1200 = 200 q x 6 cells) the calls are:
  1 generation (temp 0) + 5 SelfCheck samples (temp 0.7) + 1 external judge.

Output:   results/cross_model/raw_results_<provider>.json   (+ manifest)
Checkpointing after every full question (6 cells). Resumable.

Run:  python run_experiment_multi.py --provider openai
      python run_experiment_multi.py --provider google
"""

import argparse
import io
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src import providers
from src import generator_multi
from src import selfcheck_multi
from src import metrics_v2  # judge is a separate pass (run_judge.py)

RAW_HAIKU = config.RAW_RESULTS_FILE
OUT_DIR = "./results/cross_model"


def _load_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, obj):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _key(r):
    return (r["question_id"], r["prompt_type"], r["condition"])


def process_one(src_rec):
    """Generate one (question, prompt_type, condition) with the configured model,
    reusing src_rec's evidence. Returns a raw record matching the pipeline schema."""
    question = src_rec["question"]
    ground_truth = src_rec["ground_truth"]
    prompt_type = src_rec["prompt_type"]
    condition = src_rec["condition"]
    chunks = src_rec.get("retrieved_chunks", [])

    # 1) main answer + verbalized confidence (None if missing — never 50)
    answer, confidence, full_response = generator_multi.generate_with_confidence(
        question=question, context_chunks=chunks,
        temperature=config.TEMPERATURE_DETERMINISTIC, prompt_type=prompt_type,
    )

    # 2) SelfCheck (same model, temp 0.7) + fixed NLI scorer
    try:
        samples = selfcheck_multi.selfcheck_sample(question, chunks, prompt_type=prompt_type)
        consistency, uncertainty = selfcheck_multi.compute_selfcheck_score(answer, samples)
    except Exception as e:
        samples, consistency, uncertainty = [], 0.0, 1.0
        print(f"  SelfCheck failed q={src_rec['question_id']} {prompt_type}/{condition}: {e}")

    # 3) JUDGE IS DEFERRED: the external Qwen judge runs as a separate pass
    #    (run_judge.py) once the OpenRouter key exists, applied uniformly to all
    #    three models. Judge fields are left pending here.

    # 4) EM-side labels via the validated (v2) functions, for internal consistency.
    em = metrics_v2.exact_match(answer, ground_truth)
    abstention = metrics_v2.detect_abstention_v2(answer)
    is_hal_em = 1 if (em == 0 and not abstention and answer not in ("", "ERROR")) else 0

    return {
        "question_id": src_rec["question_id"],
        "question": question,
        "ground_truth": ground_truth,
        "prompt_type": prompt_type,
        "condition": condition,
        "retrieved_chunks": chunks,
        "answer": answer,
        "verbalized_confidence": confidence,            # int or None (never 50)
        "selfcheck_samples": samples,
        "selfcheck_consistency": round(consistency, 4),
        "selfcheck_uncertainty": round(uncertainty, 4),
        "judge_verdict_ext": None,                       # filled later by run_judge.py
        "judge_reasoning_ext": None,
        "is_hallucinated_judge_ext": None,
        "is_hallucinated_em": is_hal_em,
        "is_abstention": abstention,
        "exact_match": em,
        "generator_provider": config.GENERATOR_PROVIDER,
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "full_api_response": full_response,              # carries raw_text for v2 reparse
        "timestamp": datetime.utcnow().isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "google"], default=None,
                    help="Override config.GENERATOR_PROVIDER for this run.")
    args = ap.parse_args()
    if args.provider:
        config.GENERATOR_PROVIDER = args.provider
        config.GENERATOR_MODEL = config.GENERATOR_REGISTRY[args.provider]["model"]

    provider = config.GENERATOR_PROVIDER
    if provider == "anthropic":
        print("Refusing to regenerate Haiku (provider=anthropic). Use rejudge_haiku.py for C2.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"raw_results_{provider}.json")
    manifest_file = os.path.join(OUT_DIR, f"manifest_{provider}.json")

    haiku = _load_json(RAW_HAIKU)
    print(f"Reusing evidence from {len(haiku)} Haiku records. Generator: {config.GENERATOR_MODEL}")

    done = {}
    if os.path.exists(out_file):
        for r in _load_json(out_file):
            done[_key(r)] = r
        print(f"Resuming: {len(done)} cells already generated.")

    providers.reset_usage()
    out = []
    n_new = 0
    # group by question to checkpoint after each full question (6 cells)
    by_q = {}
    for r in haiku:
        by_q.setdefault(r["question_id"], []).append(r)

    for qid in sorted(by_q):
        for src_rec in by_q[qid]:
            k = _key(src_rec)
            if k in done:
                out.append(done[k])
                continue
            try:
                out.append(process_one(src_rec))
            except Exception as e:
                print(f"  FAILED q={qid} {src_rec['prompt_type']}/{src_rec['condition']}: {e}")
                out.append({**{kk: src_rec.get(kk) for kk in
                               ("question_id", "question", "ground_truth", "prompt_type", "condition")},
                            "error": str(e), "answer": "ERROR", "verbalized_confidence": None,
                            "retrieved_chunks": src_rec.get("retrieved_chunks", []),
                            "full_api_response": {"error": str(e)},
                            "timestamp": datetime.utcnow().isoformat()})
            n_new += 1
        _save_json(out_file, out)  # checkpoint after each full question
        print(f"  q={qid} done ({n_new} new cells)")

    # manifest with real usage/cost
    agg = providers.usage_totals()
    g = config.GENERATOR_REGISTRY[provider]
    cost = 0.0
    usage_report = {}
    for (prov, model), u in agg.items():
        if prov == provider:
            p_in, p_out = g["price_in"], g["price_out"]
        else:  # judge
            p_in, p_out = config.JUDGE_PRICE_IN, config.JUDGE_PRICE_OUT
        c = (u["input_tokens"] / 1e6) * p_in + (u["output_tokens"] / 1e6) * p_out
        cost += c
        usage_report[f"{prov}:{model}"] = {**u, "cost_usd": round(c, 4)}

    _save_json(manifest_file, {
        "task": "D_generate",
        "date": datetime.utcnow().isoformat(),
        "generator_provider": provider,
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "judge_provider_pin": config.JUDGE_PROVIDER_PIN or "(unpinned)",
        "google_thinking_budget": config.GOOGLE_THINKING_BUDGET if provider == "google" else None,
        "evidence_source": "reused from results/raw_results.json (bitwise-identical to Haiku)",
        "cells_total": len(haiku),
        "cells_new_this_run": n_new,
        "usage": usage_report,
        "total_cost_usd": round(cost, 4),
    })
    print(f"Done. Wrote {out_file}. New cells: {n_new}. Cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
