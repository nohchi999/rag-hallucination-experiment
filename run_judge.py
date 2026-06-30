"""
run_judge.py  (Tasks C2 + D-judge, added 2026-06-29)

Standalone external-judge pass. Applies the fixed external judge
(Qwen2.5-72B-Instruct via OpenRouter, temp 0) over the stored answers of ANY raw
results file, writing NEW judge fields and keeping everything else (auditability):
    judge_verdict_ext, judge_reasoning_ext, is_hallucinated_judge_ext, judge_model_ext

Why a separate pass:
  * The judge is a post-hoc step, not part of generation. Running it standalone lets
    generation (GPT/Gemini) run in parallel now and the judge run later once the
    OpenRouter/Qwen key exists.
  * ALL three generators (Haiku, GPT, Gemini) are judged by the SAME judge — the
    _ext fields are read uniformly at analysis time.

Usage (parallel-safe; each input -> its own output, no collisions):
  python run_judge.py --input results/raw_results.json                 # Haiku (C2)
  python run_judge.py --input results/cross_model/raw_results_openai.json
  python run_judge.py --input results/cross_model/raw_results_google.json

Resumable: re-running skips already-judged (qid, prompt_type, condition) triples.
NEVER modifies the input file.
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
from src import judge_external

CHECKPOINT_EVERY = 10


def _load_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, obj):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _key(r):
    return (r["question_id"], r["prompt_type"], r["condition"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw results JSON to judge.")
    ap.add_argument("--out", default=None, help="Output path (default: <stem>_judged.json in same dir).")
    args = ap.parse_args()

    if not config.JUDGE_API_KEY:
        print("No judge API key set (JUDGE_API_KEY / OPENROUTER_API_KEY / DASHSCOPE_API_KEY) "
              "— cannot run the external Qwen judge yet.")
        sys.exit(1)

    in_path = args.input
    if args.out:
        out_path = args.out
    else:
        d, base = os.path.split(in_path)
        out_path = os.path.join(d, base.replace(".json", "") + "_judged.json")
    manifest_path = out_path.replace(".json", "") + "_manifest.json"

    raw = _load_json(in_path)
    print(f"Judging {len(raw)} records from {in_path} with {config.JUDGE_MODEL}")

    done = {}
    if os.path.exists(out_path):
        for r in _load_json(out_path):
            if r.get("is_hallucinated_judge_ext") is not None:
                done[_key(r)] = r
        print(f"Resuming: {len(done)} already judged.")

    providers.reset_usage()
    out, n_new = [], 0
    for i, r in enumerate(raw):
        k = _key(r)
        if k in done:
            out.append(done[k])
            continue
        verdict, reasoning, is_hal = judge_external.judge_answer(
            question=r["question"],
            context_chunks=r.get("retrieved_chunks", []),
            answer=r.get("answer", ""),
        )
        rec = dict(r)
        rec["judge_verdict_ext"] = verdict
        rec["judge_reasoning_ext"] = reasoning
        rec["is_hallucinated_judge_ext"] = is_hal
        rec["judge_model_ext"] = config.JUDGE_MODEL
        rec["rejudge_timestamp"] = datetime.utcnow().isoformat()
        out.append(rec)
        n_new += 1
        if n_new % CHECKPOINT_EVERY == 0:
            _save_json(out_path, out)
            print(f"  [{i+1}/{len(raw)}] checkpoint ({n_new} new)")

    _save_json(out_path, out)

    agg = providers.usage_totals()
    cost = 0.0
    usage_report = {}
    for (prov, model), u in agg.items():
        c = (u["input_tokens"] / 1e6) * config.JUDGE_PRICE_IN + (u["output_tokens"] / 1e6) * config.JUDGE_PRICE_OUT
        cost += c
        usage_report[f"{prov}:{model}"] = {**u, "cost_usd": round(c, 4)}

    _save_json(manifest_path, {
        "task": "external_judge_pass",
        "date": datetime.utcnow().isoformat(),
        "input": in_path,
        "output": out_path,
        "judge_provider": config.JUDGE_PROVIDER,
        "judge_model": config.JUDGE_MODEL,
        "judge_base_url": config.JUDGE_BASE_URL,
        "judge_provider_pin": config.JUDGE_PROVIDER_PIN or "(unpinned)",
        "judge_temperature": config.JUDGE_TEMPERATURE,
        "records_total": len(raw),
        "records_new_this_run": n_new,
        "usage": usage_report,
        "total_cost_usd": round(cost, 4),
    })
    print(f"Done. Wrote {out_path} ({len(out)} records). New judge calls: {n_new}. Cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
