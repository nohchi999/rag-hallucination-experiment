# Cross-Model Extension — Execution Plan

Briefing: Cross-Model-Erweiterung der Masterarbeit (RAG / Hallucination / UE).
Single varying factor = generator model. Everything else (RAG params, questionset,
evidence operationalization, prompts, NLI scorer, validated pipeline) stays bit-identical.

## Model decisions (locked with user 2026-06-29)
- Generator 1 (existing anchor): Claude Haiku 4.5 `claude-haiku-4-5-20251001` — NOT regenerated.
- Generator 2 (OpenAI): **gpt-4.1-mini** (Haiku size class). Pinned snapshot TBD from research.
- Generator 3 (Google): **gemini-2.5-flash** (Haiku size class). Pinned string TBD from research.
- External judge (3rd family, non-thinking, temp 0): **Qwen2.5-72B-Instruct** via EU-routable
  OpenAI-compatible provider (Together AI / OpenRouter — TBD from research, documented in manifest).

## Task A — Baseline reproduction  [DONE]
- [x] Repo map (a–g modules) built.
- [x] Text↔code divergences flagged (judge docstring "3.5"; generator 50-default lives only in v1
      parser, analysis layer uses metrics_v2.reparse → None; abstention detection is metrics_v2 post-hoc).
- [x] Baseline reproduced bit-identically: F1 = 23.74% → 9.09%, n=198/cell, cells sum 100%, 20/20 tests.

## Task B — Generator abstraction  [CODE DONE]
- [x] Provider layer `src/providers.py`: `chat(provider, model, ...) -> (text, full_dict)`, raw_text always preserved, usage accounting.
- [x] Providers: anthropic (call shape identical to original), openai, google (thinking disabled).
- [x] config: single `GENERATOR_PROVIDER` + `GENERATOR_MODEL` switch + pinned strings + pricing registry.
- [x] Per-provider temperature control (0.0 main / 0.7 selfcheck via selfcheck_multi).
- [x] `src/generator_multi.parse_response`: format-tolerant, missing conf -> None (8/8 offline tests pass).
- [ ] Refinement-2 markers: DEFER finalization to Task F — only add GPT/Gemini markers that do NOT
      reclassify any stored Haiku row (verify against raw_results), so Haiku stays bit-identical.

## Task C — Externalize judge  [CODE DONE]
- [x] `src/judge_external.py`: Qwen via OpenAI-compatible OpenRouter, temp 0, pinned, single-backend pin option.
- [x] Judge prompt UNCHANGED; verdict parsing reused verbatim from src/judge.py.

## RESTRUCTURE (user request): split generation from judging for parallel runs
- Generation (run_experiment_multi.py) and judging (run_judge.py) are now SEPARATE scripts.
- Reason: Qwen/OpenRouter key not set up yet -> run GPT+Gemini generation NOW in parallel,
  run the external judge later as one uniform pass over all 3 models (Haiku, GPT, Gemini).
- Keys: OpenAI + Gemini pulled from Desktop/api-key.txt into .env (gitignored). OPENROUTER pending.

## Task C2 + D-judge — external judge pass  [CODE DONE — awaiting OpenRouter/Qwen key]
- [x] `run_judge.py --input <raw>`: judges ANY raw file, writes _ext fields, checkpoint+resume, manifest.
      Supersedes rejudge_haiku.py (removed). Run for all 3: results/raw_results.json + the 2 new files.

## Task D — generation for GPT + Gemini ONLY  [DONE 2026-06-30]
- [x] GPT-4.1-mini: 1200 cells, 0 errors, all conf parsed, cost $1.23 -> raw_results_openai.json
- [x] Gemini-2.5-flash: 1200 cells, 0 errors, all conf parsed, cost $1.23 -> raw_results_google.json
- [x] External Qwen judge run over BOTH + Haiku (C2): *_judged.json with _ext labels. Judge cost ~$0.20 each.
- Preview (EM, pre-validated-pipeline): prompt mitigation REPLICATES all 3 models, magnitude differs —
  Gemini strongest (39.5%->6.0%), Haiku 23.7%->9.1%, GPT weakest (28.0%->16.5%). Confirm in Task F.

## Task E — validation sheet  [DONE — awaiting human labels]
- [x] validation_sheet.csv: 144 rows (48/model, 48/condition, 70 abstention / 74 attempted), human_label empty.
- >>> GATE: handed to user for hand-labeling. NO auto-labeling. <<<

## Task F (label-independent) + G — [DONE 2026-06-30]  analyze_crossmodel.py
- [x] Validated v2 pipeline run per model with SAME external judge (remap _ext -> canonical judge fields).
      Anti-circularity intact (judge-UE AUROC vs EM only). Tables -> results/cross_model/analysis/<model>/.
- [x] Haiku reproduces BIT-IDENTICALLY (23.74/9.09) -> confirms remap + data integrity.
- [x] F1 prompt mitigation REPLICATES all 3: Haiku 23.74->9.09, GPT 28.28->16.67, Gemini 39.9->6.06.
- [x] EM-vs-extJudge kappa per cell (table4): big model-dependent dual-label divergence
      (e.g. Gemini uncon/partial agreement 62%, kappa 0.06 — judge lenient vs strict EM).
- [x] cross_model_summary.{csv,md} (Task G F1 synthesis).

## Task F (human kappa) — [STILL PENDING human labels]
- [ ] compute human-vs-judge & human-vs-EM Cohen's kappa from labeled validation_sheet.csv.
- [ ] (to write) compute_human_kappa.py once labels returned.

## JUDGE MODEL DEVIATION (decided with user 2026-06-29)
- Locked choice Qwen2.5-72B-Instruct is NOT offered on the user's Alibaba Model Studio
  account (region ap-southeast-1 / Singapore). Catalog only has Qwen3.5/3.6/3.7 + some Qwen3.
- 403 access_denied on qwen2.5-72b-instruct; qwen-plus/qwen-turbo work (auth/endpoint fine).
- DECISION: judge = `qwen3-235b-a22b-instruct-2507` (Alibaba Model Studio, dashscope-intl endpoint).
  Satisfies briefing intent: Qwen family, NON-thinking INSTRUCT variant (not the -thinking-2507
  sibling), pinned dated version, temp 0, 3rd family vs Claude/GPT/Gemini. First-party (not quantized).
- Endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1 ; key prefix sk-ws- (pay-as-you-go).
- Validated: correct-answer->supported, honest-abstention->supported; early old-vs-new label agree ~97%.
- NOTE: JUDGE_PRICE_IN/OUT in config still 0.36/0.40 (2.5-72B figures) — token usage IS logged, so
  manifest cost is recomputable once exact qwen3-235b-a22b-instruct-2507 price is confirmed.

## Change-log (text/code deviations + additive changes)
- Originals (generator/judge/selfcheck/metrics/run/prompts/raw_results) left BYTE-IDENTICAL; all new code in new modules.
- config.py extended additively (the briefing's intended single switch surface); existing values untouched.
- requirements.txt += openai, google-genai (unavoidable per briefing; legacy google-generativeai NOT used).
- Deviation noted: judge.py docstring says "Haiku 3.5" but runs config.MODEL_NAME (4.5). Not changed.
- Deviation noted: original generator.parse_response defaults conf to 50; invariant held only at analysis layer.
  New generator_multi fixes this at source (None). Haiku raw untouched; analysis still uses reparse.
- Design choice: Gemini 2.5-flash thinking disabled (thinking_budget=0) so it is a plain instruct generator
  comparable to Haiku and output cost is controlled. Recorded in run manifest.

## Task E — Validation sheet, then STOP  [GATE: human labels]
- [ ] Stratified ~120–150 sample across models/prompt/evidence incl. abstentions + attempted.
- [ ] Export labeling sheet (q, GT, context chunks, model answer, empty human_label). Hand off. NO auto-labeling.

## Task F — Kappa + full analysis  [after human labels returned]
- [ ] Cohen's kappa (hand vs judge, hand vs EM). Run validated pipeline per model; all tables.

## Task G — Cross-model synthesis  [TODO]
- [ ] Model × Finding × {replicated / partial / divergent} table + directional F2 (report attempted-n,
      no point-AUROC for n < 20; bootstrap CIs 1000 resamples).

## Guardrails
- Code wins over thesis text; flag divergence, don't blind-change.
- Never fabricate human labels. Never default confidence to 50. Judge-UE AUROC vs EM only.
- Don't touch RAG params / embeddings / chunking / questionset / evidence ops / NLI scorer.
