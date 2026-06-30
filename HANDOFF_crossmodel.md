# Handoff — Cross-Model Extension of the RAG-Hallucination Thesis

**For:** the main thesis-writing chat. **Scope of this work:** extended the existing, finished
single-model experiment (Claude Haiku 4.5) to three generator models and swapped the judge to a
fixed external model, then validated everything. The thesis *prose* was NOT touched — this is the
experiment/code/results layer only. Everything below is committed to `master` of
`rag-hallucination-experiment` (GitHub: nohchi999/rag-hallucination-experiment).

---

## 1. What was done, in one paragraph

The full 2×3 factorial experiment (prompt_type × evidence) was replicated across **three
generators** — Claude Haiku 4.5 (existing anchor), GPT-4.1-mini, Gemini-2.5-flash — with the
**generator as the only varying factor**. The LLM-as-Judge was moved off Haiku onto a fixed
**external third-family model** (Qwen via Alibaba Model Studio), used identically for all three
generators, and Haiku's stored answers were **re-judged** by it (no regeneration). A 144-row human
labelling validated the judge/pipeline (Cohen's κ). All four findings (F1–F4) were recomputed per
model and synthesised. Total new API cost ≈ **$3.07**.

---

## 2. Models (exact pinned versions)

| Role | Model | Pinned id | Notes |
|---|---|---|---|
| Generator (anchor) | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | NOT regenerated; only re-judged |
| Generator | GPT-4.1-mini | `gpt-4.1-mini-2025-04-14` | OpenAI; Haiku size class |
| Generator | Gemini 2.5 Flash | `gemini-2.5-flash` | Google; **thinking disabled** (see §4) |
| External judge | Qwen3-235B-A22B-Instruct | `qwen3-235b-a22b-instruct-2507` | Alibaba Model Studio (Singapore), temp 0, non-thinking |

---

## 3. What is NEW vs the original repo (file-by-file)

**New source modules** (all additive; the original `src/generator.py`, `judge.py`, `selfcheck.py`,
`metrics.py`, `metrics_v2.py`, `analyze_results*.py`, prompts and `raw_results.json` are
**byte-identical / untouched**):
- `src/providers.py` — unified `chat()` over Anthropic / OpenAI / Google / OpenAI-compatible judge endpoints; normalises every backend to the same return shape (always preserves `raw_text`); token-usage accounting.
- `src/generator_multi.py` — generator routed through `providers`; **confidence is never defaulted to 50** (missing → `None`, propagated); format-tolerant `Answer:/Confidence:` parser across model styles.
- `src/selfcheck_multi.py` — per-model SelfCheck sampling (same generator, T=0.7), **same fixed NLI scorer** reused unchanged.
- `src/judge_external.py` — external judge call; **judge prompt unchanged**; verdict parsing reused verbatim from `src/judge.py`.
- `src/abstention_ext.py` — **Refinement-2** abstention-marker extension (see §4); additive, `metrics_v2.py` untouched.

**New scripts:**
- `run_experiment_multi.py` — full 2×3 generation for GPT/Gemini; **reuses Haiku's retrieved chunks bitwise**; generation only (judge deferred); per-question checkpointing; writes manifest.
- `run_judge.py` — standalone external-judge pass over ANY raw file → adds `*_ext` judge fields (keeps originals); used for Haiku (C2) and both new models.
- `build_validation_sheet.py` — stratified 144-row human-labelling sheet (no auto-labelling).
- `analyze_crossmodel.py` — runs the **validated v2 pipeline per model** with the external judge remapped in + the extended abstention detector; writes per-model tables + cross-model synthesis.
- `compute_human_kappa.py` — Cohen's κ of the human labels vs pipeline/judge/EM.

**Modified (additively):** `config.py` (single generator switch + pinned model registry + judge config + pricing; existing values untouched), `requirements.txt` (+`openai`, `google-genai`, `openpyxl`), `.gitignore`.

**New data/results:** `results/raw_results_judged.json` (Haiku re-judged), `results/cross_model/raw_results_{openai,google}.json` (+ `_judged` + manifests), `results/cross_model/analysis/<model>/table1-4.csv`, `cross_model_summary.{csv,md}`, `human_kappa.{json,md}`, `validation_sheet.csv`, `validation_sheet_labeled.xlsx`, `RESULTS_SECTION_DRAFT.md`.

---

## 4. Decisions & deviations from the briefing (important for the write-up)

1. **Generators chosen:** GPT-4.1-mini and Gemini-2.5-flash, both selected as the Haiku **size class**, with pinned (non-`-latest`) version strings.
2. **Gemini thinking disabled** (`thinking_budget=0`): Gemini-2.5-flash is a hybrid-reasoning model with thinking on by default. It was disabled so Gemini behaves as a plain instruct generator comparable to Haiku and so output cost/length is controlled. Documented design choice.
3. **Judge model — DEVIATION (must be stated in the thesis):** the briefing's locked choice was **Qwen2.5-72B-Instruct**, but that model is **not offered** on the user's Alibaba Model Studio account/region (Singapore catalog has only Qwen3.5/3.6/3.7 + some Qwen3; a direct test returned 403). The substitute is **`qwen3-235b-a22b-instruct-2507`**, which satisfies the briefing's actual requirements: Qwen family, **non-thinking INSTRUCT** variant (explicitly *not* the `-thinking-2507` sibling), pinned dated version, temp 0, third family vs Claude/GPT/Gemini, first-party (not quantized). Endpoint: `dashscope-intl.aliyuncs.com/compatible-mode/v1`.
4. **Generation/judging split into separate scripts** (user request) so GPT and Gemini ran in parallel; the judge ran as a separate uniform pass over all three models afterwards.
5. **Evidence reuse:** instead of re-querying ChromaDB for the new models, the exact `retrieved_chunks` from Haiku's `raw_results.json` were reused per cell → evidence is **bitwise identical** across models (strongest form of "single varying factor"; no retrieval drift).
6. **Refinement-2 abstention markers — applied UNIFORMLY, Haiku-safe:** human labelling revealed the auto-detector missed GPT/Gemini abstentions phrased *"The provided text does not …"* (original markers say "context"), over-counting their hallucinations. Per the user's directive that the experiment must be **exactly like Haiku** — and noting that the Haiku v2 pipeline itself fixed abstention under-detection by extending markers — the detector was extended with **one anchored opener pattern**, applied **uniformly to all three models**. **Proven Haiku-safe: 0 Haiku rows change; Haiku EM table bit-identical (F1 = 23.74 % → 9.09 %).** Effect: GPT 0 rows, Gemini 87 rows (genuine abstentions). Validated against human labels: human-vs-pipeline κ rose 0.818 → 0.872.
7. **Human label scheme:** the labeller used a **correctness** scheme (`abstention` / `correct` / `incorrect`) rather than the sheet's requested **groundedness** scheme (`supported` / `partially_supported` / `not_supported`). Both are handled; this is *why* human-vs-judge κ (0.58) is lower than human-vs-pipeline κ (0.87) — they measure different constructs (answer correctness vs context groundedness).

---

## 5. Invariants preserved (unchanged from the Haiku study)

Question set (198/cell + quality filter), ChromaDB index (all-MiniLM-L6-v2, chunk 500/overlap 50,
top-k=3, cosine), evidence operationalisation (full/partial/none), prompts and the
`Answer: … Confidence: N` format, SelfCheckGPT (5×, T=0.7, NLI `cross-encoder/nli-deberta-v3-small`),
the validated v2 analysis pipeline with all six refinements, anti-circularity rule (judge-UE AUROC
vs EM only), confidence missingness never defaulted to 50. **Haiku's headline F1 is bit-identical.**

---

## 6. Results (committed; see `results/cross_model/analysis/`)

**Judge / pipeline validation (n=144 human labels):** human vs pipeline-classification κ = **0.872**
(92.4 %); human vs EM κ = 0.731; human vs external judge κ = 0.576.

**F1 — constrained prompting reduces EM-hallucination under partial evidence — REPLICATES (all 3):**
Haiku 23.74 % → 9.09 %; GPT 28.28 % → 16.67 %; Gemini 17.68 % → 6.06 %.

**F2 — population-dependent UE evaluation / VC inversion — replicates directionally** (small
attempted-n cells, e.g. n=3/4/16, are reported but NOT point-compared; VC inversion AUROC<0.5 appears
model-specifically: Haiku uncon/partial 0.17, Gemini const/partial 0.04).

**F3 — overconfidence on attempted answers under reduced evidence — REPLICATES (all 3)**
(ECE non-abstention in constrained/partial: Haiku 0.58, GPT 0.68, Gemini 0.73).

**F4 — prompt-conditional abstention-confidence asymmetry — DIVERGENT / model-dependent.** Haiku
reproduces it (constrained abstentions 76.8 % at conf≥90; unconstrained 92.1 % at conf≤10); GPT
partial; **Gemini inverts it** (constrained 93.8 % at conf≤10; unconstrained 66.9 % at conf≥90).

**Dual-label divergence (EM vs judge):** largest under reduced evidence + unconstrained (e.g. Gemini
uncon/partial: EM-hal 17.68 % vs judge-hal ≈2 %; per-cell κ down to 0.06). Model-dependent.

**Synthesis:** F1, F3 = robustly replicated; F2 = directional; F4 = model-dependent. Take-away:
models are similarly steerable away from hallucination, but differ sharply in *how* they abstain.

A ready-to-adapt write-up is in `results/cross_model/RESULTS_SECTION_DRAFT.md`.

---

## 7. Cost, reproducibility, open items

- **Cost:** GPT gen $1.23, Gemini gen $1.23, three judge passes ≈ $0.20 each ≈ **$3.07** total. (Judge manifest cost figures use a placeholder price; real token counts are logged for exact recompute.)
- **Reproduce:** `git pull`; recreate `.env` (keys are gitignored — Anthropic/OpenAI/Gemini + `JUDGE_API_KEY` for Alibaba; `JUDGE_PROVIDER=dashscope`, `JUDGE_MODEL=qwen3-235b-a22b-instruct-2507`, `JUDGE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1`); `pip install -r requirements.txt`. Analysis re-runs with `python analyze_crossmodel.py` and `python compute_human_kappa.py` (no API needed). Generation/judge re-runs cost money.
- **Open items for the thesis:** (a) state the judge-model deviation (Qwen2.5-72B → qwen3-235b-a22b-instruct-2507) and why; (b) note F4 divergence as a genuine cross-model finding (supports Feng et al. 2025 model-dependence); (c) the judge price placeholder if exact cost is reported; (d) optionally a second judge as a robustness check (out of scope here).

---

## 8. Commits (this session, on `master`)

```
09c35c8  cross-model extension: GPT + Gemini generators, external Qwen judge
ce51b26  cross-model analysis (Task F label-independent + G synthesis)
de0c77a  compute_human_kappa.py
acb7279  Refinement-2 abstention markers + judge validation (Tasks F/G complete)
d8156a9  remove stray stackdump; gitignore
5f30fec  results-section draft
```
