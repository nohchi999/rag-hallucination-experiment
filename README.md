# RAG Hallucination Experiment

Experiment for the Master's thesis:
**"Hallucination Under Incomplete Evidence: Evaluating Black-Box Uncertainty Estimation Methods in Retrieval-Augmented Generation Systems"**

## What it does

Runs 198 factoid questions/cell from **SQuAD** across a **2×3 factorial design**:
- **Factor 1 — Prompt Type:** Constrained (with abstention instruction) vs. Unconstrained (no abstention instruction)
- **Factor 2 — Evidence Condition:** Full, Partial, No Evidence

For each cell it generates an answer and evaluates 3 black-box uncertainty estimation methods: Verbalized Confidence, SelfCheckGPT (NLI-based), and LLM-as-Judge. Ground truth uses a **dual-label system** — EM-based (objective substring match) and Judge-based (semantic) — with the anti-circularity rule that the Judge-as-UE-method is scored against EM only.

The project has **three layers**:
1. **Single-model experiment** (Claude Haiku 4.5) — the original run.
2. **Validated reanalysis pipeline (v2)** — fixes six metrics bugs; the analysis of record.
3. **Cross-model extension** — replicates the full experiment across three generators (Claude / GPT / Gemini) with a fixed external judge (Qwen), plus human-label judge validation. See [`HANDOFF_crossmodel.md`](HANDOFF_crossmodel.md).

## Installation

```bash
pip install -r requirements.txt
```

## API keys

Create a `.env` file in the repo root (gitignored — never committed):

```
ANTHROPIC_API_KEY=sk-ant-...
# Cross-model extension:
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=...
# External judge (Alibaba Model Studio / DashScope, OpenAI-compatible):
JUDGE_API_KEY=sk-...
JUDGE_PROVIDER=dashscope
JUDGE_MODEL=qwen3-235b-a22b-instruct-2507
JUDGE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

`config.py` reads `.env` automatically. Only the keys for the layer you run are required.

---

## Layer 1 — Single-model experiment (Claude Haiku 4.5)

```bash
python run_experiment.py
```
- Resumes from checkpoint; saves after every question. Final output: `results/raw_results.json` (1200 data points).
- ~8,400 API calls (200 q × 2 prompt types × 3 conditions × 7 calls); ~4–8 h; ~$4–10.

## Layer 2 — Validated reanalysis pipeline (v2)

Re-analyses `raw_results.json` with six bug fixes (confidence re-parse → never defaults to 50; extended abstention detection; disjoint response classification; SQuAD-F1; degenerate-datapoint filter; ECE variants). Originals are kept untouched for auditability.

```bash
python analyze_results_v2.py      # -> results/reanalysis_v2/table1-4_v2.csv + reports
python tests/test_metrics_v2.py   # 20 standalone assert-based tests
```

## Layer 3 — Cross-model extension (Claude / GPT / Gemini + external Qwen judge)

The generator is the **only** varying factor; evidence is reused bitwise from the Haiku run, and the same external judge scores all three models. No API key → analysis still runs from committed data.

```bash
# 1) Generation for the two new models (parallel-safe; reuses Haiku evidence)
python run_experiment_multi.py --provider openai    # -> results/cross_model/raw_results_openai.json
python run_experiment_multi.py --provider google    # -> results/cross_model/raw_results_google.json

# 2) External-judge pass over ANY raw file (Haiku C2 + both new models)
python run_judge.py --input results/raw_results.json
python run_judge.py --input results/cross_model/raw_results_openai.json
python run_judge.py --input results/cross_model/raw_results_google.json

# 3) Human-label validation sheet (fill the label column by hand, then compute kappa)
python build_validation_sheet.py                    # -> results/cross_model/validation_sheet.csv
python compute_human_kappa.py                        # reads validation_sheet_labeled.xlsx

# 4) Cross-model analysis + synthesis (no API needed)
python analyze_crossmodel.py                          # -> results/cross_model/analysis/
```

Outputs: per-model tables `results/cross_model/analysis/<model>/table1-4.csv`, `cross_model_summary.md`, `human_kappa.md`, and a ready-to-adapt `results/cross_model/RESULTS_SECTION_DRAFT.md`. Models are pinned (`gpt-4.1-mini-2025-04-14`, `gemini-2.5-flash`, judge `qwen3-235b-a22b-instruct-2507`). Total new API cost ≈ $3.

## Configuration

All parameters are in `config.py` (experiment params, the single generator switch `GENERATOR_PROVIDER`, the pinned model registry, and the judge config). For a quick test run, temporarily set `NUM_QUERIES = 3`.

## Project structure

```
rag-hallucination-experiment/
├── config.py                   # All params + generator switch + judge config
├── run_experiment.py           # Layer 1: single-model 2×3 loop
├── analyze_results.py          # Layer 1: original analysis (kept for audit)
├── analyze_results_v2.py       # Layer 2: validated reanalysis pipeline
├── run_experiment_multi.py     # Layer 3: GPT/Gemini generation
├── run_judge.py                # Layer 3: standalone external-judge pass
├── build_validation_sheet.py   # Layer 3: human-labeling sheet
├── compute_human_kappa.py      # Layer 3: judge validation (Cohen's kappa)
├── analyze_crossmodel.py       # Layer 3: per-model tables + synthesis
├── HANDOFF_crossmodel.md       # Full write-up of the cross-model extension
├── requirements.txt
├── src/
│   ├── dataset.py              # SQuAD loading + filtering
│   ├── vectorstore.py          # ChromaDB setup + evidence retrieval
│   ├── generator.py            # Layer 1 generator (Claude Haiku)
│   ├── selfcheck.py            # SelfCheckGPT (NLI-based)
│   ├── judge.py                # LLM-as-Judge (verdict parsing)
│   ├── metrics.py              # Layer 1 metrics
│   ├── metrics_v2.py           # Layer 2 validated metrics
│   ├── providers.py            # Layer 3 unified multi-provider chat
│   ├── generator_multi.py      # Layer 3 provider-routed generator
│   ├── selfcheck_multi.py      # Layer 3 per-model SelfCheck
│   ├── judge_external.py       # Layer 3 external judge
│   ├── abstention_ext.py       # Layer 3 uniform abstention-marker extension
│   └── visualize.py            # matplotlib figures (stratified 2×3)
├── prompts/                    # generation_{constrained,unconstrained}.txt, llm_judge.txt
├── tests/test_metrics_v2.py    # 20 standalone tests
├── data/                       # filtered_squad.json + chroma_db/ (auto-generated)
└── results/
    ├── raw_results.json                 # Layer 1 output
    ├── reanalysis_v2/                    # Layer 2 tables + reports
    └── cross_model/                      # Layer 3 raw data, judged data, analysis/, drafts
```
