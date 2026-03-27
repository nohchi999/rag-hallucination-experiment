# RAG Hallucination Experiment

Experiment for the Master's thesis:
**"Hallucination Under Incomplete Evidence: Evaluating Black-Box Uncertainty Estimation Methods in Retrieval-Augmented Generation Systems"**

## What it does

Runs 200 factoid questions from **SQuAD** across a **2×3 factorial design**:
- **Factor 1 — Prompt Type:** Constrained (with abstention instruction) vs. Unconstrained (no abstention instruction)
- **Factor 2 — Evidence Condition:** Full, Partial, No Evidence

Generates answers with Claude Haiku 4.5 and evaluates 3 black-box uncertainty estimation methods: Verbalized Confidence, SelfCheckGPT (NLI-based), and LLM-as-Judge.

Ground truth uses a **dual-label system**: EM-based (objective) and Judge-based (semantic) to avoid circularity when computing AUROC.

## Installation

```bash
pip install -r requirements.txt
```

## API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run the experiment

```bash
python run_experiment.py
```

- Resumes automatically from checkpoint if interrupted
- Saves progress to `results/checkpoint.json` after every question
- Final output: `results/raw_results.json` (1200 data points)
- Estimated runtime: 4–8 hours
- Estimated cost: ~$4–10 (Claude Haiku)
- Total API calls: ~8,400 (200 questions × 2 prompt types × 3 conditions × 7 calls)

## Analyze results

```bash
python analyze_results.py
```

Produces:
- `results/table1_hallucination_rate.csv`
- `results/table2_auroc.csv`
- `results/table3_calibration.csv`
- `results/table4_agreement.csv`
- `results/fig1_hallucination_rate.png`
- `results/fig2_auroc_comparison.png`
- `results/fig3_calibration.png`
- `results/fig4_confidence_distribution.png`

## Configuration

All parameters are in `config.py`. To do a quick test run, temporarily set `NUM_QUERIES = 3` there.

## Project structure

```
rag-hallucination-experiment/
├── config.py               # All experiment parameters
├── run_experiment.py       # Main script (2×3 factorial loop)
├── analyze_results.py      # Analysis, tables, figures
├── requirements.txt
├── src/
│   ├── dataset.py          # SQuAD loading + filtering
│   ├── vectorstore.py      # ChromaDB setup + evidence retrieval
│   ├── generator.py        # Claude Haiku + verbalized confidence
│   ├── selfcheck.py        # SelfCheckGPT (NLI-based)
│   ├── judge.py            # LLM-as-Judge
│   ├── metrics.py          # AUROC, ECE, Exact Match, Cohen's Kappa, Wilson CI
│   └── visualize.py        # matplotlib figures (stratified 2×3)
├── prompts/
│   ├── generation_constrained.txt    # With abstention instruction
│   ├── generation_unconstrained.txt  # Without abstention instruction
│   └── llm_judge.txt
├── data/
│   ├── filtered_squad.json  # 200 SQuAD questions (1 per article)
│   └── chroma_db/           # Auto-generated
└── results/                 # All outputs (auto-generated)
```
