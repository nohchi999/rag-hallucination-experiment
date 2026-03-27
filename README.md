# RAG Hallucination Experiment

Experiment for the Master's thesis:
**"Hallucination Under Incomplete Evidence: Evaluating Black-Box Uncertainty Estimation Methods in Retrieval-Augmented Generation Systems"**

## What it does

Runs 200 factoid questions from NQ-Open across 3 evidence conditions (Full, Partial, No Evidence), generates answers with Claude Haiku 3.5, and evaluates 3 uncertainty estimation methods: Verbalized Confidence, SelfCheckGPT, and LLM-as-Judge.

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
- Saves progress to `results/checkpoint.json` every 10 questions
- Final output: `results/raw_results.json`
- Estimated runtime: 2–4 hours
- Estimated cost: ~$2–5 (Claude Haiku)
- Total API calls: ~4,200 (200 questions × 3 conditions × 7 calls)

## Analyze results

```bash
python analyze_results.py
```

Produces:
- `results/table1_hallucination_rate.csv`
- `results/table2_auroc.csv`
- `results/table3_calibration.csv`
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
├── run_experiment.py       # Main script
├── analyze_results.py      # Analysis & figures
├── requirements.txt
├── src/
│   ├── dataset.py          # NQ-Open loading + Wikipedia fetch
│   ├── vectorstore.py      # ChromaDB setup + evidence retrieval
│   ├── generator.py        # Claude Haiku + verbalized confidence
│   ├── selfcheck.py        # SelfCheckGPT (NLI-based)
│   ├── judge.py            # LLM-as-Judge
│   ├── metrics.py          # AUROC, ECE, Exact Match
│   └── visualize.py        # matplotlib figures
├── prompts/                # Prompt templates
├── data/                   # Datasets & ChromaDB (auto-generated)
└── results/                # All outputs (auto-generated)
```
