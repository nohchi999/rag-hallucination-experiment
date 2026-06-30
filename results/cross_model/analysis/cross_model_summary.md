# Cross-Model Synthesis (Task G) — label-independent

## F1 — Prompt mitigation under partial evidence (EM-based hallucination)

| Model | unconstrained/partial | constrained/partial | F1 status |
|---|---|---|---|
| claude-haiku-4.5 | 23.74% | 9.09% | **replicated** |
| gpt-4.1-mini | 28.28% | 16.67% | **replicated** |
| gemini-2.5-flash | 39.9% | 6.06% | **replicated** |

*EM-based; the validated pipeline (disjoint classify_response) and the SAME external judge (qwen3-235b-a22b-instruct-2507) were used for all models. AUROC/ECE/judge-rate tables per model in ./<model>/. Human-vs-judge kappa pending labels.*
