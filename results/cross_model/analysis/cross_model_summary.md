# Cross-Model Synthesis (Task G) — label-independent

## F1 — Prompt mitigation under partial evidence (EM-based hallucination)

| Model | unconstrained/partial | constrained/partial | F1 status |
|---|---|---|---|
| claude-haiku-4.5 | 23.74% | 9.09% | **replicated** |
| gpt-4.1-mini | 28.28% | 16.67% | **replicated** |
| gemini-2.5-flash | 17.68% | 6.06% | **replicated** |

*EM-based; the validated pipeline (disjoint classify_response) and the SAME external judge (qwen3-235b-a22b-instruct-2507) were used for all models. AUROC/ECE/judge-rate tables per model in ./<model>/.*


## Refinement-2 — uniform abstention-marker extension (src/abstention_ext.py)

Applied to ALL models (same instrument). One anchored opener catches GPT/Gemini abstentions phrased 'The provided text does not ...' that the original 'context' markers missed. **Haiku-safe: 0 Haiku rows change** (F1 23.74->9.09 bit-identical). Effect: Gemini only (87 rows hallucinated/em -> abstention; GPT 0). This is the same kind of fix already applied to Haiku in the v2 pipeline. Validated by 144 human labels: human-vs-pipeline kappa 0.818 -> 0.872 (see human_kappa.md).
