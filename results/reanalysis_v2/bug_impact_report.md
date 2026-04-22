# Bug Impact Report — v2 Reanalysis

This report quantifies the effect of the six bug fixes defined in `BRIEFING_BUG_FIX.md`. The v1 pipeline (`src/metrics.py`, `analyze_results.py`) is kept unchanged for auditability.


## Summary

- Total datapoints: 1200
- Invalid datapoints filtered (Bug #5): 12
- Confidence values reparsed differently (Bug #1): 151 changed, 0 now None (vs v1 default 50)
- Abstentions reclassified from v1-false to v2-true (Bug #3): 4
- Abstentions reclassified from v1-true to v2-false (Bug #3/4): 0


## Bug #1 — Parser Default Contamination

- v1 had `verbalized_confidence = 50` on 151 rows
- Of those, v2 recovered a real Confidence value on 151 rows
- Of those, v2 determined no Confidence was parseable on 0 rows (now stored as None, not 50)

## Bug #2 — Abstention Confidence Semantics

Confidence distribution among v2 abstentions (where confidence is parseable):

| prompt_type | condition | n | pct_high (≥90) | pct_low (≤10) | pct_middle (40<c<60) |
|---|---|---|---|---|---|
| constrained | full | 9 | 55.6 | 11.1 | 0.0 |
| constrained | partial | 175 | 76.6 | 18.3 | 0.6 |
| constrained | none | 195 | 77.9 | 21.0 | 0.0 |
| unconstrained | full | 6 | 0.0 | 83.3 | 0.0 |
| unconstrained | partial | 141 | 0.0 | 90.1 | 0.0 |
| unconstrained | none | 171 | 0.0 | 94.2 | 0.0 |

## Bug #3 — Abstention Detector Gaps

- Reclassified v1-hallucination → v2-abstention: constrained=0, unconstrained=4

Sample of up to 10 reclassifications:

| question_id | prompt_type | condition | answer excerpt |
|---|---|---|---|
| 20 | unconstrained | partial | The answer cannot be determined from the provided context. |

## Bug #4 — EM / Abstention Disjoint Check

Every cell's three category percentages must sum to exactly 100.00%.

| prompt_type | condition | em_pct | hall_em_pct | abst_pct | sum |
|---|---|---|---|---|---|
| constrained | full | 86.36 | 9.09 | 4.55 | 100.0 |
| constrained | partial | 2.53 | 9.09 | 88.38 | 100.0 |
| constrained | none | 1.01 | 0.51 | 98.48 | 100.0 |
| unconstrained | full | 86.36 | 10.61 | 3.03 | 100.0 |
| unconstrained | partial | 5.05 | 23.74 | 71.21 | 100.0 |
| unconstrained | none | 2.53 | 11.11 | 86.36 | 100.0 |

## Bug #5 — Degenerate Datapoints

Filtered out 12 datapoints:

| question_id | prompt_type | condition | reason |
|---|---|---|---|
| 61 | constrained | full | ground_truth_too_short: 'H' |
| 61 | constrained | partial | ground_truth_too_short: 'H' |
| 61 | constrained | none | ground_truth_too_short: 'H' |
| 61 | unconstrained | full | ground_truth_too_short: 'H' |
| 61 | unconstrained | partial | ground_truth_too_short: 'H' |
| 61 | unconstrained | none | ground_truth_too_short: 'H' |
| 74 | constrained | full | question_too_short: 'k' |
| 74 | constrained | partial | question_too_short: 'k' |
| 74 | constrained | none | question_too_short: 'k' |
| 74 | unconstrained | full | question_too_short: 'k' |
| 74 | unconstrained | partial | question_too_short: 'k' |
| 74 | unconstrained | none | question_too_short: 'k' |

## Bug #6 — ECE Variants

| prompt_type | condition | ECE_all | ECE_non_abstention | ECE_parsed_only | N_all | N_non_abs | N_unparsed |
|---|---|---|---|---|---|---|---|
| constrained | full | 0.0542 | 0.0452 | 0.0542 | 198 | 189 | 0 |
| constrained | partial | 0.1389 | 0.5826 | 0.1389 | 198 | 23 | 0 |
| constrained | none | 0.0543 | 0.2833 | 0.0543 | 198 | 3 | 0 |
| unconstrained | full | 0.0759 | 0.0645 | 0.0759 | 198 | 192 | 0 |
| unconstrained | partial | 0.4302 | 0.3512 | 0.4302 | 198 | 57 | 0 |
| unconstrained | none | 0.3664 | 0.1611 | 0.3664 | 198 | 27 | 0 |

## Impact on Thesis Findings

| Finding | v1 claim | v2 status (to be filled after review) |
|---|---|---|
| Prompt reduces hallucination (Constr vs Unconstr / Partial) | Δ = 24.5% → 9% | _see table 1_ |
| SelfCheckGPT outperforms VC | AUROC_SC > AUROC_VC | _see table 2_ |
| AUROC VC < 0.5 at Partial | confidence inversion | _see table 2_ |
| Under-confidence contradicts Xiong | mean_conf < accuracy | _see table 3_ |
| Partial Evidence is most dangerous | highest hall_rate under Partial | _see table 1_ |
