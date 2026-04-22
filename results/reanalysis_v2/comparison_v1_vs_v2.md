# v1 vs v2 Comparison — Headline Metrics

This document pairs every headline number from the original analysis pipeline (`results/table{1,2,3,4}_*.csv`) against the reanalysis pipeline (`results/reanalysis_v2/table*_v2.csv`). `delta = v2 − v1`. Rows marked _new in v2_ have no v1 counterpart and are reported for completeness.

Note: v1 uses n=200 per cell; v2 uses n=198 after filtering the two degenerate SQuAD items (Q61 gt='H', Q74 question='k').


## EM rate (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 87.50 | 86.36 | -1.14 |  |

## Hallucination EM (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 9.00 | 9.09 | +0.09 |  |

## Hallucination Judge (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 3.00 | 3.03 | +0.03 |  |

## Abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 5.00 | 4.55 | -0.45 |  |

## Category sum (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 101.50 | 100.00 | -1.50 | v1 >100% evidences Bug #4 |

## SQuAD F1 (mean)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 0.4790 | — | new in v2 |

## EM rate (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 5.00 | 2.53 | -2.47 |  |

## Hallucination EM (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 9.00 | 9.09 | +0.09 |  |

## Hallucination Judge (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 2.50 | 2.53 | +0.03 |  |

## Abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 88.50 | 88.38 | -0.12 |  |

## Category sum (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 102.50 | 100.00 | -2.50 | v1 >100% evidences Bug #4 |

## SQuAD F1 (mean)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 0.0254 | — | new in v2 |

## EM rate (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 4.00 | 1.01 | -2.99 |  |

## Hallucination EM (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 0.50 | 0.51 | +0.01 |  |

## Hallucination Judge (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 0.50 | 0.51 | +0.01 |  |

## Abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 98.50 | 98.48 | -0.02 |  |

## Category sum (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 103.00 | 100.00 | -3.00 | v1 >100% evidences Bug #4 |

## SQuAD F1 (mean)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 0.0116 | — | new in v2 |

## EM rate (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 86.00 | 86.36 | +0.36 |  |

## Hallucination EM (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 11.00 | 10.61 | -0.39 |  |

## Hallucination Judge (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 3.00 | 3.03 | +0.03 |  |

## Abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 3.00 | 3.03 | +0.03 |  |

## Category sum (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 100.00 | 100.00 | +0.00 |  |

## SQuAD F1 (mean)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 0.5244 | — | new in v2 |

## EM rate (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 7.00 | 5.05 | -1.95 |  |

## Hallucination EM (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 24.50 | 23.74 | -0.76 |  |

## Hallucination Judge (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 10.00 | 10.10 | +0.10 |  |

## Abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 70.50 | 71.21 | +0.71 |  |

## Category sum (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 102.00 | 100.00 | -2.00 | v1 >100% evidences Bug #4 |

## SQuAD F1 (mean)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 0.0626 | — | new in v2 |

## EM rate (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 5.00 | 2.53 | -2.47 |  |

## Hallucination EM (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 11.50 | 11.11 | -0.39 |  |

## Hallucination Judge (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 5.50 | 5.56 | +0.06 |  |

## Abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 86.00 | 86.36 | +0.36 |  |

## Category sum (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 102.50 | 100.00 | -2.50 | v1 >100% evidences Bug #4 |

## SQuAD F1 (mean)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 0.0387 | — | new in v2 |

## AUROC VC (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | full | 0.6378 | 0.6174 | -0.0204 |  |

## AUROC VC (Judge)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | full | 0.7414 | 0.7502 | +0.0088 |  |

## AUROC SC (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | full | 0.5382 | 0.6062 | +0.0680 |  |

## AUROC SC (Judge)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | full | 0.5339 | 0.6132 | +0.0793 |  |

## AUROC Judge (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | full | 0.5087 | 0.5118 | +0.0031 |  |

## AUROC VC (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | partial | 0.1795 | 0.4567 | +0.2772 |  |

## AUROC VC (Judge)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | partial | 0.1101 | 0.4456 | +0.3355 |  |

## AUROC SC (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | partial | 0.7121 | 0.5857 | -0.1264 |  |

## AUROC SC (Judge)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | partial | 0.6643 | 0.5900 | -0.0743 |  |

## AUROC Judge (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | partial | 0.6365 | 0.6104 | -0.0261 |  |

## AUROC VC (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | none | 0.4213 | 0.6080 | +0.1867 |  |

## AUROC VC (Judge)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | none | 0.0127 | 0.3999 | +0.3872 |  |

## AUROC SC (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | none | 0.7113 | 0.6452 | -0.0662 |  |

## AUROC SC (Judge)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | none | 0.8057 | 0.6919 | -0.1138 |  |

## AUROC Judge (EM)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| avg_over_prompts | none | 0.5917 | 0.7983 | +0.2066 |  |

## AUROC VC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 0.6153 | — | new in v2 |

## AUROC SC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 0.6890 | — | new in v2 |

## AUROC VC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 0.6306 | — | new in v2 |

## AUROC SC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 0.7032 | — | new in v2 |

## AUROC VC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 0.7407 | — | new in v2 |

## AUROC SC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 0.4640 | — | new in v2 |

## AUROC VC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 0.8833 | — | new in v2 |

## AUROC SC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 0.7111 | — | new in v2 |

## AUROC VC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 0.7868 | — | new in v2 |

## AUROC SC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 0.5787 | — | new in v2 |

## AUROC VC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 1.0000 | — | new in v2 |

## AUROC SC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 0.0000 | — | new in v2 |

## AUROC VC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 0.6195 | — | new in v2 |

## AUROC SC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 0.5234 | — | new in v2 |

## AUROC VC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 0.6412 | — | new in v2 |

## AUROC SC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 0.5284 | — | new in v2 |

## AUROC VC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 0.1728 | — | new in v2 |

## AUROC SC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 0.7074 | — | new in v2 |

## AUROC VC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 0.7053 | — | new in v2 |

## AUROC SC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 0.7096 | — | new in v2 |

## AUROC VC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 0.4292 | — | new in v2 |

## AUROC SC (EM, per cell)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 0.7116 | — | new in v2 |

## AUROC VC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 0.8545 | — | new in v2 |

## AUROC SC (EM, non-abst only)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 0.4364 | — | new in v2 |

## Mean Confidence (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 93.42 | 93.35 | -0.07 |  |

## Accuracy (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 91.00 | 90.91 | -0.09 |  |

## ECE

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 0.0542 | 0.0542 | +0.0000 |  |

## ECE non-abstention

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 0.0452 | — | new in v2 (Bug #6) |

## Overconf. Gap

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 0.0242 | 0.0244 | +0.0002 |  |

## Mean Conf non-abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | — | 94.99 | — | new in v2 (Bug #2) |

## Mean Confidence (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 69.67 | 75.81 | +6.14 |  |

## Accuracy (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 91.00 | 90.91 | -0.09 |  |

## ECE

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 0.2582 | 0.1389 | -0.1193 |  |

## ECE non-abstention

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 0.5826 | — | new in v2 (Bug #6) |

## Overconf. Gap

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | -0.2133 | -0.1510 | +0.0623 |  |

## Mean Conf non-abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | — | 80.00 | — | new in v2 (Bug #2) |

## Mean Confidence (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 69.25 | 76.14 | +6.89 |  |

## Accuracy (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 99.50 | 99.49 | -0.01 |  |

## ECE

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 0.1800 | 0.0543 | -0.1257 |  |

## ECE non-abstention

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 0.2833 | — | new in v2 (Bug #6) |

## Overconf. Gap

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | -0.3025 | -0.2336 | +0.0689 |  |

## Mean Conf non-abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | — | 88.33 | — | new in v2 (Bug #2) |

## Mean Confidence (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 91.89 | 92.29 | +0.40 |  |

## Accuracy (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 89.00 | 89.39 | +0.39 |  |

## ECE

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 0.0749 | 0.0759 | +0.0010 |  |

## ECE non-abstention

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 0.0645 | — | new in v2 (Bug #6) |

## Overconf. Gap

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 0.0289 | 0.0289 | +0.0000 |  |

## Mean Conf non-abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | — | 94.99 | — | new in v2 (Bug #2) |

## Mean Confidence (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 17.86 | 18.02 | +0.16 |  |

## Accuracy (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 75.50 | 76.26 | +0.76 |  |

## ECE

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 0.4206 | 0.4302 | +0.0096 |  |

## ECE non-abstention

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 0.3512 | — | new in v2 (Bug #6) |

## Overconf. Gap

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | -0.5764 | -0.5825 | -0.0061 |  |

## Mean Conf non-abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | — | 52.32 | — | new in v2 (Bug #2) |

## Mean Confidence (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 6.15 | 6.19 | +0.04 |  |

## Accuracy (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 88.50 | 88.89 | +0.39 |  |

## ECE

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 0.3625 | 0.3664 | +0.0039 |  |

## ECE non-abstention

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 0.1611 | — | new in v2 (Bug #6) |

## Overconf. Gap

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | -0.8235 | -0.8270 | -0.0035 |  |

## Mean Conf non-abstention (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | — | 25.74 | — | new in v2 (Bug #2) |

## Cohen's Kappa

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 0.0401 | 0.0397 | -0.0004 |  |

## Agreement (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 89.00 | 88.89 | -0.11 |  |

## n disagreements

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | full | 22 | 22 | +0 |  |

## Cohen's Kappa

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 0.2308 | 0.2305 | -0.0003 |  |

## Agreement (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 91.50 | 91.41 | -0.09 |  |

## n disagreements

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | partial | 17 | 17 | +0 |  |

## Cohen's Kappa

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 1.0000 | 1.0000 | +0.0000 |  |

## Agreement (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 100.00 | 100.00 | +0.00 |  |

## n disagreements

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| constrained | none | 0 | 0 | +0 |  |

## Cohen's Kappa

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 0.0255 | 0.0283 | +0.0028 |  |

## Agreement (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 87.00 | 87.37 | +0.37 |  |

## n disagreements

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | full | 26 | 25 | -1 |  |

## Cohen's Kappa

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 0.3412 | 0.3566 | +0.0154 |  |

## Agreement (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 80.50 | 81.31 | +0.81 |  |

## n disagreements

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | partial | 39 | 37 | -2 |  |

## Cohen's Kappa

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 0.2374 | 0.2473 | +0.0099 |  |

## Agreement (%)

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 88.00 | 88.38 | +0.38 |  |

## n disagreements

| prompt_type | condition | v1 | v2 | Δ | note |
|---|---|---|---|---|---|
| unconstrained | none | 24 | 23 | -1 |  |
