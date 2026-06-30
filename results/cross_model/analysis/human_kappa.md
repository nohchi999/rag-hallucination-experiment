# Judge / pipeline validation — Cohen's kappa vs human labels (Task F)

Hand-labeled rows: **144** (matched to data: 144).

| Comparison | Cohen's kappa | Agreement |
|---|---|---|
| Human vs pipeline response-category (3-way) | **0.872** | 92.4% |
| Human vs external Judge (hallucination) | **0.576** | 90.3% |
| Human vs EM (hallucination) | **0.731** | 92.4% |

*Human labels: correctness scheme (abstention/correct/incorrect). Pipeline uses the extended, Haiku-safe abstention detector. Judge = qwen3-235b-a22b-instruct-2507.*
