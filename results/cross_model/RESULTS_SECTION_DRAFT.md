# Cross-Model Replication — Results Section (DRAFT)

> Hand-off draft for the thesis-writing session. All numbers are from the committed
> artefacts in `results/cross_model/analysis/`. Not thesis prose yet — adapt voice,
> citation style and cross-references to the main document. Figures/tables referenced
> by filename so they can be pulled in.

## X.1 Setup of the cross-model extension

To address Limitation 1 of the single-model study ("single model"), the full 2×3
factorial experiment was replicated across **three generator models** while holding
every other factor bit-identical. The generator is the **only** varying factor.

| Role | Model | Pinned version |
|---|---|---|
| Generator (anchor) | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` |
| Generator | OpenAI GPT-4.1-mini | `gpt-4.1-mini-2025-04-14` |
| Generator | Google Gemini 2.5 Flash | `gemini-2.5-flash` (thinking disabled) |
| External judge (all three) | Qwen3-235B-A22B-Instruct | `qwen3-235b-a22b-instruct-2507` (Alibaba Model Studio, temp 0) |

Held identical to the Haiku run: the 198 SQuAD questions/cell and quality filter, the
ChromaDB index (all-MiniLM-L6-v2, chunk 500/overlap 50, top-k=3, cosine), the evidence
operationalisation (full/partial/none), the constrained/unconstrained prompts and the
`Answer: … Confidence: N` format, SelfCheckGPT (5 samples @ T=0.7, NLI scorer
`cross-encoder/nli-deberta-v3-small`), and the validated (v2) analysis pipeline with all
six refinements. For the new models, the retrieved evidence was **reused verbatim** from
the Haiku run (same chunks per cell), eliminating retrieval drift.

**Judge externalisation.** In the single-model study the generator and judge were both
Haiku, risking self-enhancement bias (Zheng et al. 2023). The judge was moved to a fixed
model from a **third family** (Qwen), used identically for all three generators, making it
symmetrically independent of Claude, GPT and Gemini. Haiku's stored answers were
**re-judged** by this external judge (no regeneration); the original Haiku judge labels are
retained for audit. The judge prompt is unchanged.

**Judge / pipeline validation against human labels.** A stratified sample of 144 responses
(48/model, balanced across prompt type, evidence condition and abstention/attempted) was
hand-labelled. Agreement (Cohen's κ):

| Comparison | κ | Agreement |
|---|---|---|
| Human vs automated response-classification (3-way) | **0.872** | 92.4 % |
| Human vs EM hallucination label | 0.731 | 92.4 % |
| Human vs external judge label | 0.576 | 90.3 % |

The high human–pipeline agreement (κ = 0.87) substantiates the automated abstention/EM
classification. The lower human–judge κ (0.58) is expected and informative: the human
labelled answer *correctness* whereas the judge labels context *groundedness* — these
diverge precisely for answers that are factually correct but not supported by the
(reduced) evidence, which is the phenomenon under study.

**Refinement 2 (abstention markers), applied uniformly.** Human labelling revealed that
GPT/Gemini phrase abstentions differently from Haiku ("The provided text does not …" vs
Haiku's "context"-based phrasing), causing natural-language abstentions to be miscounted as
hallucinations. Consistent with the single-model methodology — where the v2 pipeline
already extended Haiku's abstention markers for the same reason — the detector was extended
with one anchored pattern and applied **uniformly to all three models** (same instrument).
The extension is **provably Haiku-safe**: zero Haiku rows change and Haiku's EM table is
bit-identical (F1 = 23.74 % → 9.09 %). It affects GPT in 0 rows and reclassifies 87 Gemini
rows (genuine abstentions). Validated against the human labels, it raises human–pipeline
κ from 0.818 to 0.872.

## X.2 Finding 1 (RQ1): constrained prompting mitigates hallucination under partial evidence

The headline single-model result — constrained prompting sharply reduces EM-hallucination
under partial evidence — **replicates in all three models** (EM-based; Wilson CIs in
`analysis/<model>/table1_hallucination.csv`):

| Model | unconstrained / partial | constrained / partial | absolute reduction |
|---|---|---|---|
| Claude Haiku 4.5 | 23.74 % | 9.09 % | −14.6 pp |
| GPT-4.1-mini | 28.28 % | 16.67 % | −11.6 pp |
| Gemini-2.5-flash | 17.68 % | 6.06 % | −11.6 pp |

The direction and significance are consistent across models; the **magnitude is
model-dependent**. Partial evidence remains the most hallucination-prone regime for every
model. A notable cross-model nuance only visible after the uniform abstention correction:
under unconstrained/partial, **GPT hallucinates most and Gemini least** — not because Gemini
is "safer" in its assertions but because it *abstains* far more often ("the provided text
does not …"). This is itself a model-level behavioural difference in how reduced evidence is
handled.

## X.3 Finding 2 (RQ2): population-dependence of UE-method evaluation

The methodological core finding — that the apparent quality/ranking of UE methods depends on
whether AUROC is computed over the **full population** or restricted to **attempted (non-
abstention) answers** — holds directionally across models, but must be read with the
attempted-answer *n* in hand (`table2_auroc.csv`). Several attempted-answer subsamples are
very small (e.g. Haiku constrained/none n = 3; Gemini constrained/none n = 4, constrained/
partial n = 16), so **point AUROC comparisons in these cells are not interpreted** — their
bootstrap CIs are correspondingly wide.

The **verbalized-confidence inversion** (AUROC_VC ≪ 0.5, i.e. confidence anti-correlated with
correctness) appears in all three models under reduced evidence, in model-specific cells:
Haiku unconstrained/partial AUROC_VC = 0.17; Gemini constrained/partial = 0.04 and
constrained/none = 0.07; GPT is closer to chance (unconstrained/partial = 0.50).
SelfCheckGPT is the more robust signal over the full population for every model. The key
qualitative claim — that switching the evaluation population changes which method looks best
— reproduces directionally; it is **not** asserted as a numeric per-cell comparison in
thin cells.

## X.4 Finding 3 (RQ3): overconfidence on attempted answers under reduced evidence

Restricting calibration to attempted answers (`table3_calibration.csv`, `ece_non_abstention`)
shows substantial over-confidence under partial/none evidence for all models — e.g.
constrained/partial ECE(non-abstention): Haiku 0.58, GPT 0.68, Gemini 0.73 — versus
well-calibrated full-evidence cells (ECE ≈ 0.05–0.09). The pattern is **consistent across
models**: when forced to answer with insufficient evidence, attempted answers are
systematically over-confident.

## X.5 Finding 4: prompt-conditional abstention-confidence asymmetry — model-dependent

The single-model novelty finding — constrained abstentions carry *high* confidence
("confident I cannot answer") while unconstrained abstentions carry *low* confidence
("no confidence in any answer") — does **not** replicate uniformly; it is the most
**model-dependent** result. Among abstentions with parsed confidence:

| Model | constrained: mean (% ≥90 / % ≤10) | unconstrained: mean (% ≥90 / % ≤10) |
|---|---|---|
| Claude Haiku 4.5 | 75.2 (76.8 % / 19.5 %) | 3.6 (0 % / 92.1 %) |
| GPT-4.1-mini | 98.0 (97.4 % / 1.7 %) | 54.5 (52.3 % / 40.6 %) |
| Gemini-2.5-flash | 6.2 (6.2 % / 93.8 %) | 71.6 (66.9 % / 24.9 %) |

Haiku reproduces the bimodal asymmetry; GPT reproduces the constrained-high pole but is
mixed when unconstrained; **Gemini inverts the pattern entirely** (low-confidence
constrained abstentions, high-confidence unconstrained abstentions). The asymmetry is thus a
**property of a specific model's prompt-conditioned abstention style, not a general law** —
directly supporting the literature's claim (Feng et al. 2025) that prompt-based abstention
behaviour is model-dependent.

## X.6 Cross-model synthesis

| Finding | Haiku | GPT-4.1-mini | Gemini-2.5-flash | Verdict |
|---|---|---|---|---|
| F1 — constrained mitigates hallucination (partial) | ✓ | ✓ | ✓ | **replicated** |
| F2 — population-dependent UE evaluation / VC inversion | ✓ | partial | ✓ | **replicated (directional; small-n cells excluded)** |
| F3 — overconfidence on attempted under reduced evidence | ✓ | ✓ | ✓ | **replicated** |
| F4 — prompt-conditional abstention-confidence asymmetry | ✓ | partial | inverted | **divergent / model-dependent** |

**Dual-label divergence (EM vs judge).** Across models the strict EM label and the
groundedness judge diverge most under reduced evidence and unconstrained prompting (e.g.
Gemini unconstrained/partial: EM-hallucination 17.68 % but judge-hallucination ≈ 2 %; per-cell
κ as low as 0.06). EM penalises non-substring-correct answers and ignores groundedness; the
judge accepts context-grounded answers regardless of GT wording. The gap is itself
model-dependent and motivates reporting both labels rather than either alone.

## X.7 Take-aways

The **robust** cross-model results are F1 (constrained prompting mitigates hallucination
under partial evidence) and F3 (over-confidence on attempted answers under reduced
evidence). The **abstention-side** behaviour (F4, and the EM–judge gap) is strongly
**model-dependent**: models differ not so much in whether they can be steered away from
hallucination, but in *how* they abstain and how confident they are when they do. This both
confirms the central single-model contribution and sharpens its scope.

---
*Artefacts: `results/cross_model/analysis/<model>/table1-4.csv`, `cross_model_summary.md`,
`human_kappa.md`; run manifests with exact versions/costs in `results/cross_model/`.
Judge = qwen3-235b-a22b-instruct-2507; total API cost ≈ $3.07.*
