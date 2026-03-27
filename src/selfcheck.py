"""
src/selfcheck.py

SelfCheckGPT: generates stochastic samples and measures consistency
via NLI (cross-encoder/nli-deberta-v3-small).

Consistency score: fraction of samples that entail the main answer.
Uncertainty score: 1 - consistency (higher = more uncertain = likely hallucinated).
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.generator import generate_with_confidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_nli_pipeline = None


def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        logger.info(f"Loading NLI model: {config.NLI_MODEL} ...")
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "text-classification",
            model=config.NLI_MODEL,
            truncation=True,
            max_length=512,
            framework="pt",  # force PyTorch, avoid TF/Keras 3 conflict
        )
        logger.info("NLI model loaded.")
    return _nli_pipeline


def selfcheck_sample(
    question: str,
    context_chunks: list[str],
    n: int = config.SELFCHECK_SAMPLES,
    prompt_type: str = "constrained",
) -> list[str]:
    """
    Generate n stochastic answer samples using temperature=0.7.
    Returns list of answer strings.
    """
    samples = []
    for i in range(n):
        answer, _, _ = generate_with_confidence(
            question=question,
            context_chunks=context_chunks,
            temperature=config.TEMPERATURE_STOCHASTIC,
            prompt_type=prompt_type,
        )
        samples.append(answer)
    return samples


def _nli_entailment_score(premise: str, hypothesis: str) -> float:
    """Returns entailment probability from NLI model."""
    nli = _get_nli_pipeline()
    try:
        result = nli(f"{premise} [SEP] {hypothesis}")
        # result is a list of dicts [{label, score}, ...]
        if isinstance(result, list) and isinstance(result[0], dict):
            for item in result:
                if item["label"].upper() in ("ENTAILMENT", "entailment"):
                    return float(item["score"])
        return 0.0
    except Exception as e:
        logger.warning(f"NLI scoring failed: {e}")
        return 0.0


def compute_selfcheck_score(
    main_answer: str,
    samples: list[str],
) -> tuple[float, float]:
    """
    Compute NLI-based consistency and uncertainty scores.

    Returns:
        (consistency_score, uncertainty_score)
        consistency = fraction of samples that entail the main answer
        uncertainty = 1 - consistency
    """
    if not main_answer or main_answer == "ERROR":
        return 0.0, 1.0

    if not samples:
        return 0.0, 1.0

    entailment_scores = []
    for sample in samples:
        score = _nli_entailment_score(main_answer, sample)
        entailment_scores.append(score)

    consistency = sum(entailment_scores) / len(entailment_scores)
    uncertainty = 1.0 - consistency
    return consistency, uncertainty
