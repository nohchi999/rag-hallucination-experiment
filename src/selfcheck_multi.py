"""
src/selfcheck_multi.py  (cross-model extension, added 2026-06-29)

SelfCheckGPT sampling for the configured generator. The 5 stochastic samples come
from the SAME generator model as the main generation (briefing invariant), at
temperature 0.7. The NLI scorer is UNCHANGED and generator-independent — reused
verbatim from src.selfcheck (cross-encoder/nli-deberta-v3-small).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.generator_multi import generate_with_confidence
# Reuse the fixed NLI scorer untouched.
from src.selfcheck import compute_selfcheck_score  # noqa: F401  (re-exported for callers)


def selfcheck_sample(question, context_chunks, n=config.SELFCHECK_SAMPLES, prompt_type="constrained"):
    """Generate n stochastic samples from the configured generator at temp 0.7."""
    samples = []
    for _ in range(n):
        answer, _, _ = generate_with_confidence(
            question=question,
            context_chunks=context_chunks,
            temperature=config.TEMPERATURE_STOCHASTIC,
            prompt_type=prompt_type,
        )
        samples.append(answer)
    return samples
