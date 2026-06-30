"""
src/generator_multi.py  (cross-model extension, added 2026-06-29)

Drop-in generator that mirrors src/generator.generate_with_confidence but routes
through src/providers.chat, so the model is chosen by config.GENERATOR_PROVIDER /
config.GENERATOR_MODEL alone.

Differences from the original generator (deliberate, invariant-preserving):
  * confidence is NEVER defaulted to 50 — a missing Confidence annotation yields
    None and is propagated as missing (briefing Refinement 1 invariant).
  * full_api_response always carries raw_text so metrics_v2.reparse_confidence is
    the single source of truth at analysis time, for every provider.

Prompts are reused unchanged from ./prompts. The original src/generator.py is
left byte-identical for audit (Haiku is not regenerated).
"""

import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import providers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Same confidence regex as metrics_v2 so generation-time and analysis-time parsing agree.
_CONFIDENCE_REGEX = re.compile(r"confidence\s*:\s*(-?\d+)", re.IGNORECASE)
_ANSWER_SPLIT_REGEX = re.compile(r"answer\s*:", re.IGNORECASE)
_CONF_SPLIT_REGEX = re.compile(r"confidence\s*:", re.IGNORECASE)


def _load_prompt(prompt_type: str) -> str:
    filename = f"generation_{prompt_type}.txt"
    with open(os.path.join(config.PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(question: str, context_chunks: list, prompt_type: str) -> str:
    context = "\n\n".join(context_chunks)
    return _load_prompt(prompt_type).format(context=context, question=question)


def parse_response(text: str):
    """
    Format-tolerant parse of 'Answer: ... Confidence: N'.

    Returns (answer: str, confidence: int|None). Confidence is None when no
    Confidence annotation is present — NEVER defaulted to 50. Values are clamped
    to [0, 100]. Works across Claude/GPT/Gemini output styles (case-insensitive
    markers, optional whitespace).
    """
    if not text:
        return "", None

    # Answer body
    m_ans = _ANSWER_SPLIT_REGEX.search(text)
    if m_ans:
        after = text[m_ans.end():]
        m_conf_split = _CONF_SPLIT_REGEX.search(after)
        answer = after[: m_conf_split.start()].strip() if m_conf_split else after.strip()
    else:
        answer = text.strip()

    # Confidence (missing -> None)
    confidence = None
    m_conf = _CONFIDENCE_REGEX.search(text)
    if m_conf:
        try:
            confidence = max(0, min(100, int(m_conf.group(1))))
        except (TypeError, ValueError):
            confidence = None

    return answer, confidence


def generate_with_confidence(
    question: str,
    context_chunks: list,
    temperature: float = config.TEMPERATURE_DETERMINISTIC,
    prompt_type: str = "constrained",
    max_retries: int = 5,
):
    """
    Generate (answer, confidence|None, full_response_dict) using the configured
    generator provider/model, with rate limiting + exponential backoff.
    """
    provider = config.GENERATOR_PROVIDER
    model = config.GENERATOR_MODEL
    user_message = _build_user_message(question, context_chunks, prompt_type)

    backoff = 1
    for attempt in range(max_retries):
        try:
            time.sleep(config.API_SLEEP)
            text, full_response = providers.chat(
                provider=provider,
                model=model,
                user_message=user_message,
                temperature=temperature,
                max_tokens=512,
            )
            answer, confidence = parse_response(text)
            return answer, confidence, full_response
        except Exception as e:  # cross-SDK: backoff on any transient API/SDK error
            wait = min(backoff, config.API_MAX_BACKOFF)
            logger.warning(
                f"[{provider}] error (attempt {attempt+1}/{max_retries}), waiting {wait}s: {e}"
            )
            time.sleep(wait)
            backoff *= 2

    logger.error(f"[{provider}] all {max_retries} attempts failed for: {question[:60]}")
    return "ERROR", None, {"error": "max_retries_exceeded", "provider": provider}
