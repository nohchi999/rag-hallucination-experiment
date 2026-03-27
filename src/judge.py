"""
src/judge.py

LLM-as-Judge: evaluates whether the model's answer is supported
by the provided context using Claude Haiku 3.5.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

import config
from src.generator import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_judge_prompt() -> str:
    prompt_path = os.path.join(config.PROMPTS_DIR, "llm_judge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_verdict(text: str) -> str:
    """Parse verdict from judge response. Returns one of: supported, partially_supported, not_supported."""
    text_lower = text.lower()
    if "not_supported" in text_lower:
        return "not_supported"
    elif "partially_supported" in text_lower:
        return "partially_supported"
    elif "supported" in text_lower:
        return "supported"
    return "not_supported"  # conservative fallback on parse error


def verdict_to_hallucinated(verdict: str) -> int:
    """Convert verdict to binary hallucination label. 1 = hallucinated, 0 = correct."""
    if verdict in ("not_supported", "partially_supported"):
        return 1
    return 0


def judge_answer(
    question: str,
    context_chunks: list[str],
    answer: str,
    max_retries: int = 5,
) -> tuple[str, str, int]:
    """
    Call Claude Haiku 3.5 as judge.

    Returns:
        (verdict, reasoning, is_hallucinated)
        verdict: supported | partially_supported | not_supported
        reasoning: 1-2 sentence explanation
        is_hallucinated: 0 or 1
    """
    if not answer or answer == "ERROR":
        return "not_supported", "Answer generation failed.", 1

    client = get_client()
    context = "\n\n".join(context_chunks)
    prompt_template = _load_judge_prompt()
    user_message = prompt_template.format(context=context, question=question, answer=answer)

    backoff = 1
    for attempt in range(max_retries):
        try:
            time.sleep(config.API_SLEEP)
            response = client.messages.create(
                model=config.MODEL_NAME,
                max_tokens=256,
                temperature=config.TEMPERATURE_DETERMINISTIC,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            verdict = parse_verdict(text)
            is_hallucinated = verdict_to_hallucinated(verdict)

            reasoning = ""
            if "Reasoning:" in text:
                reasoning = text.split("Reasoning:")[1].strip()

            return verdict, reasoning, is_hallucinated

        except anthropic.RateLimitError as e:
            wait = min(backoff, config.API_MAX_BACKOFF)
            logger.warning(f"Rate limit (attempt {attempt+1}), waiting {wait}s: {e}")
            time.sleep(wait)
            backoff *= 2

        except anthropic.APIError as e:
            wait = min(backoff, config.API_MAX_BACKOFF)
            logger.warning(f"API error (attempt {attempt+1}), waiting {wait}s: {e}")
            time.sleep(wait)
            backoff *= 2

    logger.error(f"Judge: all retries failed for question: {question[:60]}")
    return "not_supported", "ERROR: max retries exceeded", 1
