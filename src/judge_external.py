"""
src/judge_external.py  (cross-model extension, added 2026-06-29)

Externalized LLM-as-Judge. Identical judging behaviour to src/judge.py EXCEPT the
calling model: a fixed external 3rd-family model (Qwen2.5-72B-Instruct via an
OpenAI-compatible endpoint, e.g. OpenRouter), used identically for ALL generators.
This removes the generator==judge self-enhancement bias.

  * Judge prompt UNCHANGED (prompts/llm_judge.txt).
  * Verdict parsing + verdict->label mapping reused verbatim from src/judge.py.
  * Non-thinking instruct model, temperature 0, pinned version.

The original src/judge.py is left byte-identical for audit.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import providers
# Reuse the exact parsing/mapping so verdicts are comparable to the old judge.
from src.judge import parse_verdict, verdict_to_hallucinated

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_judge_prompt() -> str:
    with open(os.path.join(config.PROMPTS_DIR, "llm_judge.txt"), "r", encoding="utf-8") as f:
        return f.read()


def judge_answer(question: str, context_chunks: list, answer: str, max_retries: int = 5):
    """
    Call the external judge. Returns (verdict, reasoning, is_hallucinated).
    Behaviour for failed/empty answers matches the original judge.
    """
    if not answer or answer == "ERROR":
        return "not_supported", "Answer generation failed.", 1

    context = "\n\n".join(context_chunks)
    user_message = _load_judge_prompt().format(context=context, question=question, answer=answer)

    backoff = 1
    for attempt in range(max_retries):
        try:
            time.sleep(config.API_SLEEP)
            text, _full = providers.chat(
                provider=config.JUDGE_PROVIDER,
                model=config.JUDGE_MODEL,
                user_message=user_message,
                temperature=config.JUDGE_TEMPERATURE,
                max_tokens=256,
                provider_pin=config.JUDGE_PROVIDER_PIN,
            )
            verdict = parse_verdict(text)
            is_hallucinated = verdict_to_hallucinated(verdict)
            reasoning = ""
            if "Reasoning:" in text:
                reasoning = text.split("Reasoning:")[1].strip()
            return verdict, reasoning, is_hallucinated
        except Exception as e:
            wait = min(backoff, config.API_MAX_BACKOFF)
            logger.warning(
                f"[judge:{config.JUDGE_PROVIDER}] error (attempt {attempt+1}/{max_retries}), "
                f"waiting {wait}s: {e}"
            )
            time.sleep(wait)
            backoff *= 2

    logger.error(f"[judge] all retries failed for: {question[:60]}")
    return "not_supported", "ERROR: max retries exceeded", 1
