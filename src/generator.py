"""
src/generator.py

Generates answers with verbalized confidence using Claude Haiku 3.5.
Includes rate limiting and exponential backoff.
"""

import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _load_prompt() -> str:
    prompt_path = os.path.join(config.PROMPTS_DIR, "verbalized_confidence.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _build_user_message(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt_template = _load_prompt()
    return prompt_template.format(context=context, question=question)


def parse_response(text: str) -> tuple[str, int]:
    """Parse 'Answer: ... Confidence: N' format. Returns (answer, confidence)."""
    answer = ""
    confidence = 50  # default

    if "Answer:" in text:
        answer_part = text.split("Answer:")[1]
        if "Confidence:" in answer_part:
            answer = answer_part.split("Confidence:")[0].strip()
            conf_str = answer_part.split("Confidence:")[1].strip()
            match = re.search(r'\d+', conf_str)
            if match:
                confidence = min(100, max(0, int(match.group())))
        else:
            answer = answer_part.strip()
    else:
        answer = text.strip()

    return answer, confidence


def generate_with_confidence(
    question: str,
    context_chunks: list[str],
    temperature: float = config.TEMPERATURE_DETERMINISTIC,
    max_retries: int = 5,
) -> tuple[str, int, dict]:
    """
    Call Claude Haiku 3.5 and return (answer, confidence, full_api_response_dict).
    Includes rate limiting and exponential backoff.
    """
    client = get_client()
    user_message = _build_user_message(question, context_chunks)

    backoff = 1
    for attempt in range(max_retries):
        try:
            time.sleep(config.API_SLEEP)
            response = client.messages.create(
                model=config.MODEL_NAME,
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            answer, confidence = parse_response(text)

            full_response = {
                "id": response.id,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "raw_text": text,
            }
            return answer, confidence, full_response

        except anthropic.RateLimitError as e:
            wait = min(backoff, config.API_MAX_BACKOFF)
            logger.warning(f"Rate limit hit (attempt {attempt+1}), waiting {wait}s: {e}")
            time.sleep(wait)
            backoff *= 2

        except anthropic.APIError as e:
            wait = min(backoff, config.API_MAX_BACKOFF)
            logger.warning(f"API error (attempt {attempt+1}), waiting {wait}s: {e}")
            time.sleep(wait)
            backoff *= 2

    logger.error(f"All {max_retries} attempts failed for question: {question[:60]}")
    return "ERROR", 50, {"error": "max_retries_exceeded"}
