"""
src/dataset.py

Loads SQuAD validation split with ONE question per Wikipedia article.
This ensures that each question's evidence is independent — removing
one article's chunks does not leak relevant context from another.

Design decision: We deduplicate by article title so that the 200 questions
come from 200 different Wikipedia articles. This is critical for the
"No Evidence" condition to work correctly.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_filter(num_queries: int = config.NUM_QUERIES) -> list[dict]:
    """
    Load SQuAD validation split. Take exactly ONE question per Wikipedia article
    to ensure document independence across questions.
    """
    # Use train split — validation split has only ~48 unique articles, not enough for 200.
    # Train split has 442 unique articles across 87k examples.
    logger.info("Loading SQuAD dataset (train split)...")
    dataset = load_dataset("squad", split="train")

    results = []
    seen_titles = set()  # Track which Wikipedia articles we've already used

    logger.info(f"Filtering for {num_queries} entries (one per article)...")
    for item in tqdm(dataset, desc="Processing SQuAD"):
        if len(results) >= num_queries:
            break

        title = item["title"].strip()
        question = item["question"].strip()
        answers = item["answers"]["text"]
        context = item["context"].strip()

        # Skip if we already have a question from this article
        if title in seen_titles:
            continue

        # Basic quality filters
        if not answers or not any(a.strip() for a in answers):
            continue

        answer = answers[0].strip()
        if not answer or len(answer) > 100:
            continue

        # Verify answer is in context (should always be true for SQuAD)
        if answer.lower() not in context.lower():
            continue

        # Skip very short contexts (< 200 chars) — not enough for meaningful retrieval
        if len(context) < 200:
            continue

        seen_titles.add(title)
        results.append({
            "question_id": len(results),
            "question": question,
            "ground_truth": answer,
            "all_answers": answers,
            "document": context,
            "article_title": title,  # Store for traceability
        })

    logger.info(f"Collected {len(results)} entries from {len(seen_titles)} unique articles.")

    # Validation: confirm all articles are unique
    titles_in_results = [r["article_title"] for r in results]
    assert len(titles_in_results) == len(set(titles_in_results)), \
        "FATAL: Duplicate articles found in dataset!"

    return results


def save_dataset(data: list[dict], path: str = config.FILTERED_NQ_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(data)} entries to {path}")


def load_cached_dataset(path: str = config.FILTERED_NQ_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Always rebuild for v2
    data = load_and_filter()
    save_dataset(data)

    print(f"\nDataset: {len(data)} questions from {len(set(d['article_title'] for d in data))} unique articles")
    print(f"\nSample entries:")
    for d in data[:5]:
        print(f"  Q: {d['question'][:70]}")
        print(f"  A: {d['ground_truth']}")
        print(f"  Article: {d['article_title']}")
        print(f"  Context length: {len(d['document'])} chars")
        print()
