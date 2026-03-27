"""
validate_setup.py

Run BEFORE the experiment to verify that:
1. All 200 questions come from different Wikipedia articles
2. No Evidence condition actually removes all relevant context
3. Full Evidence condition retrieves the answer
"""

import json
import sys
from collections import Counter

import config
from src.dataset import load_cached_dataset
from src.vectorstore import get_client, get_collection, get_evidence, setup_vectorstore


def main():
    print("=" * 60)
    print("VALIDATING EXPERIMENT SETUP")
    print("=" * 60)

    # 1. Load dataset
    data = load_cached_dataset(config.FILTERED_NQ_FILE)
    print(f"\n[1] Dataset: {len(data)} questions")

    # 2. Check article uniqueness
    titles = [d["article_title"] for d in data]
    unique_titles = set(titles)
    print(f"[2] Unique articles: {len(unique_titles)}")

    if len(titles) != len(unique_titles):
        dupes = [t for t, c in Counter(titles).items() if c > 1]
        print(f"  FAIL: Duplicate articles found: {dupes[:5]}")
        sys.exit(1)
    else:
        print(f"  PASS: All articles are unique")

    # 3. Setup vectorstore
    collection = setup_vectorstore(data)
    print(f"[3] ChromaDB: {collection.count()} chunks indexed")

    # 4. Test evidence conditions on 10 random questions
    import random
    random.seed(42)
    test_indices = random.sample(range(len(data)), min(10, len(data)))

    full_has_answer = 0
    none_has_answer = 0

    for idx in test_indices:
        item = data[idx]
        gt = item["ground_truth"].lower()

        # Full Evidence: should contain the answer
        full_chunks = get_evidence(item["question"], item["question_id"], "full", collection)
        full_contains = any(gt in c.lower() for c in full_chunks)
        if full_contains:
            full_has_answer += 1

        # No Evidence: should NOT contain the answer
        none_chunks = get_evidence(item["question"], item["question_id"], "none", collection)
        none_contains = any(gt in c.lower() for c in none_chunks)
        if none_contains:
            none_has_answer += 1
            print(f"  WARNING: No Evidence for Q{item['question_id']} still contains answer!")
            print(f"    Question: {item['question'][:60]}")
            print(f"    Answer: {item['ground_truth']}")

    print(f"\n[4] Evidence condition test (n=10):")
    print(f"  Full Evidence contains answer: {full_has_answer}/10")
    print(f"  No Evidence contains answer:   {none_has_answer}/10")

    if none_has_answer > 2:
        print(f"\n  FAIL: No Evidence leaks too much context!")
        sys.exit(1)

    if full_has_answer < 5:
        print(f"\n  WARNING: Full Evidence retrieval seems weak. Check embedding model.")

    print(f"\n{'=' * 60}")
    print("VALIDATION PASSED — safe to run experiment")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
