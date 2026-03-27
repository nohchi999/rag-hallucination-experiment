"""
run_experiment.py

Main orchestration script. Runs the full RAG hallucination experiment:
  200 questions × 2 prompt types × 3 conditions × 7 API calls = ~8400 API calls

Usage:
    python run_experiment.py

Set ANTHROPIC_API_KEY before running:
    export ANTHROPIC_API_KEY=sk-ant-...

Resume: if results/checkpoint.json exists, resumes from last checkpoint.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

from tqdm import tqdm

import config
from src.dataset import load_and_filter, load_cached_dataset, save_dataset
from src.generator import generate_with_confidence, is_abstention
from src.judge import judge_answer
from src.metrics import exact_match, compute_em_hallucinated
from src.selfcheck import compute_selfcheck_score, selfcheck_sample
from src.vectorstore import get_client, get_collection, setup_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.RESULTS_PATH, "experiment.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_checkpoint() -> list[dict]:
    if os.path.exists(config.CHECKPOINT_FILE):
        logger.info(f"Loading checkpoint from {config.CHECKPOINT_FILE}")
        with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict]):
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    with open(config.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_raw_results(results: list[dict]):
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    with open(config.RAW_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(results)} results to {config.RAW_RESULTS_FILE}")


def build_done_set(results: list[dict]) -> set[tuple]:
    """Returns set of (question_id, prompt_type, condition) triples already completed."""
    return {(r["question_id"], r["prompt_type"], r["condition"]) for r in results}


def process_one(item: dict, prompt_type: str, condition: str, collection) -> dict:
    """Process a single question × prompt_type × condition. Returns result dict."""
    qid = item["question_id"]
    question = item["question"]
    ground_truth = item["ground_truth"]

    from src.vectorstore import get_evidence
    chunks = get_evidence(question, qid, condition, collection)

    # Step 1: Main answer + verbalized confidence
    answer, confidence, full_response = generate_with_confidence(
        question=question,
        context_chunks=chunks,
        temperature=config.TEMPERATURE_DETERMINISTIC,
        prompt_type=prompt_type,
    )

    # Step 2: SelfCheckGPT samples
    try:
        samples = selfcheck_sample(question, chunks, n=config.SELFCHECK_SAMPLES, prompt_type=prompt_type)
        consistency, uncertainty = compute_selfcheck_score(answer, samples)
    except Exception as e:
        logger.warning(f"SelfCheck failed for q={qid} pt={prompt_type} cond={condition}: {e}")
        samples = []
        consistency, uncertainty = 0.0, 1.0

    # Step 3: LLM-as-Judge
    verdict, reasoning, is_hallucinated_judge = judge_answer(question, chunks, answer)

    # Step 4: EM-based labels
    em = exact_match(answer, ground_truth)
    abstention = is_abstention(answer)
    is_hallucinated_em = compute_em_hallucinated(answer, ground_truth)

    return {
        "question_id": qid,
        "question": question,
        "ground_truth": ground_truth,
        "prompt_type": prompt_type,
        "condition": condition,
        "retrieved_chunks": chunks,
        "answer": answer,
        "verbalized_confidence": confidence,
        "selfcheck_samples": samples,
        "selfcheck_consistency": round(consistency, 4),
        "selfcheck_uncertainty": round(uncertainty, 4),
        "judge_verdict": verdict,
        "judge_reasoning": reasoning,
        "is_hallucinated_judge": is_hallucinated_judge,
        "is_hallucinated_em": is_hallucinated_em,
        "is_abstention": abstention,
        "exact_match": em,
        "full_api_response": full_response,
        "timestamp": datetime.utcnow().isoformat(),
    }


def main():
    if not config.ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is not set. Please export it before running.")
        sys.exit(1)

    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RAG Hallucination Experiment — 2x3 Factorial Design")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Queries: {config.NUM_QUERIES}")
    logger.info(f"Prompt types: {config.PROMPT_TYPES}")
    logger.info(f"Conditions: {config.CONDITIONS}")
    logger.info(f"Total API calls: ~{config.NUM_QUERIES * len(config.PROMPT_TYPES) * len(config.CONDITIONS) * 7}")
    logger.info(f"Start: {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    # Load or build dataset
    squad_path = config.FILTERED_SQUAD_FILE
    nq_path = "./data/filtered_nq.json"
    dataset_path = squad_path if os.path.exists(squad_path) else (nq_path if os.path.exists(nq_path) else squad_path)

    if os.path.exists(dataset_path):
        logger.info("Loading cached dataset...")
        dataset = load_cached_dataset(dataset_path)
    else:
        logger.info("Building dataset from SQuAD...")
        dataset = load_and_filter(config.NUM_QUERIES)
        save_dataset(dataset, squad_path)
    logger.info(f"Dataset: {len(dataset)} questions loaded.")

    # Setup vector store
    logger.info("Setting up ChromaDB vector store...")
    collection = setup_vectorstore(dataset)

    # Load checkpoint
    results = load_checkpoint()
    done_set = build_done_set(results)
    logger.info(f"Checkpoint: {len(results)} results already complete, resuming...")

    # Total tasks: questions × prompt_types × conditions
    total = len(dataset) * len(config.PROMPT_TYPES) * len(config.CONDITIONS)
    pbar = tqdm(total=total, desc="Experiment progress", unit="task")
    pbar.update(len(results))

    for item in dataset:
        qid = item["question_id"]
        for prompt_type in config.PROMPT_TYPES:
            for condition in config.CONDITIONS:
                if (qid, prompt_type, condition) in done_set:
                    continue

                try:
                    result = process_one(item, prompt_type, condition, collection)
                    results.append(result)
                    done_set.add((qid, prompt_type, condition))
                    pbar.update(1)
                    pbar.set_postfix(q=qid, pt=prompt_type[:4], cond=condition, em=result["exact_match"])
                except Exception as e:
                    logger.error(f"Failed q={qid} pt={prompt_type} cond={condition}: {e}", exc_info=True)
                    results.append({
                        "question_id": qid,
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "prompt_type": prompt_type,
                        "condition": condition,
                        "error": str(e),
                        "is_hallucinated_judge": 1,
                        "is_hallucinated_em": 1,
                        "is_abstention": False,
                        "exact_match": 0,
                        "verbalized_confidence": 50,
                        "selfcheck_uncertainty": 1.0,
                        "selfcheck_consistency": 0.0,
                        "judge_verdict": "not_supported",
                        "judge_reasoning": "ERROR",
                        "retrieved_chunks": [],
                        "selfcheck_samples": [],
                        "answer": "ERROR",
                        "full_api_response": {"error": str(e)},
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    done_set.add((qid, prompt_type, condition))
                    pbar.update(1)

        # Checkpoint after every full question (all 6 sub-tasks)
        save_checkpoint(results)

    pbar.close()

    save_raw_results(results)
    if os.path.exists(config.CHECKPOINT_FILE):
        os.remove(config.CHECKPOINT_FILE)
        logger.info("Checkpoint file removed (experiment complete).")

    logger.info("=" * 60)
    logger.info(f"Experiment complete! {len(results)} results saved.")
    logger.info(f"End: {datetime.utcnow().isoformat()}")
    logger.info("Run: python analyze_results.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
