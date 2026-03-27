"""
src/metrics.py

Metric computation: Exact Match, Hallucination Rate, AUROC, ECE.
"""

import math
import numpy as np
from sklearn.metrics import roc_auc_score


def exact_match(prediction: str, ground_truth: str) -> int:
    """Case-insensitive substring match. Returns 1 if ground truth is in prediction."""
    if not prediction or not ground_truth:
        return 0
    return 1 if ground_truth.strip().lower() in prediction.strip().lower() else 0


def hallucination_rate(results: list[dict]) -> float:
    """Fraction of results where is_hallucinated == 1."""
    total = len(results)
    if total == 0:
        return 0.0
    hallucinated = sum(1 for r in results if r.get("is_hallucinated", 0) == 1)
    return hallucinated / total


def abstention_rate(results: list[dict]) -> float:
    """Fraction of answers containing 'cannot answer'."""
    total = len(results)
    if total == 0:
        return 0.0
    abstained = sum(
        1 for r in results
        if "cannot answer" in r.get("answer", "").lower()
    )
    return abstained / total


def compute_auroc(uncertainty_scores: list[float], is_hallucinated_labels: list[int]) -> float:
    """
    Compute AUROC where higher uncertainty should predict hallucination.
    Returns nan if only one class is present.
    """
    labels = list(is_hallucinated_labels)
    scores = list(uncertainty_scores)

    if len(set(labels)) < 2:
        return float("nan")

    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return float("nan")


def compute_ece(confidences: list[float], accuracies: list[int], n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    confidences: list of floats in [0, 1]
    accuracies: list of 0/1
    """
    confidences = np.array(confidences, dtype=float)
    accuracies = np.array(accuracies, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_all_metrics(results: list[dict]) -> dict:
    """
    Compute all metrics for a list of result dicts (same condition).
    Returns a summary dict.
    """
    n = len(results)
    if n == 0:
        return {}

    hal_rate = hallucination_rate(results)
    abs_rate = abstention_rate(results)
    em_rate = sum(r.get("exact_match", 0) for r in results) / n

    # Verbalized Confidence → uncertainty = 1 - conf/100
    vc_uncertainty = [1.0 - (r.get("verbalized_confidence", 50) / 100.0) for r in results]
    # SelfCheckGPT uncertainty (already computed as 1 - consistency)
    sc_uncertainty = [r.get("selfcheck_uncertainty", 0.5) for r in results]
    labels = [r.get("is_hallucinated", 0) for r in results]

    auroc_vc = compute_auroc(vc_uncertainty, labels)
    auroc_sc = compute_auroc(sc_uncertainty, labels)

    # ECE: use verbalized confidence, accuracy = 1 - is_hallucinated
    confidences = [r.get("verbalized_confidence", 50) / 100.0 for r in results]
    accuracies = [1 - r.get("is_hallucinated", 0) for r in results]
    ece = compute_ece(confidences, accuracies)

    mean_conf = sum(r.get("verbalized_confidence", 50) for r in results) / n
    overconfidence_gap = mean_conf / 100.0 - (1 - hal_rate)

    return {
        "n": n,
        "exact_match_rate": round(em_rate, 4),
        "hallucination_rate": round(hal_rate, 4),
        "abstention_rate": round(abs_rate, 4),
        "auroc_verbalized_confidence": round(auroc_vc, 4) if not math.isnan(auroc_vc) else None,
        "auroc_selfcheck": round(auroc_sc, 4) if not math.isnan(auroc_sc) else None,
        "ece": round(ece, 4),
        "mean_verbalized_confidence": round(mean_conf, 2),
        "overconfidence_gap": round(overconfidence_gap, 4),
    }
