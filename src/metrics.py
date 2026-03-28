"""
src/metrics.py

Metric computation: Exact Match, Hallucination Rate, AUROC, ECE,
Cohen's Kappa, Wilson CI — for 2x3 factorial design with dual labels.
"""

import math
import numpy as np
from sklearn.metrics import roc_auc_score

_ABSTENTION_MARKERS = [
    # Constrained-prompt formulations
    "cannot answer",
    "i don't have enough",
    "not enough information",
    "no relevant information",
    "the context does not",
    "based on the provided context, i cannot",
    "i'm unable to answer",
    "unable to answer",
    # Unconstrained-prompt formulations (semantic abstentions without explicit marker)
    "does not contain information",
    "does not contain sufficient information",
    "does not provide information",
    "is not available in the provided context",
    "is not mentioned in the context",
    "no information about",
    "the provided context does not",
    "context provided does not",
    "not found in the provided",
    "not present in the context",
]


def detect_abstention(answer: str) -> bool:
    """Detect whether the model abstained from answering."""
    answer_lower = answer.strip().lower()
    return any(marker in answer_lower for marker in _ABSTENTION_MARKERS)


def exact_match(prediction: str, ground_truth: str) -> int:
    """Case-insensitive substring match. Returns 1 if ground truth is in prediction."""
    if not prediction or not ground_truth:
        return 0
    return 1 if ground_truth.strip().lower() in prediction.strip().lower() else 0


def compute_em_hallucinated(answer: str, ground_truth: str) -> int:
    """
    EM-based hallucination label.
    1 = hallucinated (wrong answer, not abstention)
    0 = correct OR abstention
    """
    if detect_abstention(answer):
        return 0
    return 0 if exact_match(answer, ground_truth) else 1


def hallucination_rate(results: list[dict], label_field: str = "is_hallucinated_em") -> float:
    """Fraction of results where the given label field == 1."""
    total = len(results)
    if total == 0:
        return 0.0
    return sum(1 for r in results if r.get(label_field, 0) == 1) / total


def abstention_rate(results: list[dict]) -> float:
    """Fraction of answers that are abstentions."""
    total = len(results)
    if total == 0:
        return 0.0
    return sum(1 for r in results if r.get("is_abstention", False)) / total


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


def cohens_kappa(labels1: list[int], labels2: list[int]) -> float:
    """Compute Cohen's Kappa between two binary label lists."""
    n = len(labels1)
    if n == 0:
        return float("nan")

    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    # Observed agreement
    p_o = np.mean(labels1 == labels2)

    # Expected agreement
    p1_pos = np.mean(labels1 == 1)
    p2_pos = np.mean(labels2 == 1)
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Returns (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 0.0)

    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_all_metrics(results: list[dict]) -> dict:
    """
    Compute all metrics for a list of result dicts (same condition + prompt_type cell).
    Returns a summary dict with dual-label metrics.
    """
    n = len(results)
    if n == 0:
        return {}

    em_rate = sum(r.get("exact_match", 0) for r in results) / n
    abs_rate = abstention_rate(results)

    # Dual hallucination rates
    hal_rate_em = hallucination_rate(results, label_field="is_hallucinated_em")
    hal_rate_judge = hallucination_rate(results, label_field="is_hallucinated_judge")

    # Wilson CIs
    hal_em_n = sum(1 for r in results if r.get("is_hallucinated_em", 0) == 1)
    hal_judge_n = sum(1 for r in results if r.get("is_hallucinated_judge", 0) == 1)
    ci_em = wilson_ci(hal_em_n, n)
    ci_judge = wilson_ci(hal_judge_n, n)

    # Cohen's Kappa between EM and Judge labels
    em_labels = [r.get("is_hallucinated_em", 0) for r in results]
    judge_labels = [r.get("is_hallucinated_judge", 0) for r in results]
    kappa = cohens_kappa(em_labels, judge_labels)
    disagree_n = sum(1 for a, b in zip(em_labels, judge_labels) if a != b)

    # Uncertainty scores
    vc_uncertainty = [1.0 - (r.get("verbalized_confidence", 50) / 100.0) for r in results]
    sc_uncertainty = [r.get("selfcheck_uncertainty", 0.5) for r in results]
    # LLM-as-Judge: binary uncertainty (1 = hallucinated_judge, 0 = not)
    judge_uncertainty = [float(r.get("is_hallucinated_judge", 0)) for r in results]

    # AUROC against EM labels
    auroc_vc_em = compute_auroc(vc_uncertainty, em_labels)
    auroc_sc_em = compute_auroc(sc_uncertainty, em_labels)
    auroc_judge_em = compute_auroc(judge_uncertainty, em_labels)

    # AUROC against Judge labels (not computed for LLM-as-Judge — circular)
    auroc_vc_judge = compute_auroc(vc_uncertainty, judge_labels)
    auroc_sc_judge = compute_auroc(sc_uncertainty, judge_labels)

    # Calibration (ECE uses EM accuracy)
    confidences = [r.get("verbalized_confidence", 50) / 100.0 for r in results]
    accuracies_em = [1 - r.get("is_hallucinated_em", 0) for r in results]
    ece = compute_ece(confidences, accuracies_em)

    mean_conf = sum(r.get("verbalized_confidence", 50) for r in results) / n
    overconfidence_gap = mean_conf / 100.0 - (1 - hal_rate_em)

    def _fmt(v):
        return round(v, 4) if not math.isnan(v) else None

    return {
        "n": n,
        "exact_match_rate": round(em_rate, 4),
        "abstention_rate": round(abs_rate, 4),
        "hallucination_rate_em": round(hal_rate_em, 4),
        "hallucination_rate_judge": round(hal_rate_judge, 4),
        "hallucination_rate_em_ci_lower": round(ci_em[0], 4),
        "hallucination_rate_em_ci_upper": round(ci_em[1], 4),
        "hallucination_rate_judge_ci_lower": round(ci_judge[0], 4),
        "hallucination_rate_judge_ci_upper": round(ci_judge[1], 4),
        "cohens_kappa": _fmt(kappa),
        "label_disagreements": disagree_n,
        "auroc_vc_em": _fmt(auroc_vc_em),
        "auroc_sc_em": _fmt(auroc_sc_em),
        "auroc_judge_em": _fmt(auroc_judge_em),
        "auroc_vc_judge": _fmt(auroc_vc_judge),
        "auroc_sc_judge": _fmt(auroc_sc_judge),
        "ece": round(ece, 4),
        "mean_verbalized_confidence": round(mean_conf, 2),
        "overconfidence_gap": round(overconfidence_gap, 4),
    }
