"""
src/metrics_v2.py

Corrected metric computation for the RAG hallucination experiment. This module
is a PARALLEL implementation of src/metrics.py that fixes six known bugs in the
original pipeline. The original file is kept unchanged for auditability.

Bugs addressed (see BRIEFING_BUG_FIX.md for full specification):
  #1 Parser default contamination — verbalized_confidence defaulted to 50 when
     the raw response did not start with "Answer:". Fixed by reparse_confidence()
     which returns None (not 50!) when no Confidence annotation is found.
  #2 Confidence semantics undefined for abstentions — computed confidence
     statistics are meaningless over a mixed abstention/answer population.
     compute_ece_multiple() reports ECE variants so the answered-only signal
     is separated from the abstention noise.
  #3 Abstention detector biased against the unconstrained prompt — the original
     marker list missed natural-language abstentions such as "I cannot provide".
     detect_abstention_v2() uses an extended marker list plus regex patterns.
  #4 EM / abstention double-counting — original pipeline let the same response
     be counted in both categories, producing per-cell sums above 100%.
     classify_response() returns exactly one of {abstention, em_correct,
     hallucinated}.
  #5 Degenerate SQuAD data points slipped through — e.g. Q74 (question="k",
     ground_truth="ks"). is_valid_datapoint() filters these out and reports them.
  #6 ECE mixed abstentions with real answers — compute_ece_multiple() returns
     ECE computed (a) over all parsed responses, (b) over non-abstentions only,
     (c) over responses with a parsed confidence, with explicit N.

No new dependencies. Only numpy, sklearn, scipy — identical to metrics.py.
"""

import math
import re
import string

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Bug #3: Extended abstention detection
# ---------------------------------------------------------------------------

EXTENDED_ABSTENTION_MARKERS = [
    # Original markers from src/metrics.py (retained for parity)
    "cannot answer",
    "i don't have enough",
    "not enough information",
    "no relevant information",
    "the context does not",
    "based on the provided context, i cannot",
    "i'm unable to answer",
    "unable to answer",
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
    # New markers — missing from v1, caused unconstrained-prompt abstentions
    # to be falsely labelled as hallucinations.
    "cannot provide",
    "unable to provide",
    "unable to determine",
    "insufficient to answer",
    "insufficient information",
    "no way to determine",
    "cannot be determined",
    "not clear from the context",
    "the context lacks",
    "there is no information",
    "there's no information",
    "no direct information",
    "incomplete or unclear",
    "not a complete question",
    "without a clear question",
    "cannot formulate",
    "i must abstain",
    "i'll abstain",
    "i decline to answer",
]


ABSTENTION_REGEX_PATTERNS = [
    re.compile(r"i\s+(can\s*not|cannot|can't)\s+(answer|provide|determine|tell|say)", re.IGNORECASE),
    re.compile(r"(the\s+)?(provided\s+)?context\s+(does\s+not|doesn't|lacks|contains\s+no)", re.IGNORECASE),
    re.compile(r"no\s+(information|mention|reference|data)\s+(about|regarding|on)", re.IGNORECASE),
    re.compile(r"unable\s+to\s+(answer|provide|determine)", re.IGNORECASE),
    re.compile(r"insufficient\s+(information|context|evidence)", re.IGNORECASE),
]


def detect_abstention_v2(answer):
    """Return True if the answer semantically abstains."""
    if not answer:
        return False
    answer_lower = answer.strip().lower()
    for marker in EXTENDED_ABSTENTION_MARKERS:
        if marker in answer_lower:
            return True
    for pattern in ABSTENTION_REGEX_PATTERNS:
        if pattern.search(answer):
            return True
    return False


# ---------------------------------------------------------------------------
# Bug #1: Re-parse verbalized confidence from raw_text
# ---------------------------------------------------------------------------

_CONFIDENCE_REGEX = re.compile(r"confidence\s*:\s*(-?\d+)", re.IGNORECASE)


def reparse_confidence(raw_text):
    """
    Extract the verbalized confidence from a raw Claude response.

    Returns an int in [0, 100] if a Confidence annotation was found,
    otherwise None. Never returns a default of 50. Values outside [0, 100]
    are clamped.
    """
    if not raw_text:
        return None
    match = _CONFIDENCE_REGEX.search(raw_text)
    if not match:
        return None
    try:
        val = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return max(0, min(100, val))


# ---------------------------------------------------------------------------
# Bug #4: Disjoint classification + SQuAD F1 as secondary metric
# ---------------------------------------------------------------------------

def exact_match(prediction, ground_truth):
    """Case-insensitive substring match. Returns 1 if ground_truth is in prediction."""
    if not prediction or not ground_truth:
        return 0
    return 1 if ground_truth.strip().lower() in prediction.strip().lower() else 0


def normalize_answer(s):
    """SQuAD-style normalization: lowercase, strip punctuation, drop articles, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    s = " ".join(s.split())
    return s


def squad_f1(prediction, ground_truth):
    """Token-level F1 (SQuAD official). Returns 0.0 on empty predictions."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    from collections import Counter
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def classify_response(result, abstention_predicate=detect_abstention_v2):
    """
    Return exactly one of 'abstention', 'em_correct', 'hallucinated'.

    Abstention check takes precedence over EM check so that short ground truths
    (e.g. "KU") appearing as substrings inside abstention prose do not get
    double-counted as exact matches.
    """
    answer = result.get("answer", "")
    ground_truth = result.get("ground_truth", "")

    if abstention_predicate(answer):
        return "abstention"
    if exact_match(answer, ground_truth):
        return "em_correct"
    return "hallucinated"


# ---------------------------------------------------------------------------
# Bug #5: Dataset integrity filter
# ---------------------------------------------------------------------------

def is_valid_datapoint(result):
    """
    Return (is_valid, reason). is_valid=False marks the datapoint as degenerate
    and should exclude it from downstream metrics.
    """
    question = (result.get("question") or "").strip()
    gt = (result.get("ground_truth") or "").strip()

    if len(question) < 5:
        return False, f"question_too_short: {question!r}"
    if len(gt) < 2:
        return False, f"ground_truth_too_short: {gt!r}"
    words = question.split()
    if not any(w.isalpha() and len(w) >= 3 for w in words):
        return False, f"no_meaningful_words: {question!r}"
    if gt.lower() == question.lower():
        return False, f"gt_equals_question: {gt!r}"
    return True, ""


# ---------------------------------------------------------------------------
# Bug #6: ECE variants
# ---------------------------------------------------------------------------

def _compute_ece_single(confidences, accuracies, n_bins=10):
    if not confidences:
        return float("nan")
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


def compute_ece_multiple(results, n_bins=10):
    """
    Compute three ECE variants plus the N used for each. Each input result
    must already carry the v2 fields (verbalized_confidence_v2 possibly None,
    is_abstention_v2 bool, is_hallucinated_em_v2 in {0,1}).

    Returns a dict with keys:
      ece_all, ece_non_abstention, ece_parsed_only,
      n_all, n_non_abstention, n_parsed_only,
      n_unparsed
    """
    all_confs, all_accs = [], []
    non_abs_confs, non_abs_accs = [], []
    parsed_confs, parsed_accs = [], []
    n_unparsed = 0

    for r in results:
        conf = r.get("verbalized_confidence_v2")
        is_abs = bool(r.get("is_abstention_v2"))
        acc = 1 - int(r.get("is_hallucinated_em_v2", 0))

        if conf is None:
            n_unparsed += 1
            continue

        c = conf / 100.0
        all_confs.append(c)
        all_accs.append(acc)
        parsed_confs.append(c)
        parsed_accs.append(acc)
        if not is_abs:
            non_abs_confs.append(c)
            non_abs_accs.append(acc)

    return {
        "ece_all": _compute_ece_single(all_confs, all_accs, n_bins),
        "ece_non_abstention": _compute_ece_single(non_abs_confs, non_abs_accs, n_bins),
        "ece_parsed_only": _compute_ece_single(parsed_confs, parsed_accs, n_bins),
        "n_all": len(all_confs),
        "n_non_abstention": len(non_abs_confs),
        "n_parsed_only": len(parsed_confs),
        "n_unparsed": n_unparsed,
    }


# ---------------------------------------------------------------------------
# Helpers carried over from metrics.py (identical formulas)
# ---------------------------------------------------------------------------

def cohens_kappa(labels1, labels2):
    n = len(labels1)
    if n == 0:
        return float("nan")
    a = np.array(labels1)
    b = np.array(labels2)
    p_o = float(np.mean(a == b))
    p1 = float(np.mean(a == 1))
    p2 = float(np.mean(b == 1))
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def wilson_ci(successes, total, confidence=0.95):
    if total == 0:
        return (0.0, 0.0)
    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _auroc_point(scores, labels):
    if len(set(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_auroc_with_ci(uncertainty_scores, labels, n_boot=1000, confidence=0.95, seed=42):
    """
    AUROC point estimate plus bootstrap percentile CI.

    Pairs where uncertainty is None are dropped before resampling. Returns
    (point, ci_lower, ci_upper, n_used). All three are nan if the class is
    degenerate or no data remain.
    """
    pairs = [(s, l) for s, l in zip(uncertainty_scores, labels) if s is not None]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), 0
    scores = np.array([p[0] for p in pairs], dtype=float)
    lbls = np.array([p[1] for p in pairs], dtype=int)
    n = len(scores)

    point = _auroc_point(scores.tolist(), lbls.tolist())
    if math.isnan(point):
        return float("nan"), float("nan"), float("nan"), n

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s_boot = scores[idx]
        l_boot = lbls[idx]
        if len(set(l_boot.tolist())) < 2:
            continue
        try:
            boots.append(float(roc_auc_score(l_boot, s_boot)))
        except Exception:
            continue

    if not boots:
        return point, float("nan"), float("nan"), n

    alpha = (1 - confidence) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return point, lo, hi, n


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except (TypeError, ValueError):
        return None
    return round(v, nd)


def compute_all_metrics_v2(results, n_bins=10, n_boot=1000):
    """
    Compute the full metrics summary for one cell (prompt_type, condition).
    Expects results enriched with v2 fields.
    """
    n = len(results)
    if n == 0:
        return {}

    # Category rates (disjoint by construction — Bug #4 fix)
    n_abs = sum(1 for r in results if r.get("response_category_v2") == "abstention")
    n_em = sum(1 for r in results if r.get("response_category_v2") == "em_correct")
    n_hal = sum(1 for r in results if r.get("response_category_v2") == "hallucinated")

    abstention_rate = n_abs / n
    em_rate = n_em / n
    hal_rate_em = n_hal / n

    # SQuAD F1 (secondary metric)
    f1_scores = [r.get("squad_f1_v2", 0.0) for r in results]
    f1_mean = float(np.mean(f1_scores)) if f1_scores else float("nan")

    # Judge label rate (unchanged — uses stored is_hallucinated_judge)
    hal_rate_judge = sum(1 for r in results if r.get("is_hallucinated_judge", 0) == 1) / n

    # Wilson CIs
    ci_em = wilson_ci(n_hal, n)
    ci_judge_n = sum(1 for r in results if r.get("is_hallucinated_judge", 0) == 1)
    ci_judge = wilson_ci(ci_judge_n, n)

    # Cohen's Kappa — EM v2 vs Judge v1 (judge label is unchanged)
    em_labels = [1 if r.get("response_category_v2") == "hallucinated" else 0 for r in results]
    judge_labels = [int(r.get("is_hallucinated_judge", 0)) for r in results]
    kappa = cohens_kappa(em_labels, judge_labels)
    disagree_n = sum(1 for a, b in zip(em_labels, judge_labels) if a != b)

    # Uncertainty scores
    vc_uncertainty = [
        (1.0 - (r["verbalized_confidence_v2"] / 100.0)) if r.get("verbalized_confidence_v2") is not None else None
        for r in results
    ]
    sc_uncertainty = [r.get("selfcheck_uncertainty", 0.5) for r in results]
    judge_uncertainty = [float(r.get("is_hallucinated_judge", 0)) for r in results]

    # AUROC primary — against EM v2 labels, plus secondary excluding abstentions
    auroc_vc_em = compute_auroc_with_ci(vc_uncertainty, em_labels, n_boot=n_boot)
    auroc_sc_em = compute_auroc_with_ci(sc_uncertainty, em_labels, n_boot=n_boot)
    auroc_judge_em = compute_auroc_with_ci(judge_uncertainty, em_labels, n_boot=n_boot)
    auroc_vc_judge = compute_auroc_with_ci(vc_uncertainty, judge_labels, n_boot=n_boot)
    auroc_sc_judge = compute_auroc_with_ci(sc_uncertainty, judge_labels, n_boot=n_boot)

    # Non-abstention subset
    non_abs = [r for r in results if r.get("response_category_v2") != "abstention"]
    em_labels_na = [1 if r.get("response_category_v2") == "hallucinated" else 0 for r in non_abs]
    vc_u_na = [
        (1.0 - (r["verbalized_confidence_v2"] / 100.0)) if r.get("verbalized_confidence_v2") is not None else None
        for r in non_abs
    ]
    sc_u_na = [r.get("selfcheck_uncertainty", 0.5) for r in non_abs]
    auroc_vc_em_na = compute_auroc_with_ci(vc_u_na, em_labels_na, n_boot=n_boot)
    auroc_sc_em_na = compute_auroc_with_ci(sc_u_na, em_labels_na, n_boot=n_boot)

    # ECE variants — Bug #6 fix
    eces = compute_ece_multiple(results, n_bins=n_bins)

    # Mean confidence (only over parsed values — Bug #1 means v1 was contaminated)
    parsed_confs = [r["verbalized_confidence_v2"] for r in results if r.get("verbalized_confidence_v2") is not None]
    mean_conf_parsed = float(np.mean(parsed_confs)) if parsed_confs else float("nan")
    parsed_confs_na = [r["verbalized_confidence_v2"] for r in non_abs if r.get("verbalized_confidence_v2") is not None]
    mean_conf_na = float(np.mean(parsed_confs_na)) if parsed_confs_na else float("nan")

    overconfidence_gap = mean_conf_parsed / 100.0 - (1 - hal_rate_em) if not math.isnan(mean_conf_parsed) else float("nan")

    def _auroc_row(prefix, tup):
        point, lo, hi, n_used = tup
        return {
            prefix: _fmt(point),
            prefix + "_ci_lower": _fmt(lo),
            prefix + "_ci_upper": _fmt(hi),
            prefix + "_n": n_used,
        }

    out = {
        "n": n,
        "n_em_correct": n_em,
        "n_hallucinated": n_hal,
        "n_abstention": n_abs,
        "em_rate": round(em_rate, 4),
        "abstention_rate": round(abstention_rate, 4),
        "hallucination_rate_em": round(hal_rate_em, 4),
        "hallucination_rate_judge": round(hal_rate_judge, 4),
        "hallucination_rate_em_ci_lower": round(ci_em[0], 4),
        "hallucination_rate_em_ci_upper": round(ci_em[1], 4),
        "hallucination_rate_judge_ci_lower": round(ci_judge[0], 4),
        "hallucination_rate_judge_ci_upper": round(ci_judge[1], 4),
        "cohens_kappa": _fmt(kappa),
        "label_disagreements": disagree_n,
        "squad_f1_mean": _fmt(f1_mean),
        "mean_verbalized_confidence_parsed": _fmt(mean_conf_parsed, 2),
        "mean_verbalized_confidence_non_abstention": _fmt(mean_conf_na, 2),
        "overconfidence_gap": _fmt(overconfidence_gap),
        "ece_all": _fmt(eces["ece_all"]),
        "ece_non_abstention": _fmt(eces["ece_non_abstention"]),
        "ece_parsed_only": _fmt(eces["ece_parsed_only"]),
        "ece_n_all": eces["n_all"],
        "ece_n_non_abstention": eces["n_non_abstention"],
        "ece_n_unparsed": eces["n_unparsed"],
    }
    out.update(_auroc_row("auroc_vc_em", auroc_vc_em))
    out.update(_auroc_row("auroc_sc_em", auroc_sc_em))
    out.update(_auroc_row("auroc_judge_em", auroc_judge_em))
    out.update(_auroc_row("auroc_vc_judge", auroc_vc_judge))
    out.update(_auroc_row("auroc_sc_judge", auroc_sc_judge))
    out.update(_auroc_row("auroc_vc_em_non_abstention", auroc_vc_em_na))
    out.update(_auroc_row("auroc_sc_em_non_abstention", auroc_sc_em_na))
    return out
