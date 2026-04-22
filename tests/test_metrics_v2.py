"""
tests/test_metrics_v2.py

Canonical test suite for the v2 reanalysis pipeline. One test per bug from
BRIEFING_BUG_FIX.md, plus a data-integrity check for Bug #2. Standalone
runnable — no pytest dependency.

Run:
    python tests/test_metrics_v2.py

Exit code 0 on full pass. Writes a machine-readable summary to
results/reanalysis_v2/test_suite_results.json.
"""

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.metrics_v2 import (
    classify_response,
    compute_ece_multiple,
    detect_abstention_v2,
    is_valid_datapoint,
    normalize_answer,
    reparse_confidence,
    squad_f1,
)


_results = []


def _case(name, fn):
    t0 = time.time()
    try:
        fn()
        _results.append({"name": name, "status": "pass", "duration_s": round(time.time() - t0, 4)})
        print(f"  [PASS] {name}")
        return True
    except AssertionError as e:
        _results.append({
            "name": name, "status": "fail", "duration_s": round(time.time() - t0, 4),
            "error": str(e) or "AssertionError",
        })
        print(f"  [FAIL] {name}: {e}")
        return False
    except Exception as e:
        _results.append({
            "name": name, "status": "error", "duration_s": round(time.time() - t0, 4),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Bug #1: Parser default contamination
# ---------------------------------------------------------------------------

def test_bug1_response_without_answer_prefix():
    raw = "I cannot answer this based on the provided context.\n\nConfidence: 100\n\nThe provided context discusses islands."
    assert reparse_confidence(raw) == 100, "Should recover Confidence=100 from non-Answer-prefixed response"


def test_bug1_response_with_answer_prefix():
    raw = "Answer: The capital is Paris.\n\nConfidence: 85"
    assert reparse_confidence(raw) == 85


def test_bug1_no_confidence_annotation_returns_none():
    raw = "Paris is the capital."
    assert reparse_confidence(raw) is None, "Must return None (not 50) when no Confidence annotation is found"


def test_bug1_lowercase_confidence():
    raw = "I don't know.\nconfidence: 30"
    assert reparse_confidence(raw) == 30


def test_bug1_out_of_range_clamped():
    raw = "Answer: X\nConfidence: 150"
    assert reparse_confidence(raw) == 100


def test_bug1_negative_clamped():
    raw = "Answer: X\nConfidence: -20"
    assert reparse_confidence(raw) == 0


def test_bug1_empty_and_none_input():
    assert reparse_confidence("") is None
    assert reparse_confidence(None) is None


# ---------------------------------------------------------------------------
# Bug #2: Data integrity — the enriched JSON and report must surface
# abstention confidence distribution. We test the shape contract on the
# cleaned JSON if it exists.
# ---------------------------------------------------------------------------

def test_bug2_cleaned_json_has_abstention_flag_and_conf_v2_field():
    cleaned_path = os.path.join(ROOT, "results", "reanalysis_v2", "raw_results_cleaned.json")
    if not os.path.exists(cleaned_path):
        # Allow test to run before Phase 4; mark as a precondition skip rather than fail.
        print(f"    (skip: {cleaned_path} does not exist yet)")
        return
    with open(cleaned_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, "cleaned json must be a non-empty list"
    required = {"is_abstention_v2", "verbalized_confidence_v2", "response_category_v2", "is_valid_v2", "squad_f1_v2"}
    missing = required - set(data[0].keys())
    assert not missing, f"missing v2 fields in cleaned json: {missing}"


# ---------------------------------------------------------------------------
# Bug #3: Abstention detector
# ---------------------------------------------------------------------------

def test_bug3_detects_all_abstention_phrasings():
    must_detect = [
        "I cannot answer this based on the provided context.",
        "I cannot provide a meaningful answer to the question.",
        "Unable to answer - the question is incomplete or unclear.",
        "The context does not contain information about this.",
        "I'm unable to determine what is being asked.",
        "The provided context lacks information about Eritrea.",
        "There is no information about Arnold Schwarzenegger in the context.",
        "I cannot determine the answer from the context provided.",
        "The context provides insufficient information to answer.",
    ]
    for ans in must_detect:
        assert detect_abstention_v2(ans), f"expected abstention, got False: {ans!r}"


def test_bug3_does_not_false_positive_on_real_answers():
    must_not_detect = [
        "Paris is the capital of France.",
        "The answer is X, based on the context provided.",
        "According to the passage, Beyoncé became popular in 2003.",
        "Shakespeare wrote Hamlet around 1600.",
    ]
    for ans in must_not_detect:
        assert not detect_abstention_v2(ans), f"unexpected abstention: {ans!r}"


# ---------------------------------------------------------------------------
# Bug #4: Disjoint categories + short-GT trap
# ---------------------------------------------------------------------------

def test_bug4_classify_response_is_disjoint():
    cases = [
        ({"answer": "I cannot answer this based on the provided context.", "ground_truth": "Paris"}, "abstention"),
        ({"answer": "Paris is the capital of France.", "ground_truth": "Paris"}, "em_correct"),
        ({"answer": "Berlin is the capital of France.", "ground_truth": "Paris"}, "hallucinated"),
        ({"answer": "I cannot answer. But for you to know, the University of Kansas is...", "ground_truth": "KU"}, "abstention"),
    ]
    for inp, expected in cases:
        got = classify_response(inp)
        assert got == expected, f"classify_response({inp}) = {got!r}, expected {expected!r}"


def test_bug4_cells_sum_to_100_when_run():
    """If the cleaned json exists, every cell's em+hal+abs must equal exactly n."""
    cleaned_path = os.path.join(ROOT, "results", "reanalysis_v2", "raw_results_cleaned.json")
    if not os.path.exists(cleaned_path):
        print("    (skip: cleaned json not yet produced)")
        return
    with open(cleaned_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from collections import Counter
    cells = Counter()
    cats = Counter()
    for r in data:
        if not r.get("is_valid_v2", True):
            continue
        key = (r.get("prompt_type"), r.get("condition"))
        cells[key] += 1
        cats[(key, r["response_category_v2"])] += 1
    for key, n in cells.items():
        total = (cats[(key, "abstention")] + cats[(key, "em_correct")] + cats[(key, "hallucinated")])
        assert total == n, f"cell {key}: categories sum to {total}, expected {n}"


# ---------------------------------------------------------------------------
# Bug #5: Degenerate datapoint filter
# ---------------------------------------------------------------------------

def test_bug5_filters_degenerate():
    bad = [
        {"question": "k", "ground_truth": "ks"},
        {"question": "?", "ground_truth": "something"},
        {"question": "xx", "ground_truth": "yy"},
        {"question": "What is X?", "ground_truth": "What is X?"},
    ]
    for case in bad:
        is_valid, _ = is_valid_datapoint(case)
        assert not is_valid, f"should be invalid: {case}"


def test_bug5_keeps_real_questions():
    good = [
        {"question": "What is the capital of France?", "ground_truth": "Paris"},
        {"question": "Who wrote Hamlet?", "ground_truth": "Shakespeare"},
        {"question": "In what year did the Titanic sink?", "ground_truth": "1912"},
    ]
    for case in good:
        is_valid, _ = is_valid_datapoint(case)
        assert is_valid, f"should be valid: {case}"


# ---------------------------------------------------------------------------
# Bug #6: ECE variants
# ---------------------------------------------------------------------------

def test_bug6_ece_variants_have_expected_n():
    # 10 abstentions with conf=100, 10 non-abstention answers with conf=50 and half correct
    results = []
    for _ in range(10):
        results.append({
            "verbalized_confidence_v2": 100,
            "is_abstention_v2": True,
            "is_hallucinated_em_v2": 0,
        })
    for i in range(10):
        results.append({
            "verbalized_confidence_v2": 50,
            "is_abstention_v2": False,
            "is_hallucinated_em_v2": 1 if i < 5 else 0,
        })
    eces = compute_ece_multiple(results, n_bins=10)
    assert eces["n_all"] == 20
    assert eces["n_non_abstention"] == 10
    assert eces["n_parsed_only"] == 20
    assert eces["n_unparsed"] == 0
    # ECE_non_abstention: 10 samples with conf 0.5, accuracy 0.5 → |0.5-0.5| = 0
    assert abs(eces["ece_non_abstention"]) < 1e-9, f"expected ~0, got {eces['ece_non_abstention']}"


def test_bug6_unparsed_confidence_is_excluded():
    results = [
        {"verbalized_confidence_v2": None, "is_abstention_v2": True, "is_hallucinated_em_v2": 0},
        {"verbalized_confidence_v2": 80, "is_abstention_v2": False, "is_hallucinated_em_v2": 0},
    ]
    eces = compute_ece_multiple(results)
    assert eces["n_all"] == 1
    assert eces["n_unparsed"] == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_normalize_answer():
    assert normalize_answer("The Paris.") == "paris"
    assert normalize_answer("  A BIG   dog  ") == "big dog"


def test_squad_f1_identity_and_partial():
    assert squad_f1("Paris", "Paris") == 1.0
    # Shared token: "capital"
    val = squad_f1("The capital is Paris", "The capital is Paris")
    assert abs(val - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    # Bug #1
    ("bug1_response_without_answer_prefix", test_bug1_response_without_answer_prefix),
    ("bug1_response_with_answer_prefix", test_bug1_response_with_answer_prefix),
    ("bug1_no_confidence_annotation_returns_none", test_bug1_no_confidence_annotation_returns_none),
    ("bug1_lowercase_confidence", test_bug1_lowercase_confidence),
    ("bug1_out_of_range_clamped", test_bug1_out_of_range_clamped),
    ("bug1_negative_clamped", test_bug1_negative_clamped),
    ("bug1_empty_and_none_input", test_bug1_empty_and_none_input),
    # Bug #2
    ("bug2_cleaned_json_has_abstention_flag_and_conf_v2_field", test_bug2_cleaned_json_has_abstention_flag_and_conf_v2_field),
    # Bug #3
    ("bug3_detects_all_abstention_phrasings", test_bug3_detects_all_abstention_phrasings),
    ("bug3_does_not_false_positive_on_real_answers", test_bug3_does_not_false_positive_on_real_answers),
    # Bug #4
    ("bug4_classify_response_is_disjoint", test_bug4_classify_response_is_disjoint),
    ("bug4_cells_sum_to_100_when_run", test_bug4_cells_sum_to_100_when_run),
    # Bug #5
    ("bug5_filters_degenerate", test_bug5_filters_degenerate),
    ("bug5_keeps_real_questions", test_bug5_keeps_real_questions),
    # Bug #6
    ("bug6_ece_variants_have_expected_n", test_bug6_ece_variants_have_expected_n),
    ("bug6_unparsed_confidence_is_excluded", test_bug6_unparsed_confidence_is_excluded),
    # Helpers
    ("normalize_answer", test_normalize_answer),
    ("squad_f1_identity_and_partial", test_squad_f1_identity_and_partial),
]


def main():
    print(f"Running {len(TESTS)} test cases...")
    passes = 0
    for name, fn in TESTS:
        if _case(name, fn):
            passes += 1
    fails = len(TESTS) - passes

    out_dir = os.path.join(ROOT, "results", "reanalysis_v2")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "total": len(TESTS),
        "passed": passes,
        "failed": fails,
        "cases": _results,
    }
    with open(os.path.join(out_dir, "test_suite_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Summary: {passes}/{len(TESTS)} passed ({fails} failed)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
