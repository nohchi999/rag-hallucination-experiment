"""
src/abstention_ext.py  (Refinement-2 cross-model marker extension; added 2026-06-30)

Extends abstention detection for the cross-model experiment, applied UNIFORMLY to
all generators (same instrument for everyone — "exactly like Haiku"). This mirrors
what was already done for Haiku in the v2 pipeline: when abstentions are phrased in
natural language and miscounted as hallucinations, the marker set is extended.

The extension is a SINGLE anchored pattern: an answer that STARTS with
  "(the) (provided/given) {text|passage|document|excerpt|information} (does not|doesn't|do not) ..."
This catches GPT/Gemini-style abstentions ("The provided text does not mention ...")
that the original markers (which say "context") missed. It is anchored to the START
so a real answer that merely contains a "... does not mention ..." clause mid-sentence
is NOT misflagged.

Haiku-safety (verified): this changes ZERO Haiku rows -> Haiku results stay
bit-identical (F1 = 23.74% -> 9.09%). Effect is Gemini-only (GPT: 0 changes).
Validated against 144 human labels: human-vs-pipeline kappa 0.818 -> 0.872.

The original src/metrics_v2.py is left byte-identical; this is additive.
"""

import re

from src.metrics_v2 import detect_abstention_v2, classify_response

# Anchored opener: answer begins by stating the source does not contain the answer.
_EXT_START_RX = re.compile(
    r"^\W*(the\s+)?(provided\s+|given\s+)?"
    r"(text|passage|document|excerpt|information)\s+"
    r"(does\s+not|doesn'?t|do\s+not)\b",
    re.IGNORECASE,
)


def detect_abstention_ext(answer):
    """detect_abstention_v2 OR the anchored cross-model opener pattern."""
    if detect_abstention_v2(answer):
        return True
    return bool(_EXT_START_RX.match((answer or "").strip()))


def classify_response_ext(result):
    """Disjoint classification using the extended abstention predicate."""
    return classify_response(result, abstention_predicate=detect_abstention_ext)
