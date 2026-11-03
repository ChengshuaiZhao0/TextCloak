"""Extract multiple-choice answers (A-D) from references and model output."""

import re
from typing import Optional

__all__ = ["ANSWER_EXTRACTORS"]

_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}
_VALID = set("ABCD")


def _choice_reference(text: str) -> Optional[str]:
    value = text.strip()
    value = _NUM_TO_LETTER.get(value, value.upper())
    return value if value in _VALID else None


def _choice_prediction(text: str) -> Optional[str]:
    """Extract a leading choice, an answer phrase, a parenthesized choice, or a lone choice."""
    text = text.strip()
    if not text:
        return None
    if text[0].upper() in _VALID:
        return text[0].upper()
    patterns = [
        r"[Aa]nswer\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        r"\(([A-Da-d])\)",
        r"\b([A-Da-d])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


ANSWER_EXTRACTORS = {
    "arc": (_choice_prediction, _choice_reference),
    "medqa": (_choice_prediction, _choice_reference),
}
