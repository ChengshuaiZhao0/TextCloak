"""Boundaries within a multiple-choice sample formatted for causal-LM fine-tuning.

Each sample is represented as:

    Question: {stem}
    (A) ...
    (D) ...
    Answer: {answer}

TextCloak rewrites only the stem. Choices and the answer are kept verbatim,
and only answer tokens are supervised during fine-tuning.
"""

from typing import List

import torch

__all__ = [
    "ANSWER_MARK", "parse_stems", "stem_tail_boundary", "original_stem",
    "get_answer_start_positions",
]

ANSWER_MARK = "\nAnswer:"
FIRST_CHOICE_MARK = "\n(A)"


def parse_stems(generated_text: str, batch_size: int):
    """Split one rewriter response into exactly batch_size candidate stems."""
    if "<think>" in generated_text:
        think_end = generated_text.find("</think>")
        if think_end != -1:
            generated_text = generated_text[think_end + len("</think>"):].strip()
    stems = [stem.strip() for stem in generated_text.split("---")]
    if len(stems) >= batch_size:
        return stems[:batch_size], True
    return stems + [""] * (batch_size - len(stems)), False


def stem_tail_boundary(text: str) -> int:
    """Return the character index where the first answer choice begins."""
    return text.find(FIRST_CHOICE_MARK)


def original_stem(text: str) -> str:
    """Return the question stem, or an empty string for malformed input."""
    marker = stem_tail_boundary(text)
    return text[len("Question: "):marker] if marker != -1 else ""


def get_answer_start_positions(batch_text: List[str], input_ids: torch.Tensor,
                               max_length: int, tokenizer) -> List[int]:
    """Return the token index where each answer begins.

    Samples without an answer marker receive the sequence length, so none of
    their tokens are supervised.
    """
    seq_len = input_ids.size(1)
    prefixes, no_marker = [], []
    for i, text in enumerate(batch_text):
        pos = text.find(ANSWER_MARK)
        if pos == -1:
            no_marker.append(i)
            prefixes.append("")
        else:
            prefixes.append(text[:pos])
    enc = tokenizer(prefixes, truncation=True, max_length=max_length,
                    return_tensors="pt", padding=True)
    starts = enc["attention_mask"].sum(dim=1).tolist()
    for i in no_marker:
        starts[i] = seq_len
    return starts
