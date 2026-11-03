"""CPU-only checks: configs, text boundaries, answer extraction (no downloads)."""

import os

import pytest
import torch

from textcloak.config import load_config, output_dir
from textcloak.data.datasets import MedQADataset
from textcloak.data.text import (
    get_answer_start_positions,
    original_stem,
    parse_stems,
    stem_tail_boundary,
)
from textcloak.eval.answers import ANSWER_EXTRACTORS

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")
MC = (
    "Question: Which gas do plants absorb?\n"
    "(A) oxygen\n(B) carbon dioxide\n(C) helium\n(D) neon\nAnswer: B"
)


def test_arc_config_loads():
    cfg = load_config(
        os.path.join(CONFIGS, "arc.yaml"),
        ["--run.seed=7", "--protect.lambda_sem=0"],
    )
    assert cfg.dataset.name == "arc"
    assert cfg.protect.lambda_sem == 0
    assert cfg.run.seed == 7
    assert output_dir(cfg) == "outputs/arc/seed7"
    assert isinstance(cfg.finetune.lr, float)
    assert cfg.model.lora.rank == 8


def test_medqa_config_loads():
    cfg = load_config(os.path.join(CONFIGS, "medqa.yaml"))
    assert cfg.dataset.name == "medqa"
    assert cfg.dataset.max_length == 512
    assert cfg.dataset.args.max_train_samples == 1200
    assert cfg.protect.batch_size == 4
    assert cfg.protect.group_size == 4  # inherited from _base.yaml
    assert output_dir(cfg) == "outputs/medqa/seed42"


def test_medqa_format():
    sample = {
        "question": "A 45-year-old man presents with chest pain. What is the next step?",
        "options": {"A": "ECG", "B": "MRI", "C": "Discharge", "D": "Biopsy"},
        "answer_idx": "a",
    }
    text, question, answer = MedQADataset.format(None, sample)
    assert answer == "A"
    assert text == f"Question: {question}\nAnswer: A"
    assert original_stem(text) == sample["question"]
    assert text[stem_tail_boundary(text):].startswith("\n(A) ECG\n(B) MRI")


def test_override_forms():
    cfg = load_config(
        os.path.join(CONFIGS, "arc.yaml"),
        ["--protect.engine", "hf", "--protect.generator_path=None"],
    )
    assert cfg.protect.engine == "hf"
    assert cfg.protect.generator_path is None


def test_stem_boundaries():
    assert MC[stem_tail_boundary(MC):].startswith("\n(A) oxygen")
    assert original_stem(MC) == "Which gas do plants absorb?"
    assert stem_tail_boundary("malformed") == -1
    assert original_stem("malformed") == ""


def test_answer_start_positions_count_prefix_tokens():
    class WordTokenizer:
        def __call__(self, texts, **kwargs):
            longest = max(len(text.split()) for text in texts)
            masks = [
                [1] * len(text.split()) + [0] * (longest - len(text.split()))
                for text in texts
            ]
            return {"attention_mask": torch.tensor(masks)}

    ids = torch.zeros(2, 50, dtype=torch.long)
    starts = get_answer_start_positions([MC, "no marker here"], ids, 50, WordTokenizer())
    assert starts == [len(MC.split("\nAnswer:")[0].split()), 50]


def test_parse_stems():
    assert parse_stems("one\n---\ntwo\n---\nthree", 2) == (["one", "two"], True)
    assert parse_stems("only one", 3) == (["only one", "", ""], False)
    assert parse_stems("<think>hmm</think> a --- b", 2) == (["a", "b"], True)


@pytest.mark.parametrize(
    "prediction,reference,expected",
    [
        ("B", "B", True),
        (" (c) because", "C", True),
        ("The answer is D", "4", True),
        ("", "A", False),
        ("E", "A", False),
    ],
)
@pytest.mark.parametrize("dataset", ["arc", "medqa"])
def test_choice_answers(dataset, prediction, reference, expected):
    prediction_fn, reference_fn = ANSWER_EXTRACTORS[dataset]
    matched = (
        prediction_fn(prediction) is not None
        and prediction_fn(prediction) == reference_fn(reference)
    )
    assert matched == expected
