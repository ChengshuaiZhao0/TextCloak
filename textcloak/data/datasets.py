"""The multiple-choice benchmarks, loaded from the Hugging Face Hub.

Every split is carved deterministically from the official data with
`random_seed`, so a seed fixes the exact train / valid / test samples.

| name  | Hugging Face dataset          | train                  | valid           | test            |
|-------|-------------------------------|------------------------|-----------------|-----------------|
| arc   | allenai/ai2_arc (Challenge)   | official train (1119)  | official (299)  | official (1172) |
| medqa | GBaker/MedQA-USMLE-4-options  | train minus valid, cap | 10% of train    | official (1273) |

"cap" is `max_train_samples` (1200 in configs/medqa.yaml).
"""

from typing import Optional

from datasets import load_dataset

from .base import QADataset

__all__ = ["ARCDataset", "MedQADataset"]


def _train_valid(full, split, random_seed, val_split_ratio, max_train_samples):
    """Hold out `val_split_ratio` of `full` as valid; cap train at `max_train_samples`."""
    val_size = max(1, int(float(val_split_ratio) * len(full)))
    parts = full.train_test_split(test_size=val_size, seed=random_seed)
    if split == "valid":
        return parts["test"]
    train = parts["train"]
    if max_train_samples is not None and len(train) > int(max_train_samples):
        train = train.shuffle(seed=random_seed).select(range(int(max_train_samples)))
    return train


class ARCDataset(QADataset):
    """ARC-Challenge science questions with four answer choices."""

    _NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}

    def __init__(self, split, tokenizer=None, max_length=256, random_seed=42,
                 config_name="ARC-Challenge"):
        super().__init__(split, tokenizer, max_length)
        hf_split = "validation" if split == "valid" else split
        self.dataset = load_dataset("allenai/ai2_arc", config_name, split=hf_split)

    def format(self, sample):
        def normalize(label):
            return self._NUM_TO_LETTER.get(label, label)

        choices = "\n".join(
            f"({normalize(label)}) {text}"
            for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])
        )
        answer = normalize(sample["answerKey"])
        question = f"{sample['question']}\n{choices}"
        return f"Question: {question}\nAnswer: {answer}", question, answer


class MedQADataset(QADataset):
    """MedQA-USMLE clinical vignettes with four answer choices."""

    def __init__(self, split, tokenizer=None, max_length=512, random_seed=42,
                 val_split_ratio=0.1, max_train_samples: Optional[int] = 1200):
        super().__init__(split, tokenizer, max_length)
        name = "GBaker/MedQA-USMLE-4-options"
        if split == "test":
            self.dataset = load_dataset(name, split="test")
        else:
            self.dataset = _train_valid(load_dataset(name, split="train"), split, random_seed,
                                        val_split_ratio, max_train_samples)

    def format(self, sample):
        options = sample["options"]
        choices = "\n".join(
            f"({label}) {options[label]}"
            for label in "ABCD"
            if options.get(label) is not None
        )
        answer = str(sample["answer_idx"]).strip().upper()
        question = f"{sample['question']}\n{choices}"
        return f"Question: {question}\nAnswer: {answer}", question, answer
