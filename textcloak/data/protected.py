from typing import Any, Dict

import torch
from torch.utils.data import Dataset

__all__ = ["ProtectedDataset"]


class ProtectedDataset(Dataset):
    """A protected dataset written by `textcloak.protect` (a `.pt` file).

    The file holds `polluted_input_ids`, `attention_mask`, `original_input_ids`
    (token ids of the surrogate's tokenizer) and the lists `original_texts`,
    `polluted_texts`, `questions`, `answers`. With `tokenizer`, the texts are
    re-tokenized, so any model can be fine-tuned on the same protected text.
    """

    def __init__(self, path: str, tokenizer=None, max_length: int = 256):
        data = torch.load(path, weights_only=False)
        self.polluted_input_ids = data["polluted_input_ids"]
        self.attention_mask = data["attention_mask"]
        self.original_input_ids = data["original_input_ids"]
        self.original_texts = data["original_texts"]
        self.polluted_texts = data["polluted_texts"]
        self.questions = data["questions"]
        self.answers = data["answers"]
        if tokenizer is not None:
            enc = tokenizer(list(self.polluted_texts), return_tensors="pt", padding="max_length",
                            truncation=True, max_length=max_length)
            self.polluted_input_ids, self.attention_mask = enc["input_ids"], enc["attention_mask"]
            self.original_input_ids = tokenizer(list(self.original_texts), return_tensors="pt",
                                                padding="max_length", truncation=True,
                                                max_length=max_length)["input_ids"]

    def __len__(self) -> int:
        return self.polluted_input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.polluted_input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "original_input_ids": self.original_input_ids[idx],
            "idx": idx,
            "text": self.original_texts[idx],
            "polluted_text": self.polluted_texts[idx],
            "question": self.questions[idx],
            "answer": self.answers[idx],
        }
