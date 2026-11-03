from typing import Any, Dict, Optional

from torch.utils.data import Dataset

__all__ = ["QADataset"]


class QADataset(Dataset):
    """A split of a Hugging Face dataset, formatted for causal-LM fine-tuning.

    Subclasses set `self.dataset` (a `datasets.Dataset`) and implement
    `format(sample) -> (text, question, answer)`:
        text      the full training sequence ("Question: ...\\nAnswer: X")
        question  the evaluation prompt's body (question + choices, no answer)
        answer    the reference answer
    Items carry `text`, `question`, `answer`, `idx` and, when a tokenizer is
    given, `input_ids` / `attention_mask` padded to `max_length`.
    """

    SPLITS = ("train", "valid", "test")

    def __init__(self, split: str, tokenizer: Optional[Any] = None, max_length: int = 512):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = None

    def format(self, sample):
        raise NotImplementedError

    def extra_fields(self, sample) -> Dict[str, Any]:
        return {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        text, question, answer = self.format(sample)
        item: Dict[str, Any] = {"idx": idx}
        if self.tokenizer:
            enc = self.tokenizer(text, truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        item.update(text=text, question=question, answer=answer, **self.extra_fields(sample))
        return item
