import torch

from .datasets import ARCDataset, MedQADataset
from .protected import ProtectedDataset
from .text import ANSWER_MARK, get_answer_start_positions, original_stem, parse_stems, stem_tail_boundary

__all__ = [
    "DATASETS", "build_dataset", "make_loader", "ProtectedDataset", "ANSWER_MARK", "get_answer_start_positions",
    "original_stem", "parse_stems", "stem_tail_boundary",
]

DATASETS = {
    "arc": ARCDataset,
    "medqa": MedQADataset,
}


def build_dataset(cfg, split, tokenizer=None):
    """One split of `cfg.dataset.name`; `cfg.dataset.args` go to its constructor."""
    cls = DATASETS[cfg.dataset.name]
    ds = cls(split, tokenizer=tokenizer, max_length=int(cfg.dataset.max_length),
             random_seed=int(cfg.run.seed), **dict(cfg.get("dataset.args") or {}))
    limit = int(cfg.get(f"dataset.limit_{split}", 0) or 0)
    if limit and limit < len(ds):  # quick tests only: the first `limit` samples
        ds = torch.utils.data.Subset(ds, range(limit))
    return ds


def make_loader(dataset, batch_size, shuffle, drop_last, num_workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                                       pin_memory=True, num_workers=int(num_workers),
                                       drop_last=drop_last)
