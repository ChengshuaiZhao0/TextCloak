"""Logging, meters and seeding."""

import logging
import os
import random
import sys

import numpy as np
import torch

__all__ = ["logger", "setup_logging", "AverageMeter", "set_seed", "count_parameters", "log_memory"]

logger = logging.getLogger("textcloak")


def setup_logging(log_path=None):
    """Log to stderr and, if given, to `log_path` (appended)."""
    fmt = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)


class AverageMeter:
    """Latest value and running average."""

    def __init__(self, name):
        self.name = name
        self.val, self.sum, self.count, self.avg = None, 0.0, 0, 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    seed = int(seed) % (2 ** 32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_memory(tag=""):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    logger.info(f"  [MEM {tag}] allocated={alloc:.1f}GB reserved={reserved:.1f}GB")
