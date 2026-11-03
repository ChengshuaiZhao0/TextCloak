"""Frozen scorers for the semantic-fidelity and naturalness reward terms.

S_sem: SBERT (all-MiniLM-L6-v2) cosine similarity between a clean sample and its
       protected version (full texts), averaged over the batch.
S_ppl: min{1, PPL(clean) / PPL(protected)} under GPT-2, averaged over the batch.
Both models load lazily, once, and only when their reward weight is > 0.
"""

import gc
from typing import List

import torch
import torch.nn.functional as F

from .utils import logger

__all__ = ["sbert_embed", "gpt2_ppl", "release_scorers", "report_dataset_quality"]

SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GPT2_NAME = "gpt2"

_SBERT = None
_GPT2 = None
_GPT2_TOK = None


def _sbert(device):
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[Reward] loading SBERT ({SBERT_NAME}) ...")
        _SBERT = SentenceTransformer(SBERT_NAME, device=str(device))
    return _SBERT


def _gpt2(device):
    global _GPT2, _GPT2_TOK
    if _GPT2 is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        logger.info(f"[Reward] loading GPT-2 ({GPT2_NAME}) for perplexity ...")
        _GPT2_TOK = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
        _GPT2_TOK.pad_token = _GPT2_TOK.eos_token
        _GPT2 = GPT2LMHeadModel.from_pretrained(GPT2_NAME).to(device).eval()
        for p in _GPT2.parameters():
            p.requires_grad = False
    return _GPT2, _GPT2_TOK


@torch.no_grad()
def sbert_embed(texts: List[str], device) -> torch.Tensor:
    """SBERT embeddings [N, dim]."""
    return _sbert(device).encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32)


@torch.no_grad()
def gpt2_ppl(texts: List[str], device, micro_batch: int = 8) -> torch.Tensor:
    """Per-sample GPT-2 perplexity [N] (texts truncated to GPT-2's 1024-token context)."""
    model, tok = _gpt2(device)
    out = []
    for start in range(0, len(texts), micro_batch):
        enc = tok(texts[start:start + micro_batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        logits = model(input_ids=ids, attention_mask=mask).logits
        labels = ids[:, 1:].masked_fill(mask[:, 1:] == 0, -100)
        loss = F.cross_entropy(logits[:, :-1, :].float().transpose(1, 2), labels,
                               ignore_index=-100, reduction="none")
        n_tok = (labels != -100).sum(dim=1).clamp(min=1)
        out.append(torch.exp(loss.sum(dim=1) / n_tok))
        del logits, loss
    return torch.cat(out)


def release_scorers():
    global _SBERT, _GPT2, _GPT2_TOK
    _SBERT, _GPT2, _GPT2_TOK = None, None, None
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def report_dataset_quality(dataset, device, max_samples: int = 500):
    """Log S_sem, perplexities and S_ppl of a protected dataset (first `max_samples`)."""
    clean, protected = list(dataset.original_texts), list(dataset.polluted_texts)
    n = min(max_samples, len(clean))
    clean, protected = clean[:n], protected[:n]
    logger.info(f"[DataQuality] scoring {n}/{len(dataset.original_texts)} samples (SBERT + GPT-2) ...")
    s_sem = F.cosine_similarity(sbert_embed(clean, device), sbert_embed(protected, device), dim=-1)
    ppl_clean, ppl_prot = gpt2_ppl(clean, device), gpt2_ppl(protected, device)
    s_ppl = torch.minimum(torch.ones_like(ppl_clean), ppl_clean / ppl_prot)
    identical = sum(a == b for a, b in zip(clean, protected))
    stats = dict(n=n, s_sem=s_sem.mean().item(), s_sem_min=s_sem.min().item(),
                 ppl_clean=ppl_clean.mean().item(), ppl_protected=ppl_prot.mean().item(),
                 s_ppl=s_ppl.mean().item(), unmodified_pct=100.0 * identical / n)
    logger.info(
        f"[DataQuality] S_sem(SBERT cos)={stats['s_sem']:.4f} (min={stats['s_sem_min']:.4f}) | "
        f"PPL_clean={stats['ppl_clean']:.2f} PPL_protected={stats['ppl_protected']:.2f} "
        f"(ratio S_ppl={stats['s_ppl']:.4f}) | unmodified={stats['unmodified_pct']:.1f}%"
    )
    release_scorers()
    return stats
