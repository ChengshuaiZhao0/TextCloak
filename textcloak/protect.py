"""Stages 1 and 2 of TextCloak: train the rewriter G with GRPO, then protect the dataset.

    python -m textcloak.protect configs/arc.yaml [--key=value ...]

Stage 1 trains G's LoRA against a surrogate target T. For every training batch,
G writes K candidate rewrites of the batch's stems; each candidate is scored by
fine-tuning T on it for one step (from the same state) and measuring T's loss
on a clean validation batch (the degradation reward d_k), minus hinge penalties
when the rewrite drifts in meaning (SBERT) or naturalness (GPT-2 perplexity):

    r_k = lambda_d * d_k - lambda_sem * [tau_sem - S_sem]_+ - lambda_ppl * [tau_ppl - S_ppl]_+

G is updated with the clipped GRPO objective on group-normalized rewards.
Stage 2 runs the trained G once over the whole training set (with vLLM or HF
generate) and writes the protected dataset:

    <output_dir>/protected.pt      token ids + texts, input to textcloak.finetune
    <output_dir>/protected.jsonl   one record per sample: original and rewritten stem and text
    <output_dir>/generator/        G's LoRA adapter (protect.save_generator)
"""

import argparse
import gc
import json
import os
import shutil
import tempfile
import time

import torch
import torch.nn.functional as F

from .config import load_config, output_dir
from .data import (build_dataset, get_answer_start_positions, make_loader, original_stem,
                   parse_stems, stem_tail_boundary)
from .models.cloak import TextCloakModel
from .rewards import gpt2_ppl, sbert_embed
from .utils import AverageMeter, count_parameters, log_memory, logger, set_seed, setup_logging


# =============================================================================
# Surrogate helpers
# =============================================================================

def build_sft_labels(ids, mask, answer_starts):
    """Shifted labels with everything but the answer set to -100."""
    labels = ids[:, 1:].contiguous().masked_fill(mask[:, 1:].contiguous() == 0, -100)
    for i in range(ids.size(0)):
        end = answer_starts[i] - 1  # shifted: position j predicts token j+1
        if end > 0:
            labels[i, :end] = -100
    return labels


def snapshot(params, optimizer):
    """Copies of T's LoRA weights and optimizer state."""
    return ([p.data.clone() for p in params],
            {k: {kk: vv.clone() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
             for k, v in optimizer.state.items()})


def restore(params, optimizer, param_snap, opt_snap):
    for p, saved in zip(params, param_snap):
        p.data.copy_(saved)
    if opt_snap:
        for k, v in opt_snap.items():
            if k in optimizer.state:
                for kk, vv in v.items():
                    if torch.is_tensor(vv):
                        optimizer.state[k][kk].copy_(vv)
    else:  # the snapshot predates the first step: drop state created since
        optimizer.state.clear()


def generation_budget(cfg, batch_size):
    """Max new tokens for G: protect.max_new_tokens, else ~30 tokens per stem, capped at 512."""
    n = int(cfg.protect.get("max_new_tokens", 0) or 0)
    if n > 0:
        return n
    return min(int(30 * batch_size * float(cfg.protect.gen_budget_ratio)) + 100, 512)


def fill_with_original(stem, text):
    """A failed parse keeps the sample's original stem (the whole text if it has no stem)."""
    if stem:
        return stem
    return original_stem(text) if stem_tail_boundary(text) != -1 else text


# =============================================================================
# Stage 1: one GRPO epoch
# =============================================================================

def train_epoch(model, train_loader, val_loader, gen_optimizer, target_optimizer, epoch,
                criterion, device, tau, cfg):
    p = cfg.protect
    K = int(p.group_size)
    clip_eps = float(p.clip_eps)
    lambda_d, lambda_sem, lambda_ppl = float(p.lambda_d), float(p.lambda_sem), float(p.lambda_ppl)
    tau_sem, tau_ppl = float(p.tau_sem), float(p.tau_ppl)
    shaping = lambda_sem > 0 or lambda_ppl > 0
    max_length = int(cfg.dataset.max_length)
    max_new_tokens = generation_budget(cfg, int(p.batch_size))
    if epoch == 1:
        logger.info(f"  max_new_tokens (G generation budget) = {max_new_tokens}")
        if shaping:
            logger.info(f"[Reward] lambda_d={lambda_d} lambda_sem={lambda_sem} tau_sem={tau_sem} "
                        f"lambda_ppl={lambda_ppl} tau_ppl={tau_ppl}")

    model.train()
    meters = {n: AverageMeter(n) for n in
              ["grpo_loss", "val_loss", "reward_mean", "reward_std", "parse_rate", "s_sem", "s_ppl", "penalty"]}
    gen_params, target_params = model.get_generator_params(), model.get_target_params()
    for q in gen_params + target_params:
        q.requires_grad = True
    tok = model.target_tokenizer
    total_steps = len(train_loader)
    val_iter = iter(val_loader)

    for batch_idx, batch in enumerate(train_loader):
        texts = batch["text"]
        B = len(texts)
        if batch_idx == 0:
            log_memory("batch_start")

        # 1. one prompt with all B samples; K candidate rewrites from G
        gen_input = model.build_gen_input(texts)
        gen_ids = gen_input["gen_input_ids"].to(device)
        gen_mask = gen_input["gen_attention_mask"].to(device)
        if batch_idx == 0:
            logger.info(f"  G input length: {gen_ids.size(1)} tokens")
        cand = model.sample_candidates(gen_ids, gen_mask, B=B, K=K, tau=tau, max_new_tokens=max_new_tokens)
        candidate_texts, candidate_ids = cand["candidate_texts"], cand["candidate_output_ids"]
        old_log_probs, parse_success = cand["old_log_probs"], cand["parse_success"]
        torch.cuda.empty_cache()
        if batch_idx == 0:
            log_memory("after_sampling")
            logger.info(f"  Candidate gen lengths: {[ids.size(0) for ids in candidate_ids]}")
        first_try = sum(bool(parse_success[k]) and all(candidate_texts[k]) for k in range(K))
        # samples G failed to rewrite keep their original stem (no retries while training)
        for k in range(K):
            candidate_texts[k] = [fill_with_original(s, t) for s, t in zip(candidate_texts[k], texts)]
        parse_rate = sum(parse_success) / K
        meters["parse_rate"].update(parse_rate)

        # 2. a clean validation batch for the degradation reward
        try:
            val_batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            val_batch = next(val_iter)
        val_ids, val_mask = val_batch["input_ids"].to(device), val_batch["attention_mask"].to(device)
        val_labels = build_sft_labels(val_ids, val_mask,
                                      get_answer_start_positions(val_batch["text"], val_ids, max_length, tok))

        # 3. reward of each candidate: T fine-tuned one step on it, from the same state
        t_snap = snapshot(target_params, target_optimizer)
        rewards = torch.zeros(K, device=device)
        val_losses = torch.zeros(K, device=device)
        clean_emb = sbert_embed(texts, device) if lambda_sem > 0 else None
        clean_ppl = gpt2_ppl(texts, device) if lambda_ppl > 0 else None
        sem_log, ppl_log, pen_log = torch.ones(K), torch.ones(K), torch.zeros(K)
        for q in gen_params:
            q.requires_grad = False

        for k in range(K):
            restore(target_params, target_optimizer, *t_snap)
            polluted = model.assemble_polluted_batch(candidate_texts[k], texts, max_length=max_length)
            p_ids, p_mask = polluted["input_ids"].to(device), polluted["attention_mask"].to(device)
            p_labels = build_sft_labels(p_ids, p_mask,
                                        get_answer_start_positions(polluted["texts"], p_ids, max_length, tok))
            for q in target_params:
                q.requires_grad = True
            target_optimizer.zero_grad()
            logits = model.forward_target(p_ids, p_mask)[:, :-1, :].contiguous().float()
            sft_loss = criterion(logits.view(-1, logits.size(-1)), p_labels.view(-1))
            sft_loss.backward()
            torch.nn.utils.clip_grad_norm_(target_params, max_norm=1.0)
            target_optimizer.step()
            del logits, sft_loss

            with torch.no_grad():
                for q in target_params:
                    q.requires_grad = False
                v_logits = model.forward_target(val_ids, val_mask)[:, :-1, :].contiguous().float()
                val_loss = criterion(v_logits.view(-1, v_logits.size(-1)), val_labels.view(-1))
            d_k = val_loss.item()
            val_losses[k] = d_k

            penalty = 0.0
            if shaping:
                if lambda_sem > 0:
                    s_sem = F.cosine_similarity(clean_emb, sbert_embed(polluted["texts"], device), dim=-1).mean().item()
                    sem_log[k] = s_sem
                    penalty += lambda_sem * max(0.0, tau_sem - s_sem)
                if lambda_ppl > 0:
                    pol_ppl = gpt2_ppl(polluted["texts"], device)
                    s_ppl = torch.minimum(torch.ones_like(clean_ppl), clean_ppl / pol_ppl).mean().item()
                    ppl_log[k] = s_ppl
                    penalty += lambda_ppl * max(0.0, tau_ppl - s_ppl)
                pen_log[k] = penalty
            rewards[k] = lambda_d * d_k - penalty
            del v_logits, val_loss, p_ids, p_mask, p_labels, polluted
            if batch_idx == 0 and k == 0:
                log_memory(f"after_candidate_k={k}")

        restore(target_params, target_optimizer, *t_snap)
        del t_snap
        torch.cuda.empty_cache()

        # 4. GRPO update of G on group-normalized rewards
        reward_mean, reward_std = rewards.mean(), rewards.std()
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)
        for q in gen_params:
            q.requires_grad = True
        for q in target_params:
            q.requires_grad = False
        gen_optimizer.zero_grad()
        new_log_probs = model.compute_log_probs(gen_ids, gen_mask, candidate_ids, tau=tau)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        grpo_loss = -torch.min(ratio * advantages.detach(), clipped * advantages.detach()).mean()
        grpo_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
        gen_optimizer.step()

        meters["grpo_loss"].update(grpo_loss.item())
        meters["val_loss"].update(val_losses.mean().item())
        meters["reward_mean"].update(reward_mean.item())
        meters["reward_std"].update(reward_std.item())
        if shaping:
            meters["s_sem"].update(sem_log.mean().item())
            meters["s_ppl"].update(ppl_log.mean().item())
            meters["penalty"].update(pen_log.mean().item())

        if batch_idx % int(cfg.run.logging_interval) == 0:
            grads = [q.grad.data.norm(2).item() ** 2 for q in gen_params if q.grad is not None]
            peak = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            fmt = lambda xs, d=4: ", ".join(f"{v:.{d}f}" for v in xs.tolist())
            logger.info(
                f"[Phase 1 GRPO] Epoch {epoch} [{batch_idx}/{total_steps}]\n"
                f"  Per-candidate val_loss=[{fmt(val_losses)}]\n"
                + (f"  Shaping: S_sem=[{fmt(sem_log, 3)}] S_ppl=[{fmt(ppl_log, 3)}] "
                   f"penalty=[{fmt(pen_log, 3)}]\n" if shaping else "")
                + f"  GRPO: grpo_loss={grpo_loss.item():.4f} reward_mean={reward_mean.item():.4f} "
                f"reward_std={reward_std.item():.4f}\n"
                f"  ratio_mean={ratio.mean().item():.4f} ratio_max={ratio.max().item():.4f}\n"
                f"  Parse_rate={parse_rate:.0%} ParseOK_first_try={first_try}/{K} "
                f"G_grad_norm={sum(grads) ** 0.5:.6f} G_grad_params={len(grads)}/{len(gen_params)}\n"
                f"  GPU peak memory: {peak:.1f} GB"
            )
            if batch_idx == 0:
                for b in range(min(2, B)):
                    orig = original_stem(texts[b]) or texts[b][:80]
                    logger.info(f"  Example {b}: orig='{orig[:100]}...' gen='{candidate_texts[0][b][:100]}...'")

    m = {n: v.avg for n, v in meters.items()}
    logger.info(
        f"[Phase 1 GRPO] Epoch {epoch} Finished.\n"
        f"  Avg GRPO_loss={m['grpo_loss']:.4f}\n  Avg Val_loss={m['val_loss']:.4f}\n"
        f"  Avg Reward: mean={m['reward_mean']:.4f} std={m['reward_std']:.4f}\n"
        f"  Avg Parse_rate={m['parse_rate']:.0%}"
        + (f"\n  Avg Shaping: S_sem={m['s_sem']:.4f} S_ppl={m['s_ppl']:.4f} penalty={m['penalty']:.4f}"
           if shaping else "")
    )


# =============================================================================
# Stage 2: protect the training set
# =============================================================================

def _vllm_stems(model, cfg, batches, tau, max_new_tokens, strict_parse):
    """Rewritten stems of every batch with vLLM: the same prompts, sampling,
    stop tokens, parsing and retries as the HF path, with G's LoRA served from
    a temporary adapter directory. Returns one list of stems per batch ("" = failed)."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompt = lambda texts: model.build_gen_input(texts)["gen_input_ids"][0].tolist()
    t0 = time.time()
    adapter_dir = tempfile.mkdtemp(prefix="peft_gen_")
    model.gen_model.save_pretrained(adapter_dir)
    gc.collect()
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    gpu_util = min(0.85, max(0.15, (free_b - 2 * 1024 ** 3) / total_b))
    first = [prompt(texts) for texts in batches]
    max_model_len = max(len(p) for p in first) + max_new_tokens + 16
    llm = LLM(model=cfg.model.generator, max_model_len=max_model_len, dtype="bfloat16",
              enable_lora=True, max_lora_rank=int(cfg.model.lora.rank),
              gpu_memory_utilization=gpu_util, seed=int(cfg.run.seed), trust_remote_code=True)
    lora = LoRARequest("generator", 1, adapter_dir)
    params = SamplingParams(temperature=max(tau, 0.3), top_p=1.0, max_tokens=max_new_tokens,
                            stop_token_ids=model.stop_token_ids(), skip_special_tokens=True)
    t1 = time.time()
    logger.info(f"[convert-vLLM] free={free_b / 1e9:.1f}GB -> gpu_memory_utilization={gpu_util:.3f}, "
                f"max_model_len={max_model_len}; start-up {t1 - t0:.1f}s")

    def run(prompts):
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    stems = [parse_stems(t, len(b))[0] for t, b in zip(run(first), batches)]
    n_prompts = len(first)
    if strict_parse:  # retry: twice the missing samples of a batch together, then one by one
        for attempt in range(3):
            groups = []
            for bi, st in enumerate(stems):
                missing = [k for k in range(len(st)) if not st[k]]
                if missing:
                    groups += [(bi, missing)] if attempt < 2 else [(bi, [k]) for k in missing]
            if not groups:
                break
            outs = run([prompt([batches[bi][k] for k in grp]) for bi, grp in groups])
            n_prompts += len(groups)
            for (bi, grp), t in zip(groups, outs):
                for k, stem in zip(grp, parse_stems(t, len(grp))[0]):
                    if stem:
                        stems[bi][k] = stem
    logger.info(f"[convert-vLLM] {n_prompts} prompt(s) generated in {time.time() - t1:.1f}s")

    del llm
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(adapter_dir, ignore_errors=True)
    return stems


def _hf_stems(model, texts, device, tau, max_new_tokens, strict_parse):
    """Rewritten stems of one batch with HF generate; with strict_parse, missing
    stems are regenerated twice as a smaller batch, then one sample at a time."""
    def generate(sub):
        enc = model.build_gen_input(sub)
        with torch.no_grad():
            out = model.sample_candidates(enc["gen_input_ids"].to(device), enc["gen_attention_mask"].to(device),
                                          B=len(sub), K=1, tau=max(tau, 0.3), max_new_tokens=max_new_tokens)
        return out["candidate_texts"][0]

    stems = generate(texts)
    if strict_parse:
        for attempt in range(3):
            missing = [b for b in range(len(texts)) if not stems[b]]
            if not missing:
                break
            for grp in ([missing] if attempt < 2 else [[b] for b in missing]):
                for b, stem in zip(grp, generate([texts[b] for b in grp])):
                    if stem:
                        stems[b] = stem
    return stems


def convert(model, dataset, device, tau, cfg, out_dir):
    """Rewrite every training sample with G and save the protected dataset."""
    p = cfg.protect
    engine, strict_parse = str(p.engine), bool(int(p.strict_parse))
    batch_size = int(p.convert_batch_size)
    max_new_tokens = generation_budget(cfg, batch_size)
    max_length = int(cfg.dataset.max_length)
    n = len(dataset)
    logger.info(f"Protecting {n} samples (engine={engine}, strict_parse={int(strict_parse)}, "
                f"B={batch_size}, max_new_tokens={max_new_tokens}) ...")
    model.eval()
    t_start = time.time()
    starts = range(0, n, batch_size)
    pre = None
    if engine == "vllm":
        pre = _vllm_stems(model, cfg, [[dataset[i]["text"] for i in range(s, min(s + batch_size, n))] for s in starts],
                          tau, max_new_tokens, strict_parse)

    ids, masks, orig_ids, records = [], [], [], []
    n_fallback = n_identical = 0
    for bi, s in enumerate(starts):
        items = [dataset[i] for i in range(s, min(s + batch_size, n))]
        texts = [it["text"] for it in items]
        stems = list(pre[bi]) if pre is not None else _hf_stems(model, texts, device, tau, max_new_tokens, strict_parse)
        failed = [b for b in range(len(texts)) if not stems[b]]
        if failed and strict_parse:
            logger.warning(f"  [strict_parse] batch {s}: {len(failed)} sample(s) still unparseable "
                           f"after retries — kept original stems")
        for b in failed:
            stems[b] = original_stem(texts[b])
            n_fallback += strict_parse or stem_tail_boundary(texts[b]) != -1
        polluted = model.assemble_polluted_batch(stems, texts, max_length=max_length)
        ids.append(polluted["input_ids"])
        masks.append(polluted["attention_mask"])
        orig_ids.append(model.target_tokenizer(texts, return_tensors="pt", padding="max_length",
                                               truncation=True, max_length=max_length)["input_ids"])
        for b, it in enumerate(items):
            orig = original_stem(texts[b])
            n_identical += stems[b].strip() == orig.strip()
            records.append({"idx": s + b, "question": it["question"], "answer": it["answer"],
                            "original_text": texts[b], "polluted_text": polluted["texts"][b],
                            "original_stem": orig, "generated_stem": stems[b]})
        if s % (batch_size * 10) == 0:
            logger.info(f"  Protected {min(s + batch_size, n)}/{n} samples")

    path = os.path.join(out_dir, "protected.pt")
    torch.save({
        "polluted_input_ids": torch.cat(ids), "attention_mask": torch.cat(masks),
        "original_input_ids": torch.cat(orig_ids),
        "original_texts": [r["original_text"] for r in records],
        "polluted_texts": [r["polluted_text"] for r in records],
        "questions": [r["question"] for r in records], "answers": [r["answer"] for r in records],
    }, path)
    with open(os.path.join(out_dir, "protected.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pct = lambda x: f"{x}/{n} ({100 * x / max(1, n):.1f}%)"
    logger.info(f"[Data quality] fallback_to_original: {pct(n_fallback)} | identical_stems (incl. fallbacks): "
                f"{pct(n_identical)} | effective_modification_rate: {pct(n - n_identical)}")
    logger.info(f"Protected dataset saved to {path} ({time.time() - t_start:.0f}s)")
    return path


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="a YAML file under configs/")
    args, overrides = ap.parse_known_args(argv)
    cfg = load_config(args.config, overrides)
    out_dir = output_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(os.path.join(out_dir, "protect.log"))
    if not torch.cuda.is_available():
        raise RuntimeError("TextCloak needs a CUDA GPU (CUDA_VISIBLE_DEVICES may hide it)")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    set_seed(int(cfg.run.seed))

    p, m = cfg.protect, cfg.model
    lora = (int(m.lora.rank), int(m.lora.alpha), float(m.lora.dropout))
    model = TextCloakModel(
        m.generator,
        m.surrogate,
        gen_lora=lora,
        target_lora=lora,
        dtype=m.dtype,
    ).to(device)
    tok = model.target_tokenizer

    train_set = build_dataset(cfg, "train", tok)
    valid_set = build_dataset(cfg, "valid", tok)
    workers = int(cfg.dataset.num_workers)
    train_loader = make_loader(train_set, p.batch_size, shuffle=True, drop_last=True, num_workers=workers)
    val_loader = make_loader(valid_set, p.batch_size, shuffle=True, drop_last=True, num_workers=workers)

    gen_params, target_params = model.get_generator_params(), model.get_target_params()
    gen_optimizer = torch.optim.AdamW(gen_params, lr=float(p.lr_generator), weight_decay=1e-4)
    target_optimizer = torch.optim.AdamW(target_params, lr=float(p.lr_surrogate),
                                         weight_decay=float(p.weight_decay_surrogate))
    n_epochs = int(p.n_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=n_epochs,
                                                           eta_min=float(p.lr_min_surrogate))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)

    logger.info(f"Config:\n{cfg}")
    logger.info(f"Total trainable parameters: {count_parameters(model):,} | "
                f"G LoRA: {sum(q.numel() for q in gen_params):,} | T LoRA: {sum(q.numel() for q in target_params):,}")
    logger.info(f"Data: {len(train_set)} train / {len(valid_set)} valid samples")
    log_memory("after_model_load")

    if p.get("generator_path"):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        loaded = set_peft_model_state_dict(model.gen_model, load_file(
            os.path.join(p.generator_path, "adapter_model.safetensors")))
        if loaded.unexpected_keys:
            raise RuntimeError(f"{p.generator_path}: {len(loaded.unexpected_keys)} keys do not match G, "
                               f"e.g. {loaded.unexpected_keys[:3]}")
        logger.info(f"G's LoRA loaded from {p.generator_path}")

    tau_start, tau_end = float(p.tau_start), float(p.tau_end)
    t0 = time.time()
    logger.info(f"Stage 1: training G with GRPO for {n_epochs} epoch(s) ...")
    for epoch in range(1, n_epochs + 1):
        tau = tau_start + (tau_end - tau_start) * (epoch - 1) / max(n_epochs - 1, 1)
        train_epoch(model, train_loader, val_loader, gen_optimizer, target_optimizer, epoch,
                    criterion, device, tau, cfg)
        scheduler.step()
    logger.info(f"Stage 1 done in {time.time() - t0:.0f}s")
    if int(p.save_generator):
        model.gen_model.save_pretrained(os.path.join(out_dir, "generator"))
        logger.info(f"G's LoRA saved to {os.path.join(out_dir, 'generator')}")

    logger.info("Stage 2: protecting the training set ...")
    convert(model, train_set, device, tau_end, cfg, out_dir)


if __name__ == "__main__":
    main()
