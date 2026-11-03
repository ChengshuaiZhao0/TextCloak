"""Stage 3 of TextCloak: fine-tune a model on clean or protected data and test it.

    python -m textcloak.finetune configs/arc.yaml --data clean
    python -m textcloak.finetune configs/arc.yaml --data protected      # <output_dir>/protected.pt
    python -m textcloak.finetune configs/arc.yaml --data path/to/protected.pt

LoRA fine-tuning of `model.target` on the training set (only the answer tokens
are supervised), then greedy evaluation on the clean test set. The test
accuracy before fine-tuning is the zero-shot baseline. The log goes to
<output_dir>/finetune_<data>.log and the numbers to <output_dir>/results_<data>.json.
Protection works when fine-tuning on the protected data gains little or loses
accuracy, compared with fine-tuning on the clean data.
"""

import argparse
import gc
import json
import os
import time

import torch

from .config import load_config, output_dir
from .data import ProtectedDataset, build_dataset, get_answer_start_positions, make_loader
from .eval import evaluate_hf, evaluate_vllm
from .models.lora import LoRALM
from .rewards import report_dataset_quality
from .utils import AverageMeter, count_parameters, log_memory, logger, set_seed, setup_logging


def train_epoch(model, loader, optimizer, epoch, criterion, device, tokenizer, cfg):
    """One epoch of causal-LM fine-tuning with the loss on the answer tokens only."""
    model.train()
    loss_m, acc_m, grad_m = AverageMeter("loss"), AverageMeter("acc"), AverageMeter("grad_norm")
    max_length = int(cfg.dataset.max_length)
    for batch_idx, batch in enumerate(loader):
        input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous().masked_fill(attention_mask[..., 1:].contiguous() == 0, -100)
        starts = get_answer_start_positions(batch.get("polluted_text", batch["text"]), input_ids, max_length, tokenizer)
        starts = torch.tensor(starts, device=device)
        pos = torch.arange(shift_labels.size(1), device=device)
        shift_labels = shift_labels.masked_fill(pos.unsqueeze(0) < (starts - 1).clamp(min=0).unsqueeze(1), -100)
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_m.update(loss.item())
        grad_m.update(float(grad_norm))
        with torch.no_grad():
            mask = shift_labels != -100
            acc_m.update((shift_logits.argmax(-1)[mask] == shift_labels[mask]).float().mean().item()
                         if mask.any() else 0.0)
        if batch_idx % int(cfg.finetune.logging_interval) == 0:
            logger.info(f"[Phase 2 - SFT] Epoch {epoch} [{batch_idx}/{len(loader)}] "
                        f"Loss: {loss_m.val:.4f} ({loss_m.avg:.4f}) | Train Acc: {acc_m.val:.4f} | "
                        f"GradNorm: {grad_m.val:.4f} ({grad_m.avg:.4f}) | lr: {optimizer.param_groups[0]['lr']:.2e}")
    logger.info(f"[Phase 2 - SFT] Epoch {epoch} Done. Avg Loss: {loss_m.avg:.4f} | "
                f"Avg Train Acc: {acc_m.avg:.4f} | Avg GradNorm: {grad_m.avg:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="a YAML file under configs/")
    ap.add_argument("--data", default="protected",
                    help="clean | protected (<output_dir>/protected.pt) | a .pt file from textcloak.protect")
    args, overrides = ap.parse_known_args(argv)
    cfg = load_config(args.config, overrides)
    out_dir = output_dir(cfg)
    tag = args.data if args.data in ("clean", "protected") else os.path.splitext(os.path.basename(args.data))[0]
    data_path = None if args.data == "clean" else (
        os.path.join(out_dir, "protected.pt") if args.data == "protected" else args.data)
    if data_path and not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path}: run `python -m textcloak.protect {args.config}` first")
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(os.path.join(out_dir, f"finetune_{tag}.log"))
    if not torch.cuda.is_available():
        raise RuntimeError("TextCloak needs a CUDA GPU (CUDA_VISIBLE_DEVICES may hide it)")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    set_seed(int(cfg.run.seed))

    f, m = cfg.finetune, cfg.model
    logger.info(f"Fine-tuning {m.target} on {data_path or 'clean data'}")
    logger.info(f"Config:\n{cfg}")
    model = LoRALM(m.target, lora_rank=int(m.lora.rank), lora_alpha=int(m.lora.alpha),
                   lora_dropout=float(m.lora.dropout), dtype=m.dtype).to(device)
    tok = model.get_text_tokenizer()
    tok.padding_side = "right"  # the answer mask assumes right padding
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    log_memory("after_model_load")

    workers, bs = int(cfg.dataset.num_workers), int(f.batch_size)
    if data_path is None:
        train_set = build_dataset(cfg, "train", tok)
    else:
        train_set = ProtectedDataset(data_path, tokenizer=tok, max_length=int(cfg.dataset.max_length))
    train_loader = make_loader(train_set, bs, shuffle=True, drop_last=True, num_workers=workers)
    test_loader = make_loader(build_dataset(cfg, "test", tok), bs, shuffle=False, drop_last=False,
                              num_workers=workers)
    logger.info(f"Training samples: {len(train_set)} | test samples: {len(test_loader.dataset)}")
    quality = report_dataset_quality(train_set, device) if data_path else None

    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                  lr=float(f.lr), weight_decay=float(f.weight_decay))
    n_epochs = int(f.n_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=float(f.lr_min))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100).to(device)

    lora_rank = int(m.lora.rank)
    evaluate = ((lambda ep: evaluate_hf(model, test_loader, ep, cfg, tok)) if f.eval_engine == "hf"
                else (lambda ep: evaluate_vllm(model, test_loader, ep, cfg, lora_rank)))

    logger.info("Before fine-tuning: zero-shot test accuracy")
    zero_shot = evaluate(0)
    logger.info(f"[Pre-SFT] Test acc = {zero_shot:.4f}")

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, criterion, device, tok, cfg)
        scheduler.step()
    train_time = time.time() - t0

    del optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    test_acc = evaluate(n_epochs)
    logger.info(f"[Final] Test acc = {test_acc:.4f}")

    results = {"dataset": cfg.dataset.name, "data": data_path or "clean", "target": m.target,
               "seed": int(cfg.run.seed), "train_samples": len(train_set), "test_samples": len(test_loader.dataset),
               "zero_shot": zero_shot, "test_acc": test_acc, "train_seconds": round(train_time),
               "data_quality": quality}
    with open(os.path.join(out_dir, f"results_{tag}.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    logger.info(f"Results: {json.dumps(results)}")


if __name__ == "__main__":
    main()
