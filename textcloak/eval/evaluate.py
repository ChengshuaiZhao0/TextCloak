"""Greedy multiple-choice evaluation with vLLM or Hugging Face generation."""

import gc
import shutil
import tempfile

import torch

from ..utils import logger
from .answers import ANSWER_EXTRACTORS

__all__ = ["evaluate_vllm", "evaluate_hf"]


def _prompts_and_answers(loader):
    prompts, answers = [], []
    for batch in loader:
        prompts += [f"Question: {question}\nAnswer: " for question in batch["question"]]
        answers += list(batch["answer"])
    return prompts, answers


def _score(generated, answers, epoch, tag, dataset_name):
    pred_fn, ref_fn = ANSWER_EXTRACTORS[dataset_name]
    correct = 0
    for text, reference in zip(generated, answers):
        prediction, expected = pred_fn(text), ref_fn(reference)
        correct += prediction is not None and expected is not None and prediction == expected
    accuracy = correct / len(answers) if answers else 0.0
    logger.info(
        f"[Eval-{tag}] Epoch {epoch} | Test Acc (exact match): "
        f"{accuracy:.4f} ({correct}/{len(answers)})"
    )
    return accuracy


def evaluate_vllm(model, loader, epoch, cfg, lora_rank):
    """Serve the base model and the current LoRA adapter with vLLM."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model.eval()
    max_length = int(cfg.dataset.max_length)
    prompts, answers = _prompts_and_answers(loader)

    adapter_dir = tempfile.mkdtemp(prefix="peft_eval_")
    model.model.save_pretrained(adapter_dir)
    gc.collect()
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    gpu_util = min(0.45, max(0.15, (free_b - 2 * 1024 ** 3) / total_b))
    logger.info(
        f"[Eval-vLLM] free={free_b / 1e9:.1f}GB total={total_b / 1e9:.1f}GB "
        f"-> gpu_memory_utilization={gpu_util:.3f}"
    )
    params = SamplingParams(temperature=0, max_tokens=32, stop=["Question:"])
    llm = LLM(
        model=cfg.model.target,
        max_model_len=max(max_length * 2, 2048),
        enforce_eager=True,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
    )
    outputs = llm.generate(
        prompts,
        params,
        lora_request=LoRARequest("adapter", 1, adapter_dir),
    )
    generated = [output.outputs[0].text for output in outputs]

    del llm
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()
    shutil.rmtree(adapter_dir, ignore_errors=True)
    return _score(generated, answers, epoch, "vLLM", cfg.dataset.name)


def evaluate_hf(model, loader, epoch, cfg, tokenizer, batch_size=4):
    """Evaluate with Transformers generate() when vLLM is not available."""
    model.eval()
    max_length = int(cfg.dataset.max_length)
    prompts, answers = _prompts_and_answers(loader)
    generated = []
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.model.device)
        with torch.no_grad():
            output_ids = model.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["Question:"],
                tokenizer=tokenizer,
            )
        for i in range(len(chunk)):
            text = tokenizer.decode(
                output_ids[i, encoded["input_ids"].size(1):],
                skip_special_tokens=True,
            )
            generated.append(text.split("Question:")[0].strip())
    tokenizer.padding_side = padding_side
    return _score(generated, answers, epoch, "HF", cfg.dataset.name)
