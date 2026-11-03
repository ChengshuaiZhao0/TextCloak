"""The rewriter G and the surrogate target T that TextCloak trains against.

G (a causal LM + LoRA) reads B formatted samples in one prompt and writes B
rewritten stems separated by "---". The stems are spliced back in front of the
unchanged choices and answer, re-tokenized with T's tokenizer, and T (another
causal LM + LoRA) is fine-tuned on them to score how much they hurt learning.
G and T share nothing: they talk through text only.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.text import parse_stems, stem_tail_boundary
from .lora import DTYPES

__all__ = ["TextCloakModel", "parse_stems", "SEPARATOR"]

SEPARATOR = "\n---\n"


def _lora(rank, alpha, dropout):
    return LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=rank,
                      lora_alpha=alpha, lora_dropout=dropout, target_modules="all-linear")


class TextCloakModel(nn.Module):
    """G (rewriter) and T (surrogate target), each a frozen base model with LoRA."""

    def __init__(self, gen_model_name: str, target_model_name: str,
                 gen_lora=(8, 32, 0.05), target_lora=(8, 32, 0.05),
                 dtype: str = "bf16"):
        super().__init__()
        torch_dtype = DTYPES[dtype]

        # --- G: the rewriter ---
        print(f"Loading generator model: {gen_model_name} (dtype={dtype})...")
        gen_base = AutoModelForCausalLM.from_pretrained(
            gen_model_name, torch_dtype=torch_dtype, dtype=torch_dtype, device_map={"": 0},
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        self.gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
        if self.gen_tokenizer.pad_token is None:
            self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token
            gen_base.config.pad_token_id = self.gen_tokenizer.pad_token_id
        if hasattr(gen_base, "generation_config"):
            gen_base.generation_config.do_sample = True
        if hasattr(gen_base.config, "enable_thinking"):  # Qwen3
            gen_base.config.enable_thinking = False
        self.gen_model = get_peft_model(gen_base, _lora(*gen_lora))
        self.gen_model.enable_input_require_grads()
        print("Generator LoRA:")
        self.gen_model.print_trainable_parameters()

        # --- T: the surrogate target ---
        print(f"Loading target model: {target_model_name} (dtype={dtype})...")
        target_base = AutoModelForCausalLM.from_pretrained(
            target_model_name, torch_dtype=torch_dtype, dtype=torch_dtype, device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
            target_base.config.pad_token_id = self.target_tokenizer.pad_token_id
        self.target_model = get_peft_model(target_base, _lora(*target_lora))
        print("Target LoRA:")
        self.target_model.print_trainable_parameters()

    def get_generator_params(self):
        return [p for n, p in self.gen_model.named_parameters() if "lora_" in n]

    def get_target_params(self):
        return [p for n, p in self.target_model.named_parameters() if "lora_" in n]

    def stop_token_ids(self):
        tok = self.gen_tokenizer
        ids = [tok.eos_token_id]
        for name in ["<think>", "</think>", "<|endoftext|>"]:
            tid = tok.convert_tokens_to_ids(name)
            if tid is not None and tid != tok.unk_token_id:
                ids.append(tid)
        return ids

    # ------------------------------------------------------------------ G input

    def build_gen_input(self, texts: List[str]) -> Dict:
        """One prompt asking G to rewrite the stems of all B `texts`.

        G sees each full sample (stem, choices, answer) and is asked for B
        rewritten stems separated by "---".
        Returns `gen_input_ids` / `gen_attention_mask` of shape [1, L].
        """
        B = len(texts)
        # Kept verbatim (typos included): it is the prompt the reported results used.
        prefix = (
            f"Below are {B} multiple-choice questions. "
            f"For each question, rewrite ONLY the question stem "
            f"(the text between \"Question:\" and the first answer choice). "
            f"The purpose of these rewrites is to make further SFT fine-tuning on these questions less effective and degrade downsteam tasks for target LLMs. "
            f"Use a variety of rewriting strategies, including but not limited to: paraphrasing, synonym substitution, syntactic restructuring, active/passive voice changes, sentence splitting or merging, changing discourse markers, changing specificity while preserving meaning, reordering clauses, converting direct questions to indirect forms, nominalization or verbalization, and light stylistic variation. "
            f"You may also vary tone or phrasing"
            f"Output exactly {B} rewritten stems, one per question, "
            f"separated by \"---\". Do NOT include answer choices or the answer.\n\n"
        )
        body = "\n\n".join(f"[Q{i + 1}] {text}" for i, text in enumerate(texts))
        suffix = "\n\nRewritten stems:\n"
        enc = self.gen_tokenizer(prefix + body + suffix, return_tensors="pt", add_special_tokens=True)
        return {"gen_input_ids": enc["input_ids"], "gen_attention_mask": enc["attention_mask"], "B": B}

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def sample_candidates(self, gen_input_ids, gen_attention_mask, B: int, K: int = 8,
                          tau: float = 0.7, max_new_tokens: int = 512) -> Dict:
        """Sample K outputs of G for one prompt (one batched generate call).

        Returns candidate_texts [K][B] stems, candidate_output_ids (K token
        tensors), old_log_probs [K] (teacher-forced, no grad) and
        parse_success [K] (whether the output split into B stems).
        """
        input_len = gen_input_ids.size(1)
        stop_ids = self.stop_token_ids()
        gen_output = self.gen_model.generate(
            input_ids=gen_input_ids.expand(K, -1), attention_mask=gen_attention_mask.expand(K, -1),
            max_new_tokens=max_new_tokens, do_sample=True, temperature=tau, top_p=1.0,
            eos_token_id=stop_ids, pad_token_id=self.gen_tokenizer.pad_token_id,
        )
        candidate_texts, candidate_output_ids, parse_success = [], [], []
        for k in range(K):
            gen_ids = gen_output[k, input_len:]
            keep = gen_ids != self.gen_tokenizer.pad_token_id
            for sid in stop_ids:
                keep = keep & (gen_ids != sid)
            gen_ids = gen_ids[keep]
            candidate_output_ids.append(gen_ids)
            stems, ok = parse_stems(self.gen_tokenizer.decode(gen_ids.cpu(), skip_special_tokens=True), B)
            candidate_texts.append(stems)
            parse_success.append(ok)
        del gen_output
        old_log_probs = self._log_probs_no_grad(gen_input_ids, candidate_output_ids, tau)
        return {"candidate_texts": candidate_texts, "candidate_output_ids": candidate_output_ids,
                "old_log_probs": old_log_probs, "parse_success": parse_success}

    def _sequence_log_prob(self, gen_input_ids, output_ids, tau):
        """log P_G(output | prompt) at temperature tau, by teacher forcing."""
        input_len, g_len = gen_input_ids.size(1), output_ids.size(0)
        tf_input = torch.cat([gen_input_ids.squeeze(0), output_ids], dim=0).unsqueeze(0)
        tf_mask = torch.ones(1, tf_input.size(1), device=tf_input.device, dtype=torch.long)
        logits = self.gen_model(input_ids=tf_input, attention_mask=tf_mask, return_dict=True).logits.float()
        log_probs = F.log_softmax(logits[0, input_len - 1: input_len - 1 + g_len, :] / tau, dim=-1)
        return log_probs.gather(-1, output_ids.unsqueeze(-1)).squeeze(-1).sum()

    @torch.no_grad()
    def _log_probs_no_grad(self, gen_input_ids, candidate_output_ids, tau):
        device = gen_input_ids.device
        return torch.stack([
            self._sequence_log_prob(gen_input_ids, ids, tau) if ids.size(0) else torch.tensor(0.0, device=device)
            for ids in candidate_output_ids
        ])

    def compute_log_probs(self, gen_input_ids, gen_attention_mask, candidate_output_ids, tau: float = 0.7):
        """[K] log probs of the candidates WITH gradient through G (one at a time,
        gradient checkpointing on, to fit long prompts in memory)."""
        device = gen_input_ids.device
        self.gen_model.gradient_checkpointing_enable()
        out = []
        for ids in candidate_output_ids:
            if ids.size(0) == 0:
                out.append(torch.tensor(0.0, device=device, requires_grad=True))
                continue
            out.append(self._sequence_log_prob(gen_input_ids, ids, tau))
            torch.cuda.empty_cache()
        self.gen_model.gradient_checkpointing_disable()
        return torch.stack(out)

    # ------------------------------------------------------------------ T side

    def assemble_polluted_batch(self, candidate_stems: List[str], original_texts: List[str],
                                max_length: int = 256) -> Dict:
        """Splice the rewritten stems into the samples and tokenize them for T.

        "Question: {stem}\\n(A) ...\\nAnswer: X" -> "Question: {new stem}\\n(A) ...\\nAnswer: X".
        An empty or unparsable stem keeps the original sample.
        """
        texts = []
        for orig, stem in zip(original_texts, candidate_stems):
            marker = stem_tail_boundary(orig)
            texts.append(f"Question: {stem}{orig[marker:]}" if (marker != -1 and stem) else orig)
        enc = self.target_tokenizer(texts, return_tensors="pt", padding="max_length",
                                    truncation=True, max_length=max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "texts": texts}

    def forward_target(self, input_ids, attention_mask):
        return self.target_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits

    def state_dict(self, *args, **kwargs):
        return get_peft_model_state_dict(self.target_model)

    def load_state_dict(self, state_dict, strict=True):
        return set_peft_model_state_dict(self.target_model, state_dict)
