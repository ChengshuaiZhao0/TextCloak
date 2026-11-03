from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["LoRALM", "DTYPES"]

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class LoRALM(nn.Module):
    """A causal LM with LoRA adapters: the model fine-tuned on (protected) data.

    `state_dict()` / `load_state_dict()` cover the adapter weights only, and
    `self.model` is the PEFT model (its adapter is what vLLM serves in evaluation).
    """

    def __init__(self, model_name: str, lora_rank: int = 8, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, target_modules="all-linear", dtype: str = "bf16"):
        super().__init__()
        print(f"Loading base model: {model_name} (dtype={dtype})...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPES[dtype], device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:  # e.g. Llama-3
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = get_peft_model(self.base_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_rank,
            lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules,
        ))
        self.model.print_trainable_parameters()

    def get_text_tokenizer(self):
        return self.tokenizer

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        return {"logits": out.logits}

    def state_dict(self, *args, **kwargs):
        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state_dict, strict=True):
        return set_peft_model_state_dict(self.model, state_dict)
