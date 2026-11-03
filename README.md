# TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples

[![Paper](https://img.shields.io/badge/Paper-arXiv:2607.28862-%23B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.28862)
[![Code](https://img.shields.io/badge/Code-GitHub-%23F2806B.svg)](https://github.com/ChengshuaiZhao0/TextCloak)
[![Python](https://img.shields.io/badge/Python-3.10%2B-%233776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Datasets%20%26%20Models-%23FFD21E.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-%2343B02A.svg)](LICENSE)

This repository contains the official Python implementation of **TextCloak**, the framework described in the paper **[TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples](https://arxiv.org/abs/2607.28862)**.

## 🔭 Introduction

TextCloak is an RL-driven data synthesis framework that safeguards textual data against unauthorized LLM fine-tuning. A UE generative policy rewrites clean text into unlearnable examples (UEs) that preserve semantic fidelity and linguistic naturalness, so the text stays useful to people while LLMs fine-tuned on it derive little benefit. The policy is optimized with GRPO-UE, which simulates the attacker's fine-tuning on a surrogate LLM and rewards downstream degradation via group-relative policy optimization.

## ✨ Highlights

⭐ **Problem formulation.** A threat model for safeguarding textual data in the era of LLMs, with the defense cast as a constrained bi-level optimization problem.

⭐ **Novel framework.** The first unlearnable examples framework designed for LLMs, extending data protection from text classification to open-ended generation while preserving semantic fidelity and linguistic naturalness.

⭐ **Optimization mechanism.** GRPO-UE simulates unauthorized LLM fine-tuning and measures protection through downstream degradation, guiding the generative policy toward effective unlearnable patterns.

⭐ **Comprehensive evaluation.** Effectiveness, transferability, robustness and stealthiness across various datasets and state-of-the-art LLMs.

## 🗂️ Repository Layout

```text
TextCloak/
│
├── configs/                  one YAML per dataset, on top of _base.yaml
│   ├── _base.yaml            shared defaults (models, GRPO, rewards, fine-tuning)
│   ├── arc.yaml              ARC-Challenge
│   └── medqa.yaml            MedQA-USMLE
│
├── scripts/
│   ├── download.py           fetch the datasets and models from the Hugging Face Hub
│   ├── run.sh                end-to-end: protect, fine-tune on clean and on protected data
│   └── summarize.py          clean vs protected test accuracy of a run
│
├── tests/
│   └── test_text.py          CPU-only checks (configs, text boundaries, answer extraction)
│
└── textcloak/                core package
    ├── protect.py            Stages 1-2: train the rewriter with GRPO, then protect the data
    ├── finetune.py           Stage 3: fine-tune a model on clean or protected data, test it
    ├── rewards.py            SBERT similarity and GPT-2 perplexity scorers
    ├── config.py             YAML configs with _base_ inheritance and --key=value overrides
    ├── utils.py              logging, meters, seeding
    ├── models/
    │   ├── cloak.py          the rewriter G and the surrogate T (prompting, sampling, splicing)
    │   └── lora.py           causal LM + LoRA, the model fine-tuned in Stage 3
    ├── data/
    │   ├── datasets.py       ARC and MedQA, and their train/valid/test splits
    │   ├── protected.py      reads a protected dataset (.pt)
    │   └── text.py           where the stem, the choices and the answer of a sample are
    └── eval/
        ├── evaluate.py       greedy generation (vLLM or HF) and scoring
        └── answers.py        answer-letter extraction

Generated at runtime (outputs/<dataset>/seed<seed>/):
    protect.log  generator/  protected.pt  protected.jsonl
    finetune_{clean,protected}.log  results_{clean,protected}.json
```

## 📦 Installation

Requirements: Python ≥ 3.10, Linux with glibc ≥ 2.35 (e.g. Ubuntu 22.04; the vLLM wheels need it) and a C compiler (`gcc`, used by vLLM's `torch.compile`).

### With uv (recommended)

```console
$ git clone https://github.com/ChengshuaiZhao0/TextCloak.git
$ cd TextCloak
$ uv venv --python 3.12
$ uv pip install -e ".[dev]"
$ source .venv/bin/activate
```

<details>
<summary>With pip (alternative)</summary>

```console
$ python -m venv .venv && source .venv/bin/activate
$ pip install -e ".[dev]"
```

</details>

The dependencies are pinned in [pyproject.toml](pyproject.toml) to the versions the reported results were produced with (PyTorch 2.11, vLLM 0.20.2, Transformers 5.13, PEFT 0.18.1).

## 📥 Data and Models

| Config | Dataset | Task |
| --- | --- | --- |
| [`arc`](configs/arc.yaml) | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge) | 4-option science QA |
| [`medqa`](configs/medqa.yaml) | [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | 4-option medical QA |

| Role | Model |
| --- | --- |
| rewriter | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| surrogate, and the model fine-tuned for evaluation | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| semantic similarity | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| perplexity | [gpt2](https://huggingface.co/gpt2) |

## 🚀 Quickstart

Run the whole pipeline on one dataset:

```console
$ bash scripts/run.sh arc                      # or: bash scripts/run.sh medqa
```

Or run the stages one by one:

```console
# Stages 1-2: train the rewriter with GRPO, then rewrite the training set
$ python -m textcloak.protect configs/arc.yaml

# Stage 3: fine-tune the target model on clean and on protected data, then test it
$ python -m textcloak.finetune configs/arc.yaml --data clean
$ python -m textcloak.finetune configs/arc.yaml --data protected

# Compare the test accuracy of the two runs
$ python scripts/summarize.py configs/arc.yaml
```

Any config key can be overridden as `--section.key=value`, and the same arguments work for every command above:

```console
# another seed, or turn off a reward term
$ bash scripts/run.sh arc --run.seed=43 --protect.lambda_sem=0

# a quick smoke run on a few samples
$ bash scripts/run.sh arc --dataset.limit_train=16 --dataset.limit_valid=16 \
      --dataset.limit_test=32 --protect.n_epochs=1 --finetune.n_epochs=1

# protect again with a saved rewriter, skipping training
$ python -m textcloak.protect configs/arc.yaml --protect.n_epochs=0 \
      --protect.generator_path=outputs/arc/seed42/generator --run.output_dir=outputs/arc/again
```

Outputs go to `outputs/<dataset>/seed<seed>/`: the protected data (`protected.jsonl`, `protected.pt`), the rewriter's LoRA adapter (`generator/`), the logs, and `results_{clean,protected}.json`. See [configs/_base.yaml](configs/_base.yaml) for all settings.

## ✅ Tests

The lightweight checks run on CPU and download nothing:

```console
$ pytest -q
```

## 📝 Citation

If our repo helped you out, we'd love it if you gave us a citation! Thanks for supporting our work!

```tex
@misc{zhao2026textcloak,
  title={TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples},
  author={Chengshuai Zhao and Pingchuan Ma and Dawei Li and Bohan Jiang and Zhiyuan Yu and Zhen Tan and Huan Liu},
  year={2026},
  eprint={2607.28862},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2607.28862}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
