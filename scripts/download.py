"""Download the datasets and models from the Hugging Face Hub ahead of time.

    python scripts/download.py                  # both datasets + all models
    python scripts/download.py --datasets arc medqa --no-models

Everything goes to the Hugging Face cache ($HF_HOME, default ~/.cache/huggingface),
where the training code finds it. Downloading first is optional (the code
downloads on first use), but it lets a GPU job run offline (HF_HUB_OFFLINE=1).
meta-llama/Meta-Llama-3-8B is gated: accept its license on its Hugging Face
page, then log in (`hf auth login`) or set HF_TOKEN.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download  # noqa: E402

from textcloak.data import DATASETS  # noqa: E402
from textcloak.rewards import GPT2_NAME, SBERT_NAME  # noqa: E402

MODELS = ["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen3-8B", SBERT_NAME, GPT2_NAME]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--no-models", action="store_true", help="only the datasets")
    args = ap.parse_args()

    for name in args.datasets:
        for split in ("train", "valid", "test"):
            ds = DATASETS[name](split)
            print(f"{name:<10} {split:<5} {len(ds):>6} samples")
    if not args.no_models:
        for repo in MODELS:
            # safetensors weights only (Llama-3 also ships the original .pth checkpoint)
            path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                           "tokenizer*", "*.py", "modules.json", "1_Pooling/*"])
            print(f"{repo:<45} {path}")


if __name__ == "__main__":
    main()
