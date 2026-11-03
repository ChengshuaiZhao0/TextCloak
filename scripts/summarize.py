"""Print the clean vs protected test accuracy of one run.

    python scripts/summarize.py configs/arc.yaml [--run.seed=43 ...]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from textcloak.config import load_config, output_dir  # noqa: E402


def main():
    cfg = load_config(sys.argv[1], sys.argv[2:])
    out = output_dir(cfg)
    rows = {}
    for tag in ("clean", "protected"):
        path = os.path.join(out, f"results_{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows[tag] = json.load(f)
    if not rows:
        sys.exit(f"no results in {out}")
    first = next(iter(rows.values()))
    print(f"{cfg.dataset.name} (seed {cfg.run.seed}, {cfg.model.target}) -> {out}")
    print(f"  {'zero-shot':<23} {first['zero_shot']:.4f}")
    for tag, r in rows.items():
        print(f"  fine-tuned on {tag:<9} {r['test_acc']:.4f}")
    if len(rows) == 2:
        print(f"  {'protected - clean':<23} {rows['protected']['test_acc'] - rows['clean']['test_acc']:+.4f}")


if __name__ == "__main__":
    main()
