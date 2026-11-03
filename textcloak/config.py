"""YAML configs with `_base_` inheritance and dotted command-line overrides.

    cfg = load_config("configs/arc.yaml", ["--run.seed=43", "--protect.lambda_sem=0"])
    cfg.protect.group_size     # attribute access
    cfg.get("protect.max_new_tokens", 0)
"""

import os
from ast import literal_eval

import yaml

__all__ = ["Config", "load_config", "output_dir"]


class Config(dict):
    """A dict with attribute access; nested dicts are Configs too."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, dotted, default=None):
        node = self
        for k in dotted.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def set(self, dotted, value):
        node = self
        *parents, last = dotted.split(".")
        for k in parents:
            node = node.setdefault(k, Config())
        node[last] = value

    def merge(self, other):
        for k, v in other.items():
            if isinstance(v, dict):
                if not isinstance(self.get(k), Config):
                    self[k] = Config()
                self[k].merge(v)
            else:
                self[k] = v
        return self

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}

    def __str__(self):
        return yaml.safe_dump(self.to_dict(), sort_keys=False).rstrip()


def _read(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    cfg = Config()
    if base:
        cfg.merge(_read(os.path.join(os.path.dirname(path), base)))
    return cfg.merge(data)


def _parse(value):
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(path, overrides=()):
    """Read `path` (and its `_base_` chain), then apply `--a.b=value` overrides."""
    cfg = _read(path)
    overrides = list(overrides)
    while overrides:
        opt = overrides.pop(0)
        if not opt.startswith("--"):
            raise ValueError(f"expected --key=value, got {opt!r}")
        key, sep, value = opt[2:].partition("=")
        if not sep:
            if not overrides:
                raise ValueError(f"{opt} needs a value")
            value = overrides.pop(0)
        cfg.set(key, _parse(value))
    return cfg


def output_dir(cfg):
    """`run.output_dir` with {dataset} and {seed} filled in."""
    return str(cfg.run.output_dir).format(dataset=cfg.dataset.name, seed=cfg.run.seed)
