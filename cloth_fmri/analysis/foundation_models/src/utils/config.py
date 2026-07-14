# src/utils/config.py
import os
import copy
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _set_by_path(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    cur = cfg

    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]

    cur[keys[-1]] = value


def _parse_value(s):
    sl = s.lower()

    if sl in ("true", "false"):
        return sl == "true"

    if sl in ("none", "null"):
        return None

    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


MODEL_MAP = {
    "videomae": {
        "backbone": "configs/backbone/videomae_base.yaml",
        "output_dir": "checkpoints/videomae",
    },
    "vivit": {
        "backbone": "configs/backbone/vivit_b_16x2.yaml",
        "output_dir": "checkpoints/vivit",
    },
    "vjepa2_hf": {
        "backbone": "configs/backbone/vjepa_huge.yaml",
        "output_dir": "checkpoints/vjepa2_hf",
    },
}


def _resolve_includes(cfg, include_keys=("backbone",)):
    cfg = copy.deepcopy(cfg)

    for key in include_keys:
        include_path = cfg.get(key)

        if include_path is None:
            continue

        if not isinstance(include_path, str):
            raise ValueError(
                f"Expected cfg['{key}'] to be a path string before resolving, "
                f"but got {type(include_path)}"
            )

        included = load_yaml(include_path)

        if key not in included:
            raise ValueError(f"{include_path} must contain top-level key '{key}'")

        cfg[key] = included[key]

    return cfg


def load_config(main_path, overrides=None):

    overrides = overrides or []

    # ----------------------------------
    # 1) load main config
    # ----------------------------------
    cfg = load_yaml(main_path)
    cfg = copy.deepcopy(cfg)

    # ----------------------------------
    # 2) apply command-line overrides FIRST
    # ----------------------------------
    for item in overrides:

        if "=" not in item:
            raise ValueError(f"Bad override: {item}. Expected key=value")

        k, v = item.split("=", 1)
        _set_by_path(cfg, k, _parse_value(v))

    # ----------------------------------
    # 3) resolve model_name -> backbone + output_dir
    # ----------------------------------
    model_name = cfg.get("model_name")

    if model_name is not None:

        if model_name not in MODEL_MAP:
            raise ValueError(f"Unknown model_name: {model_name}")

        model_cfg = MODEL_MAP[model_name]

        cfg["backbone"] = model_cfg["backbone"]

        cfg.setdefault("train", {})
        cfg["train"]["output_dir"] = model_cfg["output_dir"]

    # ----------------------------------
    # 4) resolve include files
    # ----------------------------------
    cfg = _resolve_includes(cfg, include_keys=("backbone",))

    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)