# src/modeling/factory.py
import torch.nn as nn

from .backbones.videomae import VideoMAEBackbone
from .backbones.vivit import ViViTBackbone
from .backbones.vjepa2_hf import VJEPA2HFBackbone

def build_backbone(cfg) -> nn.Module:
    bcfg = cfg["backbone"]
    name = bcfg["name"].lower()

    print(f"@@@ Building backbone for model={name}...")

    if name == "videomae":
        return VideoMAEBackbone(
            hf_model_name=bcfg["hf_model_name"],
            freeze=bool(bcfg.get("freeze", True)),
        )

    if name == "vivit":
        return ViViTBackbone(
            hf_model_name=bcfg["hf_model_name"],
            freeze=bool(bcfg.get("freeze", True)),
        )

    if name == "vjepa2_hf":
        from .backbones.vjepa2_hf import VJEPA2HFBackbone

        return VJEPA2HFBackbone(
            hf_repo=bcfg["hf_repo"],
            freeze=bool(bcfg.get("freeze", True)),
        )

    raise ValueError(f"Unknown backbone: {name}")