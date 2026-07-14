# src/modeling/backbones/vjepa2_hf.py
import torch
import torch.nn as nn

from transformers import AutoModel


class VJEPA2HFBackbone(nn.Module):

    def __init__(self, hf_repo, freeze=True):
        super().__init__()

        print(f"Loading VJEPA2 HF backbone: {hf_repo}")

        self.model = AutoModel.from_pretrained(
            hf_repo,
            trust_remote_code=True,
        )

        self.embed_dim = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, x):

        # x shape must be:
        # (B, T, C, H, W)
        outputs = self.model(
            pixel_values_videos=x
        )

        return outputs.last_hidden_state