import torch
import torch.nn as nn
from transformers import VivitModel


class ViViTBackbone(nn.Module):
    def __init__(self, hf_model_name: str, freeze: bool = True):
        super().__init__()
        self.model = VivitModel.from_pretrained(hf_model_name)
        self.embed_dim = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def _forward_frozen(self, x):
        out = self.model(pixel_values=x, return_dict=True)
        return out.last_hidden_state  # (B, N, D)

    def forward(self, x):
        if any(p.requires_grad for p in self.model.parameters()):
            out = self.model(pixel_values=x, return_dict=True)
            return out.last_hidden_state
        return self._forward_frozen(x)
