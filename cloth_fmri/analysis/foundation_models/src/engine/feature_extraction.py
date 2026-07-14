import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def extract_features(loader, backbone, device, pool="mean"):
    backbone.eval()

    all_feats = []
    all_targets = []

    for x, y in tqdm(loader, desc="Extracting features"):
        x = x.to(device, non_blocking=True)

        feat = backbone(x)  # (B, N, D)

        if pool == "cls":
            feat = feat[:, 0]
        elif pool == "mean":
            #feat = feat.mean(dim=1)
            feat = feat[:,1:].mean(dim=1)
        else:
            raise ValueError(pool)

        all_feats.append(feat.cpu().numpy())
        all_targets.append(y.numpy())

    X = np.concatenate(all_feats, axis=0)
    Y = np.concatenate(all_targets, axis=0)

    return X, Y