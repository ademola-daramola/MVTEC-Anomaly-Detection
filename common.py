"""Shared config, feature extractor and scoring math for the anomaly pipeline."""
import os
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ---- Configuration (override with environment variables; no hard-coded paths) ----
MVTEC_ROOT = Path(os.environ.get("MVTEC_ROOT", "./data/mvtec"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
N_CHANNELS = int(os.environ.get("N_CHANNELS", 100))  # per-patch covariance is C x C, so memory grows with C^2
SMOOTH_SIGMA = float(os.environ.get("SMOOTH_SIGMA", 2))
COV_EPSILON = 1e-2
IMG_SIZE = 224
SEED = 42
MODEL_VERSION = "padim-r18-l123-v2"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Resize straight to 224 (no center crop) so image borders, where some defects sit, are kept.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchFeatureExtractor(nn.Module):
    """Frozen ResNet18; concatenates layer1, layer2 and layer3 features at 56x56 -> 448 channels."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)    # [B, 64, 56, 56]
        x2 = self.layer2(x1)   # [B, 128, 28, 28]
        x3 = self.layer3(x2)   # [B, 256, 14, 14]
        size = x1.shape[2:]
        x2 = F.interpolate(x2, size=size, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=size, mode="bilinear", align_corners=False)
        return torch.cat([x1, x2, x3], dim=1)  # [B, 448, 56, 56]


TOTAL_CHANNELS = 448
FEATURE_HW = 56


def select_channels(k: int = N_CHANNELS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(TOTAL_CHANNELS, size=k, replace=False))


@torch.no_grad()
def patch_features(extractor, images: torch.Tensor, channels: np.ndarray) -> torch.Tensor:
    """images [B,3,H,W] -> [B, P, K] float32 on the extractor's device."""
    feats = extractor(images)[:, torch.as_tensor(channels, device=images.device)]
    b, k, h, w = feats.shape
    return feats.permute(0, 2, 3, 1).reshape(b, h * w, k).float()


@torch.no_grad()
def mahalanobis_maps(feats: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    """Vectorized per-patch Mahalanobis distance. feats [N,P,K], mean [P,K], inv_cov [P,K,K] -> [N,P]."""
    diff = (feats - mean).permute(1, 0, 2)          # [P, N, K]
    tmp = torch.bmm(diff, inv_cov)                   # [P, N, K]
    d2 = (tmp * diff).sum(-1).clamp_min(0)           # [P, N]
    return d2.sqrt().t()                             # [N, P]


def smooth_map(dist_row: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    return scipy.ndimage.gaussian_filter(dist_row.reshape(FEATURE_HW, FEATURE_HW), sigma=sigma)
