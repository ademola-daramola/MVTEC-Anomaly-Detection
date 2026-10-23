"""Serving-side scorer: loads saved per-category profiles and scores single images.

This is what the FastAPI worker imports. CLI for quick tests:
    python scorer.py bottle path/to/image.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common import (IMG_SIZE, MODELS_DIR, RESULTS_DIR, PatchFeatureExtractor, get_device,
                    mahalanobis_maps, patch_features, smooth_map, transform)


class Scorer:
    def __init__(self, models_dir: Path = MODELS_DIR, device=None):
        self.models_dir = Path(models_dir)
        self.device = device or get_device()
        self.extractor = PatchFeatureExtractor().to(self.device)
        self._profiles = {}

    def _profile(self, category: str):
        if category not in self._profiles:
            path = self.models_dir / f"{category}.npz"
            if not path.exists():
                raise FileNotFoundError(f"No trained profile for category '{category}' at {path}")
            d = np.load(path)
            self._profiles[category] = {
                "mean": torch.from_numpy(d["mean"]).to(self.device),
                "inv_cov": torch.from_numpy(d["inv_cov"]).to(self.device),
                "channels": d["channels"],
                "threshold": float(d["threshold"]),
                "version": str(d["version"]),
            }
        return self._profiles[category]

    def available_categories(self):
        return sorted(p.stem for p in self.models_dir.glob("*.npz"))

    @torch.no_grad()
    def score_image(self, image_path: str, category: str, heatmap_path: str | None = None) -> dict:
        profile = self._profile(category)
        pil = Image.open(image_path).convert("RGB")
        x = transform(pil).unsqueeze(0).to(self.device)

        feats = patch_features(self.extractor, x, profile["channels"])
        dist = mahalanobis_maps(feats, profile["mean"], profile["inv_cov"]).cpu().numpy()[0]
        score_map = smooth_map(dist)
        score = float(score_map.max())
        threshold = profile["threshold"]

        result = {
            "score": score,
            "threshold": threshold,
            "is_anomalous": bool(threshold > 0 and score >= threshold),
            "model_version": profile["version"],
            "heatmap_path": None,
        }
        if heatmap_path:
            self._save_heatmap(pil, score_map, threshold or score_map.max(), heatmap_path)
            result["heatmap_path"] = str(heatmap_path)
        return result

    @staticmethod
    def _save_heatmap(pil_img, score_map, threshold, save_path):
        # Normalize against the category threshold, not per-image min/max, so a good image stays cool.
        norm = np.clip(score_map / (2 * threshold + 1e-8), 0, 1)
        heat = cv2.resize(norm.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
        orig = cv2.cvtColor(np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_RGB2BGR)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.addWeighted(orig, 0.5, heat, 0.5, 0))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scorer.py <category> <image_path>")
    category, image = sys.argv[1], sys.argv[2]
    out = RESULTS_DIR / f"{category}_{Path(image).stem}_heatmap.png"
    print(Scorer().score_image(image, category, heatmap_path=out))
