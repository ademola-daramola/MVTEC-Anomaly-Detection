"""Build per-category normal-state profiles from MVTec AD train/good and evaluate on test.

Usage:
    MVTEC_ROOT=/data/mvtec python train.py                    # all 15 categories
    python train.py --categories bottle screw --no-eval
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets

from common import (CATEGORIES, COV_EPSILON, MODEL_VERSION, MODELS_DIR, MVTEC_ROOT, N_CHANNELS,
                    RESULTS_DIR, PatchFeatureExtractor, get_device, mahalanobis_maps,
                    patch_features, select_channels, smooth_map, transform)


def make_loader(path, batch_size=16):
    ds = datasets.ImageFolder(root=str(path), transform=transform)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def build_profile(category, extractor, device, channels):
    _, loader = make_loader(MVTEC_ROOT / category / "train")
    chunks = []
    for images, _ in loader:
        chunks.append(patch_features(extractor, images.to(device), channels).cpu())
    feats = torch.cat(chunks).to(device)                      # [N, P, K]
    n, p, k = feats.shape

    mean = feats.mean(dim=0)                                  # [P, K]
    centered = (feats - mean).permute(1, 2, 0)                # [P, K, N]
    cov = torch.bmm(centered, centered.transpose(1, 2)) / (n - 1)
    cov = cov + COV_EPSILON * torch.eye(k, device=device)
    inv_cov = torch.linalg.inv(cov.double()).float()          # invert in float64 for stability
    return {"mean": mean, "inv_cov": inv_cov, "channels": channels}


@torch.no_grad()
def evaluate(category, profile, extractor, device):
    ds, loader = make_loader(MVTEC_ROOT / category / "test")
    scores, labels = [], []
    for images, y in loader:
        feats = patch_features(extractor, images.to(device), profile["channels"])
        dist = mahalanobis_maps(feats, profile["mean"], profile["inv_cov"]).cpu().numpy()
        scores.extend(smooth_map(row).max() for row in dist)
        labels.extend(0 if ds.classes[i] == "good" else 1 for i in y.tolist())
    scores, labels = np.array(scores), np.array(labels)

    auc = roc_auc_score(labels, scores)
    prec, rec, thr = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[int(np.argmax(f1[:-1]))])
    return auc, threshold


def save_profile(category, profile, threshold):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODELS_DIR / f"{category}.npz",
        mean=profile["mean"].cpu().numpy(),
        inv_cov=profile["inv_cov"].cpu().numpy(),
        channels=profile["channels"],
        threshold=np.float32(threshold),
        version=MODEL_VERSION,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="*", default=CATEGORIES)
    parser.add_argument("--no-eval", action="store_true", help="skip test evaluation (threshold stays 0)")
    args = parser.parse_args()

    device = get_device()
    extractor = PatchFeatureExtractor().to(device)
    channels = select_channels(N_CHANNELS)
    results = {}

    print(f"Device: {device} | channels: {len(channels)} | model: {MODEL_VERSION}\n")
    for category in args.categories:
        if not (MVTEC_ROOT / category / "train").exists():
            print(f"[skip] {category}: not found under {MVTEC_ROOT}")
            continue
        print(f"--- {category} ---")
        profile = build_profile(category, extractor, device, channels)
        threshold = 0.0
        if not args.no_eval:
            auc, threshold = evaluate(category, profile, extractor, device)
            results[category] = auc
            print(f"-> image AUROC {auc:.4f} | F1-optimal threshold {threshold:.3f}")
        save_profile(category, profile, threshold)

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "auroc.json").write_text(json.dumps(results, indent=2))
        print(f"\nMean AUROC over {len(results)} categories: {np.mean(list(results.values())):.4f}")
        for cat, auc in results.items():
            print(f"{cat}: {auc:.4f}")


if __name__ == "__main__":
    main()
