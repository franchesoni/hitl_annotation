#!/usr/bin/env python
"""DINOv3-based Point Segmentation Training and Inference

This script implements a continuous training loop for point-based image segmentation using
DINOv3 features. It monitors the annotation database and trains/applies a linear classifier
on image features extracted at annotated point locations.

Workflow:
1. **Config Check**: Verify that the 'ai_should_be_run' flag is enabled in the database config
2. **Annotation Discovery**: Check for images with point annotations (type='point') in the database
3. **Feature Extraction**:
   - Load images with point annotations
   - Resize images to 1536x1536 (with zero-padding) for consistent processing
   - Extract DINOv3 feature maps (96x96 at 1/16 resolution) for each image
   - Map point coordinates from original image size to feature map coordinates
   - Sample feature vectors at annotated point locations (nearest neighbor for sub-pixel coords)
   - Create training dataset: each annotated point becomes a separate (feature_vector, class_label) sample
4. **Linear Classifier Training**:
   - Split annotated data by images (all points from same image stay in same split) with 80/20 train/val
   - Train/update linear classifier incrementally on new annotations (preserving previous learning)
   - Validate performance on held-out images
   - Save classifier state between script runs (similar to fastai_training.py checkpoint system)
5. **Dense Prediction and Inference**:
   - Apply classifier to dense feature maps for all images (annotated + unlabeled)
   - Generate prediction maps at feature resolution (96x96) for each image
   - Process unlabeled images sequentially up to the configured budget
   - Store predictions as dense maps or sampled points in the database
6. **Logging and Storage**:
   - Log training metrics (accuracy, loss, etc.) to database using store_training_stats()
   - Store predictions for unlabeled images using set_predictions_batch()
   - Log processing statistics and cycle performance
   - Handle graceful exit on Ctrl-C/SIGTERM

Implementation Details:
- **Point sampling**: Each annotated point is a separate training sample
- **Feature aggregation**: No aggregation - individual points used directly
- **Train/val split**: By images to prevent data leakage between splits
- **Coordinate mapping**: Scale point coords from original → 1536x1536 → 96x96 feature map
- **Dense prediction**: Apply classifier to every feature map location (96x96 predictions per image)
- **Budget allocation**: Process unlabeled images in sequential order
- **Model persistence**: Save/load classifier state like fastai_training.py checkpoint system
- **Incremental learning**: Update classifier with new annotations rather than full retraining

Inspired by:
- fastai_training.py: Database integration, continuous training loop, and model persistence
- dinov3_seg.py: Image preprocessing and DINOv3 feature extraction

The implementation deliberately avoids ``joblib``; the classifier is serialized with
``pickle`` instead.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss

# ––– local imports –––
try:
    from src.backend import db as backend_db
except ModuleNotFoundError:  # script run from repo root
    _root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_root))
    from backend import db as backend_db  # type: ignore


# ---------------------------------------------------------------------------
# image and feature helpers
# ---------------------------------------------------------------------------


def resize_pad(im: Image.Image, target_size: int = 1536) -> tuple[Image.Image, int, int]:
    """Resize image with largest side to ``target_size`` and zero-pad to square.
    Returns (padded_image, new_w, new_h), where new_w/new_h are the resized image content inside the padded image."""
    from PIL import ImageOps

    w, h = im.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_size = (target_size // 16) * 16
    padded = ImageOps.pad(resized, (canvas_size, canvas_size), color=(0, 0, 0), centering=(0, 0))
    return padded, new_w, new_h


def normalize_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor for DINOv3."""
    x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0)
    x = (x / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)
    return x


def gather_annotated_items(samples: Sequence[dict]) -> List[Tuple[str, Dict[str, List[dict]]]]:
    """Get images with point annotations as ``(filepath, annotations_by_class)`` list."""
    items: List[Tuple[str, Dict[str, List[dict]]]] = []
    for s in samples:
        anns = backend_db.get_annotations(s["id"])
        points = [a for a in anns if a.get("type") == "point" and a.get("col") is not None and a.get("row") is not None]
        if points:
            grouped: Dict[str, List[dict]] = {}
            for point in points:
                cls = point.get("class", "unknown")
                grouped.setdefault(cls, []).append({"x": point["col"], "y": point["row"], "timestamp": point.get("timestamp", 0)})
            items.append((s["id"], s["sample_filepath"], grouped))
    return items


def load_dinov3_model(size: str = "small") -> torch.nn.Module:
    """Load DINOv3 model with specified size and set to eval mode."""
    if size == "large":
        weights = "src/ml/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        model_name = "dinov3_vitl16"
    elif size == "small":
        weights = "src/ml/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        model_name = "dinov3_vits16"
    else:
        raise ValueError(f"Size '{size}' not implemented. Use 'small' or 'large'.")

    dinov3_path = Path(__file__).parent / "dinov3"
    if not dinov3_path.exists():
        raise FileNotFoundError(f"DINOv3 directory not found at {dinov3_path}")

    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[WARN] Weights file not found: {weights_path}")
        print("[INFO] Loading model without local weights")
        weights = None

    model = torch.hub.load(repo_or_dir=str(dinov3_path), model=model_name, source="local", weights=weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_features(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """Extract DINOv3 features for an image tensor of shape ``(1, C, H, W)``."""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        feats = model.forward_features(image_tensor)["x_norm_patchtokens"]
        Ph = int(image_tensor.shape[2] / 16)
        Pw = int(image_tensor.shape[3] / 16)
        F = feats.shape[-1]
        feats = feats.permute(0, 2, 1).reshape(1, F, Ph, Pw)
    return feats[0]


# ---------------------------------------------------------------------------
# classifier persistence helpers (pickle instead of joblib)
# ---------------------------------------------------------------------------


def load_classifier(path: Path) -> Optional[SGDClassifier]:
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load classifier: {e}")
    return None


def save_classifier(clf: SGDClassifier, path: Path) -> None:
    try:
        with path.open("wb") as f:
            pickle.dump(clf, f)
    except Exception as e:
        print(f"[WARN] Failed to save classifier: {e}")


# ---------------------------------------------------------------------------
# main processing loop
# ---------------------------------------------------------------------------


def main() -> None:
    clf_path = Path("session/dinov3_linear_classifier_seg.pkl")
    classifier = load_classifier(clf_path)
    prev_config: Optional[dict] = None
    cycle = 0
    model = None
    model_size = None
    current_resize = 1536

    while True:
        try:
            config = backend_db.get_config()
        except Exception as e:
            print(f"[ERR] failed to load config: {e}", file=sys.stderr)
            time.sleep(5)
            continue

        if prev_config != config:
            if prev_config is not None:
                print("[INFO] Config changed; resetting classifier")
            arch = config.get("architecture", "small") or "small"
            if arch not in {"small", "large"}:
                print(f"[WARN] Unknown architecture '{arch}', defaulting to 'small'")
                arch = "small"
            if model is None or arch != model_size:
                model = load_dinov3_model(arch)
                model_size = arch
            current_resize = config.get("resize", 1536) or 1536
            classifier = None
            try:
                if clf_path.exists():
                    clf_path.unlink()
            except Exception:
                pass
            prev_config = config
            cycle = 0

        sleep_s = config.get("sleep", 5) or 5
        current_resize = config.get("resize", current_resize) or current_resize
        if not config.get("ai_should_be_run", False):
            print("[INFO] Run flag disabled — sleeping…")
            time.sleep(max(1, sleep_s))
            continue

        budget = config.get("budget", 1000)

        samples = backend_db.get_all_samples()
        annotated = gather_annotated_items(samples)
        if not annotated:
            print("[WARN] No point annotations available — sleeping…")
            time.sleep(max(1, sleep_s))
            continue

        X, y, img_ids = [], [], []
        for _, fp, pts_by_class in annotated:
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            image_padded, new_w, new_h = resize_pad(image, target_size=current_resize)
            image_tensor = normalize_image(image_padded)
            feats = extract_features(model, image_tensor)
            for cls, pts in pts_by_class.items():
                for pt in pts:
                    col = pt.get("x")
                    row = pt.get("y")
                    if col is None or row is None:
                        continue
                    # Map normalized [0,1] to resized image content (top-left aligned)
                    x_padded = float(col) * (new_w - 1)
                    y_padded = float(row) * (new_h - 1)
                    # Map to feature map coordinates (center of patch)
                    fx = int(round((x_padded + 0.5 * 16) / 16))
                    fy = int(round((y_padded + 0.5 * 16) / 16))
                    F, H, W = feats.shape
                    fx = min(max(fx, 0), W - 1)
                    fy = min(max(fy, 0), H - 1)
                    vec = feats[:, fy, fx].cpu().numpy()
                    X.append(vec)
                    y.append(cls)
                    img_ids.append(fp)

        if not X:
            print("[WARN] No valid feature/label pairs found — sleeping…")
            time.sleep(max(1, sleep_s))
            continue

        X = np.stack(X)
        y = np.array(y)
        unique_imgs = list({fp for fp in img_ids})
        np.random.seed(42)
        np.random.shuffle(unique_imgs)
        n_train = int(0.8 * len(unique_imgs))
        train_imgs = set(unique_imgs[:n_train])
        train_mask = np.array([fp in train_imgs for fp in img_ids])
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]

        if classifier is None:
            classifier = SGDClassifier(loss="log_loss", max_iter=1000)
            classifier.partial_fit(X_train, y_train, classes=np.unique(y))
        else:
            classifier.partial_fit(X_train, y_train)
        save_classifier(classifier, clf_path)

        if len(y_val) > 0:
            y_pred = classifier.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"[VAL] Accuracy: {acc:.3f} | n_val: {len(y_val)}")
        else:
            acc = None
            loss = None

        backend_db.store_training_stats(cycle, None, None, acc)

        labeled_set = set(fp for _, fp, _ in annotated)
        unlabeled = [s for s in samples if s["sample_filepath"] not in labeled_set]
        to_predict = ([dict(id=s_id, sample_filepath=fp) for s_id, fp, _ in annotated] + [s for s in unlabeled])[:budget]

        Path("session/preds").mkdir(parents=True, exist_ok=True)
        for dtp in to_predict:
            fp = dtp["sample_filepath"]
            s_id = dtp["id"]
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            image_padded, new_w, new_h = resize_pad(image, target_size=current_resize)
            image_tensor = normalize_image(image_padded)
            feats = extract_features(model, image_tensor)
            F, H, W = feats.shape
            feats_np = feats.permute(1, 2, 0).reshape(-1, F).cpu().numpy()
            classes_map = classifier.predict(feats_np).reshape(H, W)
            for class_name in np.unique(classes_map):
                mask = (classes_map == class_name)
                outpath = Path("session") / "preds" / (image_path.stem + f"_pred_class_{class_name}.png")
                Image.fromarray(mask).save(outpath)
                preds_batch = [{
                    "sample_id": s_id,
                    "sample_filepath": str(image_path),
                    "class": class_name,
                    "type": "mask",
                    "mask_path": outpath.as_posix(),
                }]
                backend_db.set_predictions_batch(preds_batch)
        print(
            f"[cycle {cycle}] trained on {len(X_train)} pts, val {len(y_val)} pts, predicted {len(to_predict)} images — sleeping {sleep_s}s"
        )
        cycle += 1
        torch.cuda.empty_cache()
        time.sleep(max(1, sleep_s))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
