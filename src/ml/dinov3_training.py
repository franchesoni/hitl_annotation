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
            items.append((s["sample_filepath"], grouped))
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


def run_forever(model_size: str, sleep_seconds: int) -> None:
    def _exit_handler(*_):
        print("\n[INFO] Exiting…")
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)

    model = load_dinov3_model(model_size)
    clf_path = Path("dinov3_linear_classifier.pkl")
    classifier = load_classifier(clf_path)
    prev_config: Optional[dict] = None
    cycle = 0

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
            classifier = None
            try:
                if clf_path.exists():
                    clf_path.unlink()
            except Exception:
                pass
            prev_config = config
            cycle = 0

        sleep_s = config.get("sleep", sleep_seconds) or sleep_seconds
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
        for fp, pts_by_class in annotated:
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            image_padded, new_w, new_h = resize_pad(image)
            image_tensor = normalize_image(image_padded)
            feats = extract_features(model, image_tensor)
            for cls, pts in pts_by_class.items():
                for pt in pts:
                    col = pt.get("col")
                    row = pt.get("row")
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
            try:
                y_proba = classifier.predict_proba(X_val)
                loss = log_loss(y_val, y_proba)
            except Exception:
                loss = None
            print(f"[VAL] Accuracy: {acc:.3f} | Loss: {loss if loss is not None else 'N/A'} | n_val: {len(y_val)}")
        else:
            acc = None
            loss = None

        backend_db.store_training_stats(cycle, None, loss, acc)

        labeled_set = set(fp for fp, _ in annotated)
        unlabeled = [s for s in samples if s["sample_filepath"] not in labeled_set]
        to_predict = [fp for fp, _ in annotated] + [s["sample_filepath"] for s in unlabeled[:budget]]

        for fp in to_predict:
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            image_padded, new_w, new_h = resize_pad(image)
            image_tensor = normalize_image(image_padded)
            feats = extract_features(model, image_tensor)
            F, H, W = feats.shape
            feats_np = feats.permute(1, 2, 0).reshape(-1, F).cpu().numpy()
            pred_labels = classifier.predict(feats_np)
            try:
                pred_probs = classifier.predict_proba(feats_np)
            except Exception:
                pred_probs = None
            pred_map = np.array(pred_labels).reshape(H, W)
            preds_batch = []
            for y_idx in range(H):
                for x_idx in range(W):
                    # Map feature map location to padded image pixel (top-left aligned)
                    x_padded = (x_idx + 0.5) * 16
                    y_padded = (y_idx + 0.5) * 16
                    # Map to normalized [0,1] in original image, then to pixel
                    if 0 <= x_padded < new_w and 0 <= y_padded < new_h:
                        col_norm = x_padded / (new_w - 1) if new_w > 1 else 0.0
                        row_norm = y_padded / (new_h - 1) if new_h > 1 else 0.0
                        cx = int(round(col_norm * (orig_w - 1)))
                        cy = int(round(row_norm * (orig_h - 1)))
                    else:
                        # Outside image content, set to -1
                        cx, cy = -1, -1
                    pred = {"type": "label", "class": str(pred_map[y_idx, x_idx])}
                    if pred_probs is not None:
                        pred["probability"] = float(np.max(pred_probs[y_idx * W + x_idx]))
                    preds_batch.append((None, [pred, {"x": cx, "y": cy}]))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 feature extraction for point annotations")
    parser.add_argument("--size", choices=["small", "large"], default="small", help="Model size")
    parser.add_argument("--sleep", type=int, default=5, help="Sleep time between checks")
    parser.add_argument("--db", help="Path to annotation SQLite db", default=None)
    args = parser.parse_args()

    if args.db:
        backend_db.DB_PATH = args.db

    run_forever(args.size, args.sleep)


if __name__ == "__main__":
    main()
