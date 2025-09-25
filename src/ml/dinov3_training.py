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
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import os
FEAT_CACHE: dict[str, torch.Tensor] = {}
FEAT_CACHE_MAX = 2000  # tune; each ~7 MB in fp16 for 1536→96x96x384

import tqdm
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ––– local imports –––
try:
    from src.backend import db_ml as backend_db
except ModuleNotFoundError:  # script run from repo root
    _root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_root))
    from backend import db_ml as backend_db  # type: ignore


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


def _sanitize_for_filename(name: str) -> str:
    """Sanitize class names for safe filenames (alnum, dash, underscore).

    Keeps the original class string for DB, only sanitizes the PNG filename.
    """
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return safe or "class"


def _from_ppm(ppm_val) -> Optional[float]:
    """Convert stored PPM integer (or already-normalized float) to [0,1]."""
    if ppm_val is None:
        return None
    try:
        f = float(ppm_val)
    except (TypeError, ValueError):
        return None
    if 0.0 <= f <= 1.0:
        return f
    try:
        i = int(round(f))
    except (TypeError, ValueError):
        return None
    if i < 0:
        i = 0
    elif i > 1_000_000:
        i = 1_000_000
    return i / 1_000_000.0


def gather_annotated_items(samples: Sequence[dict]) -> List[Tuple[int, str, Dict[str, List[dict]]]]:
    """Get images with point annotations as ``(sample_id, filepath, annotations_by_class)`` list.

    Compliance: read coordinates from ``col01``/``row01`` (PPM ints) and convert to [0,1].
    """
    items: List[Tuple[int, str, Dict[str, List[dict]]]] = []
    for s in samples:
        anns = backend_db.get_annotations(s["id"])
        points = [
            a
            for a in anns
            if a.get("type") == "point" and a.get("col01") is not None and a.get("row01") is not None
        ]
        if points:
            grouped: Dict[str, List[dict]] = {}
            for point in points:
                cls = point.get("class", "unknown")
                x = _from_ppm(point.get("col01"))
                y = _from_ppm(point.get("row01"))
                if x is None or y is None:
                    continue
                grouped.setdefault(cls, []).append({
                    "x": x,
                    "y": y,
                    "timestamp": point.get("timestamp", 0),
                })
            if grouped:
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


def extract_features_cached(model: torch.nn.Module,
                            image_padded: Image.Image,
                            fp: str,
                            resize_used: int,
                            arch: str) -> torch.Tensor:
    """Return F×H×W torch tensor on CPU (fp16), cached by (fp, mtime, resize, arch)."""
    key = f"{fp}:{int(os.path.getmtime(fp))}:{resize_used}:{arch}"
    feats = FEAT_CACHE.get(key)
    if feats is not None:
        return feats
    # compute once
    image_tensor = normalize_image(image_padded)
    with torch.no_grad():
        feats = extract_features(model, image_tensor)  # F×H×W on device
    feats = feats.to("cpu").to(torch.float16).contiguous()
    # tiny FIFO eviction
    if len(FEAT_CACHE) >= FEAT_CACHE_MAX:
        FEAT_CACHE.pop(next(iter(FEAT_CACHE)))
    FEAT_CACHE[key] = feats
    return feats


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


def load_classifier(path: Path) -> Optional[LogisticRegression]:
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load classifier: {e}")
    return None


def save_classifier(clf: LogisticRegression, path: Path) -> None:
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
    print("[INIT] Loading classifier from", clf_path)

    LOGREG_KW = dict(class_weight="balanced")
    classifier = load_classifier(clf_path)
    prev_config: Optional[dict] = None
    cycle = 0
    model = None
    model_size = None
    current_resize = 1536
    # In-memory; lost on process kill (per spec).
    split_map: dict[int, str] = {}

    while True:
        print(f"\n[LOOP] Starting cycle {cycle}")
        try:
            config = backend_db.get_config()
        except Exception as e:
            print(f"[ERR] failed to load config: {e}", file=sys.stderr)
            time.sleep(5)
            continue

        if prev_config != config:
            print("[CONFIG] Detected config change or first load.")
            # Reload backbone only if architecture changed; keep classifier persistent
            arch = config.get("architecture", "small") or "small"
            if arch not in {"small", "large"}:
                print(f"[WARN] Unknown architecture '{arch}', defaulting to 'small'")
                arch = "small"
            if model is None or arch != model_size:
                print("[INFO] Loading DINOv3 backbone:", arch)
                model = load_dinov3_model(arch)
                model_size = arch
            backend_db.reset_training_stats()
            print("[METRICS] Cleared train/val curves due to config change.")
            # Update preprocessing resize but do not reset classifier/checkpoint
            current_resize = config.get("resize", 1536) or 1536
            print(f"[CONFIG] Set resize to {current_resize}")
            prev_config = config

        current_resize = config.get("resize", current_resize) or current_resize
        # Run only for segmentation task
        task = (config.get("task") or "classification").lower()
        if task != "segmentation":
            print("[INFO] Task is not 'segmentation' — pausing 1s…")
            time.sleep(1)
            continue
        if not config.get("ai_should_be_run", False):
            print("[INFO] Run flag disabled — pausing 1s…")
            time.sleep(1)
            continue

        budget = config.get("budget", 1000)
        print(f"[INFO] Budget for this cycle: {budget}")

        print("[DB] Fetching all samples…")
        samples = backend_db.get_all_samples()
        allowed_ids = backend_db.get_sample_ids_for_path_filter(
            config.get("sample_path_filter")
        )
        print(f"[DB] Found {len(samples)} samples.")
        print("[DB] Gathering annotated items…")
        annotated = gather_annotated_items(samples)
        print(f"[DB] Found {len(annotated)} annotated items.")
        if not annotated:
            print("[WARN] No point annotations available — pausing 1s…")
            time.sleep(1)
            continue

        print("[FEAT] Extracting features for annotated points…")
        X, y, img_ids = [], [], []
        for s_id, fp, pts_by_class in tqdm.tqdm(annotated):
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            with Image.open(image_path) as image:
                image_padded, new_w, new_h = resize_pad(image, target_size=current_resize)
            feats = extract_features_cached(model, image_padded, fp, current_resize, model_size)
            for cls, pts in pts_by_class.items():
                for pt in pts:
                    col = pt.get("x")
                    row = pt.get("y")
                    if col is None or row is None:
                        continue
                    # Map normalized [0,1] to resized image content (top-left aligned)
                    x_padded = float(col) * (new_w - 1)
                    y_padded = float(row) * (new_h - 1)
                    # Map to feature map coordinates using floor to select the patch cell
                    fx = int(np.floor(x_padded / 16.0))
                    fy = int(np.floor(y_padded / 16.0))
                    _, H, W = feats.shape
                    fx = min(max(fx, 0), W - 1)
                    fy = min(max(fy, 0), H - 1)
                    vec = feats[:, fy, fx].cpu().numpy()
                    X.append(vec)
                    y.append(cls)
                    img_ids.append(s_id)

        print(f"[FEAT] Extracted {len(X)} feature/label pairs.")
        if not X:
            print("[WARN] No valid feature/label pairs found — pausing 1s…")
            time.sleep(1)
            continue

        X = np.stack(X)
        y = np.array(y)
        # In-memory per-process split of labeled images into {train, val}
        labeled_ids = sorted({int(i) for i in img_ids})
        # Compute current counts only over currently labeled ids
        train_count = sum(1 for i in labeled_ids if split_map.get(i) == "train")
        val_count = sum(1 for i in labeled_ids if split_map.get(i) == "val")
        def _assign_side() -> str:
            # Keep ~80/20 ratio while appending
            total = train_count + val_count
            if total == 0:
                return "train"
            return "train" if (train_count / max(1, total)) < 0.8 else "val"

        # Append new items without reshuffling existing ones
        for i in labeled_ids:
            if i not in split_map:
                side = _assign_side()
                split_map[i] = side
                if side == "train":
                    train_count += 1
                else:
                    val_count += 1

        print(f"[SPLIT] {train_count} train, {val_count} val images.")
        train_imgs = {i for i, side in split_map.items() if side == "train"}
        train_mask = np.array([int(i) in train_imgs for i in img_ids])
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]

        # Train/val split already computed above -> X_train, y_train, X_val, y_val

        # 1) Bail if you don’t have at least 2 classes
        labels_present = np.unique(y_train)
        if labels_present.size < 2:
            print("[INFO] Not enough classes in train split — skipping training this cycle")
            acc = None
        else:
            # 2) Reinit if first time or feature dim changed
            need_new = False
            if classifier is None:
                need_new = True
            else:
                try:
                    prev_dim = classifier.coef_.shape[1]
                except Exception:
                    prev_dim = -1
                if prev_dim != X_train.shape[1]:
                    print("[TRAIN] Feature dimension changed; reinitializing LogisticRegression.")
                    need_new = True

            if need_new:
                backend_db.reset_training_stats()
                classifier = LogisticRegression(**LOGREG_KW)

            # 3) Fit fresh each cycle for stability/determinism
            classifier.set_params(**LOGREG_KW)  # keep config stable if something touched it
            classifier.fit(X_train, y_train)
            print(f"[TRAIN] Trained multinomial LR on {X_train.shape[0]} samples, {len(classifier.classes_)} classes.")

            # 4) Validation
            if y_val.size:
                y_pred = classifier.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                print(f"[VAL] Accuracy: {acc:.3f} | n_val: {len(y_val)}")
            else:
                acc = None

            # 5) Save checkpoint
            print("[TRAIN] Saving classifier checkpoint…")
            save_classifier(classifier, clf_path)

        # Store training stats (acc may be None above)
        print("[DB] Storing training stats…")
        backend_db.store_training_stats(cycle, None, None, acc)

        if classifier is not None:
            print("[TRAIN] Saving classifier checkpoint…")
            save_classifier(classifier, clf_path)

        if len(y_val) > 0 and classifier is not None:
            y_pred = classifier.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"[VAL] Accuracy: {acc:.3f} | n_val: {len(y_val)}")
        else:
            acc = None

        print("[DB] Storing training stats…")
        backend_db.store_training_stats(cycle, None, None, acc)

        labeled_set = set(fp for _, fp, _ in annotated)
        unlabeled = [
            s
            for s in samples
            if s["sample_filepath"] not in labeled_set
            and (allowed_ids is None or s["id"] in allowed_ids)
        ]
        filtered_annotated = [
            (s_id, fp, pts)
            for s_id, fp, pts in annotated
            if allowed_ids is None or s_id in allowed_ids
        ]
        to_predict = (
            [dict(id=s_id, sample_filepath=fp) for s_id, fp, _ in filtered_annotated]
            + [s for s in unlabeled]
        )[:budget]
        print(f"[PRED] Will predict on {len(to_predict)} images (labeled + unlabeled, up to budget)")

        if classifier is None:
            print("[INFO] No trained classifier available — skipping prediction this cycle; pausing 1s…")
            time.sleep(1)
            continue

        Path("session/preds").mkdir(parents=True, exist_ok=True)
        # Don't print inside the prediction for loop to avoid console bloat
        for dtp in tqdm.tqdm(to_predict):
            fp = dtp["sample_filepath"]
            s_id = dtp["id"]
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            with Image.open(image_path) as image:
                image_padded, new_w, new_h = resize_pad(image, target_size=current_resize)
            feats = extract_features_cached(model, image_padded, fp, current_resize, model_size)
            F, H, W = feats.shape
            feats_np = feats.permute(1, 2, 0).reshape(-1, F).cpu().numpy()
            classes_map = classifier.predict(feats_np).reshape(H, W)
            # crop the map to the image 
            image_rows_01 = new_h / max(new_w, new_h)
            image_cols_01 = new_w / max(new_w, new_h)
            assert image_rows_01 == 1.0 or image_cols_01 == 1.0, "Only one side should be padded"
            if image_rows_01 < 1.0:
                # Zero-padded columns on right
                classes_map = classes_map[:int(H * image_rows_01)]
            elif image_cols_01 < 1.0:
                # Zero-padded rows on bottom
                classes_map = classes_map[:, :int(W * image_cols_01)]
            # One PNG per class under session/preds named <sample_id>_<class>.png
            preds_batch = []
            for class_name in np.unique(classes_map):
                mask_bool = (classes_map == class_name).astype(bool)  # this should be squared with image at the top left
                safe_cls = _sanitize_for_filename(class_name)
                outpath = (Path("session") / "preds" / f"{s_id}_{safe_cls}.png").resolve()
                Image.fromarray(mask_bool).save(outpath)
                preds_batch.append({
                    "sample_id": s_id,
                    "class": str(class_name),
                    "type": "mask",
                    "mask_path": outpath.as_posix(),
                })
            if preds_batch:
                backend_db.set_predictions_batch(preds_batch)
        print(
            f"[cycle {cycle}] trained on {len(X_train)} pts, val {len(y_val)} pts, predicted {len(to_predict)} images"
        )
        cycle += 1
        torch.cuda.empty_cache()
        # Normal cycles do not pause per spec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
