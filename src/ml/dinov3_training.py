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
from dataclasses import dataclass
import os
import math
import random
FEAT_CACHE: dict[str, torch.Tensor] = {}
FEAT_CACHE_MAX = 2000  # tune; each ~7 MB in fp16 for 1536→96x96x384
IGNORE_INDEX = 255

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ––– local imports –––
try:
    from src.backend import db_ml as backend_db
except ModuleNotFoundError:  # script run from repo root
    _root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_root))
    from backend import db_ml as backend_db  # type: ignore


SESSION_DIR = Path(backend_db.DB_PATH).parent
MASKS_DIR = SESSION_DIR / "masks"


# Placeholder for upcoming mask-annotation integration into the training loop.
def iter_mask_annotations_for_training() -> List[Tuple[int, str]]:
    """Return mask annotations once mask-based training support is implemented."""
    # TODO: integrate accepted mask annotations into the DINOv3 segmentation training pipeline.
    return []


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedItem:
    sample_id: int
    filepath: str
    points_by_class: Dict[str, List[dict]]
    masks_by_class: Dict[str, List[dict]]


@dataclass
class ImageTrainingSample:
    sample_id: int
    features: torch.Tensor
    mask_target: torch.Tensor
    point_targets: List[Tuple[int, int, int]]
    mask_pixel_count: int


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


def gather_annotated_items(samples: Sequence[dict]) -> List[AnnotatedItem]:
    """Collect images that have point and/or mask annotations."""

    items: List[AnnotatedItem] = []
    for s in samples:
        anns = backend_db.get_annotations(s["id"])
        points: Dict[str, List[dict]] = {}
        masks: Dict[str, List[dict]] = {}
        for ann in anns:
            ann_type = ann.get("type")
            if ann_type == "point" and ann.get("col01") is not None and ann.get("row01") is not None:
                cls = ann.get("class", "unknown")
                x = _from_ppm(ann.get("col01"))
                y = _from_ppm(ann.get("row01"))
                if x is None or y is None:
                    continue
                points.setdefault(cls, []).append({
                    "x": x,
                    "y": y,
                    "timestamp": ann.get("timestamp", 0),
                })
            elif ann_type == "mask" and ann.get("mask_path"):
                cls = ann.get("class", "unknown")
                masks.setdefault(cls, []).append({
                    "mask_path": ann.get("mask_path"),
                    "timestamp": ann.get("timestamp", 0),
                })
        if points or masks:
            items.append(
                AnnotatedItem(
                    sample_id=s["id"],
                    filepath=s["sample_filepath"],
                    points_by_class=points,
                    masks_by_class=masks,
                )
            )
    return items


def _resolve_mask_path(mask_path: str) -> Optional[Path]:
    if not mask_path:
        return None

    candidate = Path(mask_path)

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(SESSION_DIR / candidate)
        candidates.append(MASKS_DIR / candidate)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"Mask path {mask_path} could not be resolved")


def build_mask_target(
    masks_by_class: Dict[str, List[dict]],
    class_to_idx: Dict[str, int],
    canvas_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, int]:
    """Create a dense target map from binary mask files."""

    target = torch.full(canvas_shape, IGNORE_INDEX, dtype=torch.long)
    labeled_pixels = 0

    for cls, entries in masks_by_class.items():
        cls_idx = class_to_idx.get(cls)
        if cls_idx is None or not entries:
            continue
        entries_sorted = sorted(entries, key=lambda e: e.get("timestamp") or 0, reverse=True)
        mask_path = entries_sorted[0].get("mask_path")
        resolved = _resolve_mask_path(str(mask_path))
        with Image.open(resolved) as mask_im:
            mask_arr = np.array(mask_im)
        if mask_arr.ndim >= 3:
            mask_arr = mask_arr[..., 0]
        mask_bool = mask_arr.astype(bool)
        H, W = canvas_shape
        h_eff = min(mask_bool.shape[0], H)
        w_eff = min(mask_bool.shape[1], W)
        if h_eff == 0 or w_eff == 0:
            continue
        region = mask_bool[:h_eff, :w_eff]
        region_sum = int(region.sum())
        if region_sum == 0:
            continue
        labeled_pixels += region_sum
        mask_tensor = torch.from_numpy(region.astype(np.bool_))
        target_slice = target[:h_eff, :w_eff]
        target_slice[mask_tensor] = cls_idx

    return target, labeled_pixels


class SegmentationHead(nn.Module):
    """1×1 convolutional head used for segmentation logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def expand_out_channels(self, new_out: int) -> None:
        if new_out == self.classifier.out_channels:
            return
        old_layer = self.classifier
        new_layer = nn.Conv2d(old_layer.in_channels, new_out, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(new_layer.weight, a=math.sqrt(5))
        if new_layer.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_layer.bias, -bound, bound)
        with torch.no_grad():
            keep = min(old_layer.out_channels, new_layer.out_channels)
            new_layer.weight[:keep].copy_(old_layer.weight[:keep])
            if old_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias[:keep].copy_(old_layer.bias[:keep])
        self.classifier = new_layer


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


def load_checkpoint(path: Path) -> Optional[dict]:
    if not path.exists():
        return None

    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected checkpoint format in {path}")


def save_checkpoint(state: dict, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(state, f)


def prepare_training_samples(
    annotated: Sequence[AnnotatedItem],
    class_to_idx: Dict[str, int],
    model: torch.nn.Module,
    current_resize: int,
    model_size: str,
) -> Tuple[List[ImageTrainingSample], int, int]:
    samples_out: List[ImageTrainingSample] = []
    total_points = 0
    total_mask_pixels = 0
    for item in annotated:
        image_path = Path(item.filepath)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {item.filepath}")
        with Image.open(image_path) as image:
            image_padded, new_w, new_h = resize_pad(image, target_size=current_resize)
        feats = extract_features_cached(model, image_padded, item.filepath, current_resize, model_size)
        feats = feats.to(torch.float32)
        _, H, W = feats.shape
        mask_target, mask_pixels = build_mask_target(item.masks_by_class, class_to_idx, (H, W))
        point_targets: List[Tuple[int, int, int]] = []
        for cls, pts in item.points_by_class.items():
            cls_idx = class_to_idx.get(cls)
            if cls_idx is None:
                continue
            for pt in pts:
                col = pt.get("x")
                row = pt.get("y")
                if col is None or row is None:
                    continue
                x_padded = float(col) * (new_w - 1)
                y_padded = float(row) * (new_h - 1)
                fx = int(np.floor(x_padded / 16.0))
                fy = int(np.floor(y_padded / 16.0))
                fx = min(max(fx, 0), W - 1)
                fy = min(max(fy, 0), H - 1)
                point_targets.append((fy, fx, cls_idx))
        if not point_targets and mask_pixels == 0:
            continue
        samples_out.append(
            ImageTrainingSample(
                sample_id=item.sample_id,
                features=feats,
                mask_target=mask_target,
                point_targets=point_targets,
                mask_pixel_count=mask_pixels,
            )
        )
        total_points += len(point_targets)
        total_mask_pixels += mask_pixels
    return samples_out, total_points, total_mask_pixels


def evaluate_accuracy(
    head: SegmentationHead,
    samples: Sequence[ImageTrainingSample],
    device: torch.device,
) -> Optional[float]:
    if not samples:
        return None
    total_correct = 0
    total = 0
    head.eval()
    with torch.no_grad():
        for sample in samples:
            logits = head(sample.features.unsqueeze(0).to(device))
            preds = logits.argmax(dim=1).squeeze(0).cpu()
            for fy, fx, cls_idx in sample.point_targets:
                total += 1
                if int(preds[fy, fx]) == cls_idx:
                    total_correct += 1
            mask = sample.mask_target
            valid = mask != IGNORE_INDEX
            if valid.any():
                total += int(valid.sum().item())
                total_correct += int((preds[valid] == mask[valid]).sum().item())
    head.train()
    return total_correct / total if total else None




# ---------------------------------------------------------------------------
# main processing loop
# ---------------------------------------------------------------------------


def main() -> None:
    clf_path = Path("session/dinov3_linear_classifier_seg.pkl")
    print("[INIT] Loading classifier from", clf_path)

    checkpoint = load_checkpoint(clf_path)
    class_names: List[str] = list(checkpoint.get("class_names", [])) if checkpoint else []
    prev_config: Optional[dict] = None
    cycle = 0
    model: Optional[torch.nn.Module] = None
    model_size: Optional[str] = None
    head: Optional[SegmentationHead] = None
    current_resize = 1536
    split_map: dict[int, str] = {}
    mask_loss_weight = float(checkpoint.get("mask_loss_weight", 1.0)) if checkpoint else 1.0

    while True:
        print(f"\n[LOOP] Starting cycle {cycle}")
        config = backend_db.get_config()

        if prev_config != config:
            print("[CONFIG] Detected config change or first load.")
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
            current_resize = config.get("resize", 1536) or 1536
            prev_config = config

        cfg_mask_weight = config.get("mask_loss_weight", mask_loss_weight)
        mask_loss_weight = 1.0 if cfg_mask_weight is None else float(cfg_mask_weight)
        if mask_loss_weight < 0.0:
            mask_loss_weight = 0.0

        current_resize = config.get("resize", current_resize) or current_resize
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
        allowed_ids = backend_db.get_sample_ids_for_path_filter(config.get("sample_path_filter"))
        print(f"[DB] Found {len(samples)} samples.")
        print("[DB] Gathering annotated items…")
        annotated = gather_annotated_items(samples)
        print(f"[DB] Found {len(annotated)} annotated items.")
        if not annotated:
            print("[WARN] No annotations available — pausing 1s…")
            time.sleep(1)
            continue

        classes_in_data = set()
        for item in annotated:
            classes_in_data.update(item.points_by_class.keys())
            classes_in_data.update(item.masks_by_class.keys())
        for cls in sorted(classes_in_data):
            if cls not in class_names:
                class_names.append(cls)
        if not class_names:
            print("[WARN] No classes found in annotations — pausing 1s…")
            time.sleep(1)
            continue
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        if model is None or model_size is None:
            print("[ERR] Backbone model not loaded — pausing 1s…")
            time.sleep(1)
            continue

        image_samples, total_points, total_mask_pixels = prepare_training_samples(
            annotated, class_to_idx, model, current_resize, model_size
        )
        print(
            f"[DATA] Prepared {len(image_samples)} samples | {total_points} points | {total_mask_pixels} mask pixels"
        )
        if not image_samples:
            print("[WARN] No usable training samples — pausing 1s…")
            time.sleep(1)
            continue

        labeled_ids = sorted({sample.sample_id for sample in image_samples})
        train_count = sum(1 for i in labeled_ids if split_map.get(i) == "train")
        val_count = sum(1 for i in labeled_ids if split_map.get(i) == "val")

        def _assign_side() -> str:
            total = train_count + val_count
            if total == 0:
                return "train"
            return "train" if (train_count / max(1, total)) < 0.8 else "val"

        for sample_id in labeled_ids:
            if sample_id not in split_map:
                side = _assign_side()
                split_map[sample_id] = side
                if side == "train":
                    train_count += 1
                else:
                    val_count += 1

        print(f"[SPLIT] {train_count} train, {val_count} val images.")
        train_samples = [s for s in image_samples if split_map.get(s.sample_id) == "train"]
        val_samples = [s for s in image_samples if split_map.get(s.sample_id) == "val"]

        feature_dim = train_samples[0].features.shape[0] if train_samples else image_samples[0].features.shape[0]
        need_new_head = head is None or head.classifier.in_channels != feature_dim
        if need_new_head:
            initial_out = len(class_names)
            if checkpoint and checkpoint.get("feature_dim") == feature_dim:
                saved_classes = checkpoint.get("class_names", [])
                saved_state = checkpoint.get("state_dict")
                head = SegmentationHead(feature_dim, max(len(saved_classes), 1))
                if saved_state and len(saved_classes) == head.classifier.out_channels:
                    state_dict = {}
                    for key, value in saved_state.items():
                        tensor = value
                        if not isinstance(tensor, torch.Tensor):
                            tensor = torch.tensor(tensor)
                        state_dict[key] = tensor
                    head.load_state_dict(state_dict, strict=True)
                checkpoint = None
            else:
                head = SegmentationHead(feature_dim, max(initial_out, 1))
        if head is None:
            print("[ERR] Failed to initialize segmentation head — pausing 1s…")
            time.sleep(1)
            continue
        if len(class_names) > head.classifier.out_channels:
            head.expand_out_channels(len(class_names))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head = head.to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

        train_class_indices = set()
        for sample in train_samples:
            for _, _, idx in sample.point_targets:
                train_class_indices.add(idx)
            for value in sample.mask_target.unique():
                val_idx = int(value.item())
                if val_idx != IGNORE_INDEX:
                    train_class_indices.add(val_idx)

        if len(train_class_indices) < 2:
            print("[INFO] Not enough classes in train split — skipping training this cycle")
            train_loss_avg = None
        else:
            epochs = 25
            batch_size = 4
            train_loss_total = 0.0
            train_steps = 0
            for _ in range(epochs):
                random.shuffle(train_samples)
                for start in range(0, len(train_samples), batch_size):
                    batch = train_samples[start : start + batch_size]
                    features = torch.stack([s.features for s in batch]).to(device)
                    logits = head(features)
                    mask_targets = torch.stack([s.mask_target for s in batch]).to(device)
                    mask_loss = None
                    if mask_loss_weight > 0.0 and (mask_targets != IGNORE_INDEX).any():
                        mask_loss = F.cross_entropy(logits, mask_targets, ignore_index=IGNORE_INDEX)
                    point_logits = []
                    point_targets = []
                    for bi, sample in enumerate(batch):
                        for fy, fx, cls_idx in sample.point_targets:
                            point_logits.append(logits[bi, :, fy, fx])
                            point_targets.append(cls_idx)
                    point_loss = None
                    if point_logits:
                        point_logits_tensor = torch.stack(point_logits)
                        point_targets_tensor = torch.tensor(point_targets, device=device)
                        point_loss = F.cross_entropy(point_logits_tensor, point_targets_tensor)
                    losses = []
                    if point_loss is not None:
                        losses.append(point_loss)
                    if mask_loss is not None:
                        losses.append(mask_loss_weight * mask_loss)
                    if not losses:
                        continue
                    total_loss = sum(losses)
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    train_loss_total += total_loss.item()
                    train_steps += 1
            train_loss_avg = train_loss_total / train_steps if train_steps else None

        val_accuracy = evaluate_accuracy(head, val_samples, device)
        backend_db.store_training_stats(cycle, train_loss_avg, None, val_accuracy)

        checkpoint_state = {
            "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "class_names": class_names,
            "feature_dim": head.classifier.in_channels,
            "architecture": model_size,
            "resize": current_resize,
            "mask_loss_weight": mask_loss_weight,
        }
        save_checkpoint(checkpoint_state, clf_path)

        labeled_set = {Path(item.filepath) for item in annotated}
        unlabeled = [
            s
            for s in samples
            if Path(s["sample_filepath"]) not in labeled_set
            and (allowed_ids is None or s["id"] in allowed_ids)
        ]
        filtered_annotated = [
            (item.sample_id, item.filepath)
            for item in annotated
            if allowed_ids is None or item.sample_id in allowed_ids
        ]
        to_predict = (
            [dict(id=s_id, sample_filepath=fp) for s_id, fp in filtered_annotated]
            + [s for s in unlabeled]
        )[:budget]
        print(f"[PRED] Will predict on {len(to_predict)} images (labeled + unlabeled, up to budget)")

        if head is None:
            print("[INFO] No trained head available — skipping prediction this cycle; pausing 1s…")
            time.sleep(1)
            continue

        Path("session/preds").mkdir(parents=True, exist_ok=True)
        head.eval()
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
            feats = feats.to(torch.float32)
            logits = head(feats.unsqueeze(0).to(device)).squeeze(0).cpu()
            classes_map = logits.argmax(dim=0).numpy()
            Fh, Fw = logits.shape[1:]
            image_rows_01 = new_h / max(new_w, new_h)
            image_cols_01 = new_w / max(new_w, new_h)
            if image_rows_01 < 1.0:
                classes_map = classes_map[: int(Fh * image_rows_01)]
            elif image_cols_01 < 1.0:
                classes_map = classes_map[:, : int(Fw * image_cols_01)]
            preds_batch = []
            for class_idx in np.unique(classes_map):
                class_idx_int = int(class_idx)
                if class_idx_int < 0 or class_idx_int >= len(class_names):
                    continue
                class_name = class_names[class_idx_int]
                mask_bool = (classes_map == class_idx_int)
                if not mask_bool.any():
                    continue
                safe_cls = _sanitize_for_filename(class_name)
                outpath = (Path("session") / "preds" / f"{s_id}_{safe_cls}.png").resolve()
                Image.fromarray(mask_bool.astype(np.uint8) * 255).save(outpath)
                preds_batch.append({
                    "sample_id": s_id,
                    "class": str(class_name),
                    "type": "mask",
                    "mask_path": outpath.as_posix(),
                })
            if preds_batch:
                backend_db.set_predictions_batch(preds_batch)
        head.train()
        print(
            f"[cycle {cycle}] trained on {len(train_samples)} imgs, val {len(val_samples)} imgs, predicted {len(to_predict)} images"
        )
        cycle += 1
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
