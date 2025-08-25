#!/usr/bin/env python
"""DINOv3-based Point Classification Training and Inference

This script implements a continuous training loop for point-based image classification using
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
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
from typing import List, Sequence, Tuple, Dict, Optional


import torch
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight

# ––– local imports –––
try:
    from src.backend import db as backend_db
except ModuleNotFoundError:  # script run from repo root
    _root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_root))
    from backend import db as backend_db  # type: ignore


def resize_pad(im: Image.Image, target_size: int = 1536) -> Image.Image:
    """Resize image with largest side to target_size and zero-pad to square for batching."""
    from PIL import ImageOps
    
    w, h = im.size
    
    # Scale based on the largest dimension
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Ensure dimensions are multiples of 16 (required by DINOv3)
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    # Resize the image
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create a square canvas with target_size (ensuring it's multiple of 16)
    canvas_size = (target_size // 16) * 16
    
    # Use PIL's pad function to center and pad with zeros (black)
    padded = ImageOps.pad(resized, (canvas_size, canvas_size), color=(0, 0, 0), centering=(0, 0))
    
    return padded


def normalize_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor for DINOv3."""
    x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0)
    # ImageNet normalization
    x = (x / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x


def gather_annotated_items(samples: Sequence[dict]) -> List[Tuple[str, Dict[str, List[dict]]]]:
    """Get images with point annotations as (filepath, annotations_dict) list."""
    items: List[Tuple[str, dict]] = []
    for s in samples:
        anns = backend_db.get_annotations(s["id"])
        points = [a for a in anns if a.get("type") == "point" and a.get("col") is not None and a.get("row") is not None]
        if points:
            # Group points by class
            points_by_class = {}
            for point in points:
                class_name = point.get("class", "unknown")
                if class_name not in points_by_class:
                    points_by_class[class_name] = []
                points_by_class[class_name].append({
                    "x": point["col"],  # Map col to x
                    "y": point["row"],  # Map row to y
                    "timestamp": point.get("timestamp", 0)
                })
            items.append((s["sample_filepath"], points_by_class))
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
    
    # Get the correct path to the dinov3 directory
    dinov3_path = Path(__file__).parent / "dinov3"
    if not dinov3_path.exists():
        raise FileNotFoundError(f"DINOv3 directory not found at {dinov3_path}")
    
    # Check if weights file exists
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[WARN] Weights file not found: {weights_path}")
        print("[INFO] Loading model without local weights")
        weights = None
    
    model = torch.hub.load(
        repo_or_dir=str(dinov3_path),
        model=model_name,
        source="local",
        weights=weights,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Ensure model is in evaluation mode
    
    # Disable gradients for inference
    for param in model.parameters():
        param.requires_grad = False
        
    return model


def extract_features(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """Extract DINOv3 features from image tensor(s).
    
    Args:
        model: DINOv3 model in eval mode
        image_tensor: Input tensor of shape (B, C, H, W) or (C, H, W)
    
    Returns:
        features: Tensor of shape (F, Ph, Pw) for single image or (B, F, Ph, Pw) for batch
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Handle single image case by adding batch dimension
    single_image = False
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        single_image = True
    
    batch_size = image_tensor.shape[0]
    
    with torch.no_grad():
        features = model.forward_features(image_tensor)["x_norm_patchtokens"]  # (B, Ph x Pw, F)
        
        # Reshape features to spatial format
        Ph = int(image_tensor.shape[2] / 16)
        Pw = int(image_tensor.shape[3] / 16)
        F = features.shape[-1]
        
        # Reshape from (B, Ph*Pw, F) to (B, F, Ph, Pw)
        features = features.permute(0, 2, 1).reshape(batch_size, F, Ph, Pw)
        
        # If single image, remove batch dimension
        if single_image:
            features = features[0]  # (F, Ph, Pw)
    
    return features


def process_images_batch(model: torch.nn.Module, image_paths: List[str], target_size: int = 1536, batch_size: int = 4) -> List[Tuple[str, torch.Tensor]]:
    """Process multiple images in batches for efficiency.
    
    Args:
        model: DINOv3 model
        image_paths: List of image file paths
        target_size: Target size for resizing
        batch_size: Number of images to process at once
        
    Returns:
        List of (filepath, features) tuples
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        valid_paths = []
        
        # Load and preprocess batch
        for path in batch_paths:
            try:
                image = Image.open(path)
                image_resized = resize_pad(image, target_size)
                image_tensor = normalize_image(image_resized)
                batch_tensors.append(image_tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")
                continue
        
        if not batch_tensors:
            continue
            
        # Stack tensors into batch
        batch_tensor = torch.cat(batch_tensors, dim=0)  # (B, C, H, W)
        
        # Extract features for the batch
        batch_features = extract_features(model, batch_tensor)  # (B, F, Ph, Pw)
        
        # Add results
        for j, path in enumerate(valid_paths):
            results.append((path, batch_features[j]))  # (F, Ph, Pw)
    
    return results



def run_continuous_processing(model_size: str = "small", sleep_seconds: int = 30) -> None:
    """Continuously process new annotations, train and apply a linear classifier on DINOv3 features."""
    print(f"[INFO] Starting continuous processing (checking every {sleep_seconds}s)")
    model = load_dinov3_model(model_size)
    classifier_path = Path("dinov3_linear_classifier.joblib")
    prev_config: Optional[dict] = None
    classifier: Optional[LogisticRegression] = None
    cycle = 0
    
    # Try to load previous classifier
    if classifier_path.exists():
        try:
            classifier = joblib.load(classifier_path)
            print(f"[INFO] Loaded previous classifier from {classifier_path}")
        except Exception as e:
            print(f"[WARN] Could not load previous classifier: {e}")
            classifier = None
    
    while True:
        try:
            config = backend_db.get_config()
        except Exception as e:
            print(f"[ERR] failed to load config: {e}")
            time.sleep(5)
            continue

        if prev_config != config:
            if prev_config is not None:
                print("[INFO] Config changed; resetting classifier")
            classifier = None
            try:
                if classifier_path.exists():
                    classifier_path.unlink()
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

        # 1. Gather annotated samples
        samples = backend_db.get_all_samples()
        annotated_items = gather_annotated_items(samples)
        if not annotated_items:
            print("[WARN] No point annotations available — sleeping…")
            time.sleep(max(1, sleep_s))
            continue

        # 2. Build feature/label dataset from annotated points
        X, y, img_ids, img_splits = [], [], [], []
        img_to_idx = {}
        for i, (filepath, points_by_class) in enumerate(annotated_items):
            img_to_idx[filepath] = i
        
        # Precompute all features for annotated images
        features_per_image = {}
        for filepath, points_by_class in annotated_items:
            image_path = Path(filepath)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {filepath}")
                continue
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            image_resized = resize_pad(image)
            image_tensor = normalize_image(image_resized)
            features = extract_features(model, image_tensor)  # (F, 96, 96)
            features_per_image[filepath] = (features, orig_w, orig_h)

        # Build dataset: each point is a sample
        for filepath, points_by_class in annotated_items:
            features, orig_w, orig_h = features_per_image[filepath]
            for class_name, points in points_by_class.items():
                for pt in points:
                    # Map original (x, y) to 1536x1536, then to 96x96
                    x_scaled = pt["x"] * (1536 / orig_w)
                    y_scaled = pt["y"] * (1536 / orig_h)
                    fx = int(round(x_scaled / 16))
                    fy = int(round(y_scaled / 16))
                    fx = min(max(fx, 0), 95)
                    fy = min(max(fy, 0), 95)
                    feat_vec = features[:, fy, fx].cpu().numpy()  # (F,)
                    X.append(feat_vec)
                    y.append(class_name)
                    img_ids.append(filepath)

        if not X:
            print("[WARN] No valid feature/label pairs found — sleeping…")
            time.sleep(max(1, sleep_s))
            continue

        # 3. Split by images: 80% train, 20% val
        unique_imgs = list({fp for fp in img_ids})
        np.random.seed(42)
        np.random.shuffle(unique_imgs)
        n_train = int(0.8 * len(unique_imgs))
        train_imgs = set(unique_imgs[:n_train])
        val_imgs = set(unique_imgs[n_train:])
        train_idx = [i for i, fp in enumerate(img_ids) if fp in train_imgs]
        val_idx = [i for i, fp in enumerate(img_ids) if fp in val_imgs]

        X_train = np.stack([X[i] for i in train_idx])
        y_train = [y[i] for i in train_idx]
        X_val = np.stack([X[i] for i in val_idx]) if val_idx else None
        y_val = [y[i] for i in val_idx] if val_idx else None

        # 4. Train or update classifier
        if classifier is None:
            print(f"[INFO] Initializing new classifier with {len(X_train)} samples")
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            classifier = LogisticRegression(max_iter=1000, class_weight=dict(zip(classes, class_weights)), warm_start=True)
            classifier.fit(X_train, y_train)
        else:
            print(f"[INFO] Updating classifier with {len(X_train)} samples")
            classifier.fit(X_train, y_train)

        # Save classifier
        joblib.dump(classifier, classifier_path)

        # 5. Validation
        if X_val is not None and len(y_val) > 0:
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

        # Log training stats
        backend_db.store_training_stats(cycle, None, loss, acc)

        # 6. Dense prediction for annotated and unlabeled images
        # Get all images (annotated + unlabeled)
        labeled_set = set(fp for fp, _ in annotated_items)
        unlabeled = [s for s in samples if s["sample_filepath"] not in labeled_set]
        # Sequential order, up to budget
        to_predict = [fp for fp, _ in annotated_items] + [s["sample_filepath"] for s in unlabeled[:budget]]

        for fp in to_predict:
            image_path = Path(fp)
            if not image_path.exists():
                print(f"[WARN] Image file not found: {fp}")
                continue
            image = Image.open(image_path)
            orig_w, orig_h = image.size
            image_resized = resize_pad(image)
            image_tensor = normalize_image(image_resized)
            features = extract_features(model, image_tensor)  # (F, 96, 96)
            # Dense prediction
            F, H, W = features.shape
            features_np = features.permute(1, 2, 0).reshape(-1, F).cpu().numpy()  # (H*W, F)
            pred_labels = classifier.predict(features_np)
            pred_probs = None
            try:
                pred_probs = classifier.predict_proba(features_np)
            except Exception:
                pass
            # Reshape to (H, W)
            pred_labels_map = np.array(pred_labels).reshape(H, W)
            # Optionally, store dense map or sample points
            # Here, we store sampled points (center of each patch)
            predictions_batch = []
            for y_idx in range(H):
                for x_idx in range(W):
                    cx = int((x_idx + 0.5) * 16 * (orig_w / 1536))
                    cy = int((y_idx + 0.5) * 16 * (orig_h / 1536))
                    pred = {
                        "type": "label",
                        "class": str(pred_labels_map[y_idx, x_idx]),
                    }
                    if pred_probs is not None:
                        pred["probability"] = float(np.max(pred_probs[y_idx * W + x_idx]))
                    predictions_batch.append((None, [pred, {"x": cx, "y": cy}]))
            # Store predictions in DB (batch)
            backend_db.set_predictions_batch(predictions_batch)

        print(f"[cycle {cycle}] trained on {len(X_train)} pts, val {len(y_val) if y_val else 0} pts, predicted {len(to_predict)} images — sleeping {sleep_s}s")
        cycle += 1
        torch.cuda.empty_cache()
        time.sleep(max(1, sleep_s))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DINOv3 feature extraction for point annotations")
    parser.add_argument(
        "--size", 
        choices=["small", "large"], 
        default="small",
        help="Model size (default: small)"
    )
    parser.add_argument(
        "--continuous", 
        action="store_true",
        help="Run in continuous mode, checking for new annotations"
    )
    parser.add_argument(
        "--sleep", 
        type=int, 
        default=30,
        help="Sleep time between checks in continuous mode (default: 30s)"
    )
    parser.add_argument(
        "--db", 
        help="Path to annotation SQLite db", 
        default=None
    )
    
    args = parser.parse_args()
    
    if args.db:
        backend_db.DB_PATH = args.db
    
    if args.continuous:
        run_continuous_processing(args.size, args.sleep)
    else:
        print("[INFO] Use --continuous flag to start processing annotations")
        print("[INFO] This script now only runs in continuous mode")


if __name__ == "__main__":
    main()
