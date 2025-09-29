#!/usr/bin/env python
"""DINOv3-based Image Classification with a Linear Head (balanced LR) + RAM cache

What it does
------------
- Pull latest *label* per image from DB.
- Frozen DINOv3 backbone → one global feature per image (CLS or AVG over patches).
- Balanced LogisticRegression trained fresh each cycle.
- Predicts on unlabeled images, writes via set_predictions_batch.
- Logs accuracy via store_training_stats.
- RAM LRU cache for per-image global features so we don’t recompute.

Config (backend_db.get_config)
------------------------------
- task: "classification" to run
- ai_should_be_run: bool
- architecture: "small" | "large"  (vits16 | vitl16)
- resize: int (default 384)
- pooling: "avg" | "cls" (default "avg")
- debug_forward: bool (default False)  # triggers breakpoint after forward_features
- budget: int (default 1000)

TODOs for you
-------------
- Point DINO repo/weights in `load_dinov3_model`.
- Ensure DB API functions exist: get_config, get_all_samples, get_annotations,
  set_predictions_batch, store_training_stats, and DB_PATH.
"""

from __future__ import annotations

import sys
import time
import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from collections import OrderedDict
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
# simple RAM LRU for global features (per image)
# ---------------------------------------------------------------------------

FEAT_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
FEAT_CACHE_MAX_ITEMS = 20000  # global vectors are tiny; bump as you like

def _cache_key(fp: str, resize: int, pooling: str, arch: str) -> str:
    try:
        st = Path(fp).stat()
        mtime = int(st.st_mtime)
        size = int(st.st_size)
    except FileNotFoundError:
        mtime = -1
        size = -1
    return f"{fp}|{mtime}|{size}|{resize}|{pooling}|{arch}"

def _cache_get(key: str):
    vec = FEAT_CACHE.get(key)
    if vec is not None:
        # move to end to mark as recently used
        FEAT_CACHE.move_to_end(key)
    return vec

def _cache_put(key: str, vec: np.ndarray):
    FEAT_CACHE[key] = vec
    FEAT_CACHE.move_to_end(key)
    # simple LRU eviction
    while len(FEAT_CACHE) > FEAT_CACHE_MAX_ITEMS:
        FEAT_CACHE.popitem(last=False)

def _cache_clear():
    FEAT_CACHE.clear()


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def _resize_pad_longer_side(im: Image.Image, target_size: int) -> Image.Image:
    """Resize so longer side == target_size, then zero-pad to square."""
    from PIL import ImageOps
    w, h = im.size
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (new_h // 16) * 16)
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = ImageOps.pad(resized, (target_size, target_size), color=(0, 0, 0), centering=(0, 0))
    return padded

def _normalize_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor (1,C,H,W) with ImageNet normalization."""
    x = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float()
    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# backbone
# ---------------------------------------------------------------------------

def load_dinov3_model(size: str = "small") -> torch.nn.Module:
    """Load local DINOv3 model and freeze it.
    TODO: set paths to your repo/weights."""
    dinov3_repo_dir = Path(__file__).parent / "dinov3"  # TODO
    if size == "large":
        model_name = "dinov3_vitl16"
        weights_path = Path("src/ml/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")  # TODO
    else:
        model_name = "dinov3_vits16"
        weights_path = Path("src/ml/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")  # TODO

    if not dinov3_repo_dir.exists():
        raise FileNotFoundError(f"DINOv3 repo not found at {dinov3_repo_dir.resolve()}")

    weights = str(weights_path) if weights_path.exists() else None
    if weights is None:
        print(f"[WARN] Weights not found at {weights_path}. Loading without explicit weights.")

    model = torch.hub.load(
        repo_or_dir=str(dinov3_repo_dir),
        model=model_name,
        source="local",
        weights=weights,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def _extract_global_feature_tensor(
    model: torch.nn.Module,
    image_rgb: Image.Image,
    target_size: int,
    pooling: str = "avg",
    debug_forward: bool = False,
) -> np.ndarray:
    """Core: run forward_features and pool tokens into one 1D vector."""
    device = next(model.parameters()).device
    padded = _resize_pad_longer_side(image_rgb, target_size)
    x = _normalize_to_tensor(padded).to(device)

    feats: Dict[str, torch.Tensor] = model.forward_features(x)  # type: ignore[attr-defined]
    if debug_forward:
        breakpoint()  # Inspect feats.keys(), shapes

    if pooling == "cls":
        cls_tok = feats.get("x_norm_clstoken", None)
        if cls_tok is None:
            raise KeyError("Missing 'x_norm_clstoken' in forward_features output.")
        vec = cls_tok[0]  # (D,)
    else:
        patch = feats.get("x_norm_patchtokens", None)
        if patch is None:
            raise KeyError("Missing 'x_norm_patchtokens' in forward_features output.")
        vec = patch.mean(dim=1)[0]  # (D,)

    return vec.detach().float().cpu().numpy()


def extract_global_feature_cached(
    model: torch.nn.Module,
    image_path: str,
    target_size: int,
    pooling: str,
    debug_forward: bool,
    arch: str,
) -> np.ndarray:
    """LRU-cached wrapper: key on file path, mtime, size, resize, pooling, arch."""
    key = _cache_key(image_path, target_size, pooling, arch)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    p = Path(image_path)
    with Image.open(p) as im:
        vec = _extract_global_feature_tensor(
            model, im, target_size=target_size, pooling=pooling, debug_forward=debug_forward
        )
    # store as float32 numpy
    vec = np.asarray(vec, dtype=np.float32)
    _cache_put(key, vec)
    return vec


# ---------------------------------------------------------------------------
# dataset building from DB
# ---------------------------------------------------------------------------

def _gather_latest_labels(samples: Sequence[dict]) -> List[Tuple[int, str, str]]:
    """Return (sample_id, filepath, class) using latest *label* per image."""
    items: List[Tuple[int, str, str]] = []
    for s in samples:
        anns = backend_db.get_annotations(s["id"])
        labels = [a for a in anns if a.get("type") == "label" and a.get("class")]
        if labels:
            latest = max(labels, key=lambda a: a.get("timestamp") or 0)
            items.append((s["id"], s["sample_filepath"], str(latest["class"])))
    return items

def _train_val_split_by_image(items: List[Tuple[int, str, str]], valid_pct: float = 0.2, seed: int = 42):
    """Split image-level items into train/val without leakage."""
    rng = random.Random(seed)
    ids = list({sid for sid, _, _ in items})
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * valid_pct))) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    train, val = [], []
    for tup in items:
        (val if tup[0] in val_ids else train).append(tup)
    return train, val


# ---------------------------------------------------------------------------
# classifier persistence
# ---------------------------------------------------------------------------

def _load_clf(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)

def _save_clf(clf, path: Path):
    with path.open("wb") as f:
        pickle.dump(clf, f)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("[INIT] DINOv3 linear classifier trainer (balanced LR, cached)")
    clf_path = Path(backend_db.DB_PATH).with_name("dinov3_linear_cls.pkl")
    cycle = 0
    prev_config = None
    model = None
    model_size = None

    while True:
        config = backend_db.get_config()

        if (config.get("task") or "classification").lower() != "classification":
            print("[INFO] Task is not 'classification' — pausing 1s…")
            time.sleep(1)
            continue
        if not config.get("ai_should_be_run", False):
            print("[INFO] Run flag disabled — pausing 2s…")
            time.sleep(2)
            continue

        arch = (config.get("architecture") or "small").lower()
        resize = int(config.get("resize", 384))
        pooling = (config.get("pooling") or "avg").lower()
        debug_forward = bool(config.get("debug_forward", False))
        budget = int(config.get("budget", 1000))

        # reload and clear cache if config changed in a way that affects features
        if model is None or arch != model_size or config != prev_config:
            if arch not in {"small", "large"}:
                print(f"[WARN] Unknown architecture '{arch}', defaulting to 'small'")
                arch = "small"
            print(f"[INFO] Loading DINOv3 backbone: {arch}")
            model = load_dinov3_model(arch)
            model_size = arch
            prev_config = config
            _cache_clear()
            print("[CACHE] Cleared feature cache due to config change.")

        print("[DB] Fetching samples…")
        samples = backend_db.get_all_samples()
        allowed_ids = backend_db.get_sample_ids_for_path_filter(
            config.get("sample_path_filter")
        )
        labeled = _gather_latest_labels(samples)
        if not labeled:
            print("[WARN] No label annotations found — pausing 2s…")
            time.sleep(2)
            continue

        # Extract features with cache
        train_items, val_items = _train_val_split_by_image(labeled, valid_pct=0.2, seed=42)
        X_train, y_train, X_val, y_val = [], [], [], []

        def _encode_items(items, X_out, y_out):
            for _, fp, cls in items:
                if not Path(fp).exists():
                    raise FileNotFoundError(f"Missing image: {fp}")
                vec = extract_global_feature_cached(
                    model, fp, target_size=resize, pooling=pooling,
                    debug_forward=debug_forward, arch=arch
                )
                X_out.append(vec)
                y_out.append(cls)

        print(f"[FEAT] Encoding {len(train_items)} train, {len(val_items)} val images…")
        _encode_items(train_items, X_train, y_train)
        _encode_items(val_items, X_val, y_val)

        if len(set(y_train)) < 2:
            print("[INFO] Need at least 2 classes to train — pausing 2s…")
            time.sleep(2)
            continue

        X_train_np = np.stack(X_train) if X_train else None
        X_val_np = np.stack(X_val) if X_val else None
        y_train_np = np.array(y_train) if y_train else None
        y_val_np = np.array(y_val) if y_val else None

        clf = LogisticRegression(class_weight="balanced", max_iter=500)
        clf.fit(X_train_np, y_train_np)
        print(f"[TRAIN] Trained LR on {X_train_np.shape[0]} images, {len(clf.classes_)} classes.")

        acc = None
        if X_val_np is not None and len(y_val_np) > 0:
            y_pred = clf.predict(X_val_np)
            acc = float(accuracy_score(y_val_np, y_pred))
            print(f"[VAL] Accuracy: {acc:.3f} | n_val: {len(y_val_np)}")

        _save_clf(clf, clf_path)
        backend_db.store_training_stats(cycle, None, None, acc)

        # Predict unlabeled using cache too
        labeled_paths = set(fp for _, fp, _ in labeled)
        unlabeled = [
            s
            for s in samples
            if s["sample_filepath"] not in labeled_paths
            and (allowed_ids is None or s["id"] in allowed_ids)
        ]
        if unlabeled:
            subset = unlabeled[: max(0, budget)]
            print(f"[PRED] Predicting {len(subset)}/{len(unlabeled)} unlabeled images…")

            predictions_batch = []
            for s in subset:
                fp = s["sample_filepath"]
                if not Path(fp).exists():
                    raise FileNotFoundError(f"Missing image: {fp}")
                vec = extract_global_feature_cached(
                    model, fp, target_size=resize, pooling=pooling,
                    debug_forward=False, arch=arch
                )

                probs = clf.predict_proba([vec])[0]
                top_idx = int(np.argmax(probs))
                top_cls = str(clf.classes_[top_idx])
                top_p = float(probs[top_idx])

                predictions = [
                    {
                        "type": "label",
                        "class": top_cls,
                        "probability": top_p,
                    }
                ]
                predictions_batch.append((s["id"], predictions))

            if predictions_batch:
                backend_db.set_predictions_batch(predictions_batch)

        print(f"[cycle {cycle}] cache_size={len(FEAT_CACHE)} — pausing 5s")
        cycle += 1
        torch.cuda.empty_cache()
        time.sleep(5)


if __name__ == "__main__":
    main()
