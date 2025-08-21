#!/usr/bin/env python
"""fastai continuous trainer‑predictor (stateful)

Loops forever **continuing** training on the *same* model rather than starting
from scratch each cycle.

Cycle logic
-----------
1. Re‑scan DB → build/refresh ``DataLoaders`` from latest *label* annotations.
2. Fit the existing learner **one epoch** (weights accumulate over cycles).
3. Measure epoch wall‑clock time *T* → predict on ``budget = 2 × T`` *unlabeled*
   images; write predictions via the database API.
4. Sleep and repeat.

Model size flag
~~~~~~~~~~~~~~~
* ``--arch default`` ⇒ `resnet34`
* ``--arch small``   ⇒ `resnet18` (nice for MNIST)

Usage example
~~~~~~~~~~~~~
```bash
python -m src.ml.fastai_training --arch small --sleep 10
```

The process handles Ctrl‑C/SIGTERM gracefully.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

# ––– third‑party –––
import torch
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    aug_transforms,
    accuracy,
    load_learner,
    resnet18,
    resnet34,
    vision_learner,
)

# ––– local –––
try:
    from src.new_backend import db as backend_db
except ModuleNotFoundError:  # script run from repo root
    _root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_root))
    from new_backend import db as backend_db  # type: ignore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _gather_training_items(samples: Sequence[dict]) -> List[Tuple[str, str]]:
    """Latest *label* per image as ``(filepath, class)`` list."""
    items: List[Tuple[str, str]] = []
    for s in samples:
        anns = backend_db.get_annotations(s["id"])
        labels = [a for a in anns if a.get("type") == "label" and a.get("class")]
        if labels:
            latest = max(labels, key=lambda a: a.get("timestamp", 0))
            items.append((s["sample_filepath"], latest["class"]))
    return items


def _build_dls(
    paths: Sequence[str],
    labels: Sequence[str],
    resize: int,
    flip: bool,
    max_rotate: float,
):
    """Create ``DataLoaders`` with fixed transforms so we can reuse each cycle."""
    batch_tfms = aug_transforms(do_flip=flip, max_rotate=max_rotate)
    return ImageDataLoaders.from_lists(
        Path("."),
        list(paths),
        list(labels),
        valid_pct=0.20,
        seed=42,
        bs=16,
        item_tfms=Resize(resize),
        batch_tfms=batch_tfms,
    )


def _predict_subset(learner, unlabeled: List[dict], budget: int) -> int:
    """Predict for *budget* samples, return #predicted."""
    if budget <= 0 or not unlabeled:
        return 0
    subset = unlabeled[:budget]
    dl = learner.dls.test_dl(
        [s["sample_filepath"] for s in subset], bs=learner.dls.bs
    )
    preds, _, decoded = learner.get_preds(dl=dl, with_decoded=True)

    for sample, pred_tensor, pred_class in zip(subset, preds, decoded):
        pred_idx = int(pred_tensor.argmax())
        backend_db.set_predictions(
            sample["id"],
            [
                {
                    "type": "label",
                    "class": str(learner.dls.vocab[pred_class.item()]),
                    "probability": float(pred_tensor[pred_idx]),
                }
            ],
        )

    return len(subset)


# ---------------------------------------------------------------------------
# learner management
# ---------------------------------------------------------------------------


def _init_or_update_learner(dls, model_arch, model_path: Path, existing_learner):
    """Return a ``Learner`` initialized or updated for the current cycle.

    Responsibilities:
    * Load a previously exported learner from ``model_path`` when available.
    * Re‑initialize the model if the set of classes has changed since the last
      cycle.
    * Attach fresh ``DataLoaders`` to keep the dataset in sync.
    """
    new_classes = set(dls.vocab)
    learner = existing_learner
    if learner is None:
        if model_path.exists():
            try:
                learner = load_learner(model_path)
                learner.dls = dls
            except Exception as e:
                print(f"[WARN] Failed to load exported learner: {e}")
                learner = vision_learner(dls, model_arch, metrics=accuracy)
        else:
            learner = vision_learner(dls, model_arch, metrics=accuracy)
    else:
        current_classes = set(learner.dls.vocab)
        if new_classes != current_classes:
            print(
                f"[INFO] Detected class change {current_classes} -> {new_classes}; resetting model"
            )
            learner = vision_learner(dls, model_arch, metrics=accuracy)
        else:
            # re‑attach fresh DataLoaders to keep dataset up‑to‑date
            learner.dls = dls
    return learner


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def _run_forever(flip: bool, max_rotate: float) -> None:

    def _exit_handler(*_):
        print("\n[INFO] Exiting…")
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)

    learner = None
    model_path = Path(backend_db.DB_PATH).with_name("checkpoint.pkl")
    prev_config: dict | None = None
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
                print("[INFO] Config changed; resetting learner")
            learner = None
            cycle = 0
            try:
                if model_path.exists():
                    model_path.unlink()
            except Exception:
                pass
            prev_config = config

        sleep_s = config.get("sleep", 5) or 5
        if not config.get("ai_should_be_run", False):
            print("[INFO] Run flag disabled — sleeping…")
            time.sleep(max(1, sleep_s))
            continue

        arch = config.get("architecture", "resnet18")
        model_arch = resnet18 if arch == "resnet18" else (
            resnet34 if arch == "resnet34" else arch
        )
        budget = config.get("budget", 1000)
        resize = config.get("resize", 64)

        try:
            samples = backend_db.get_all_samples()
            train_items = _gather_training_items(samples)
            if not train_items:
                print("[WARN] No label annotations available — sleeping…")
                time.sleep(max(1, sleep_s))
                continue

            paths, labels = zip(*train_items)
            dls = _build_dls(paths, labels, resize, flip, max_rotate)
            learner = _init_or_update_learner(dls, model_arch, model_path, learner)

            t0 = time.time()
            learner.fit(1)
            epoch_time = time.time() - t0

            try:
                valid_res = learner.validate()
                valid_loss = float(valid_res[0]) if len(valid_res) > 0 else None
                accuracy_val = float(valid_res[1]) if len(valid_res) > 1 else None
                train_loss = (
                    float(learner.recorder.losses[-1])
                    if learner.recorder.losses
                    else None
                )
            except Exception:
                train_loss = valid_loss = accuracy_val = None

            backend_db.store_training_stats(cycle, train_loss, valid_loss, accuracy_val)

            try:
                learner.export(model_path)
            except Exception as e:
                print(f"[WARN] Failed to export learner: {e}")

            labeled_set = set(paths)
            unlabeled = [s for s in samples if s["sample_filepath"] not in labeled_set]
            predicted_n = _predict_subset(learner, unlabeled, budget)

            print(
                f"[cycle {cycle}] epoch {epoch_time:.1f}s — predicted {predicted_n}/{len(unlabeled)} "
                f"unlabeled — sleeping {sleep_s}s"
            )
            cycle += 1
        except Exception as e:
            print(f"[ERR][cycle {cycle}] {e}", file=sys.stderr)
        finally:
            torch.cuda.empty_cache()
            time.sleep(max(1, sleep_s))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", "-d", help="Path to annotation SQLite db", default=None)
    p.add_argument(
        "--no-flip",
        action="store_false",
        dest="flip",
        help="Disable random horizontal flips",
    )
    p.add_argument(
        "--max-rotate",
        type=float,
        default=10.0,
        help="Maximum rotation for data augmentation",
    )
    p.set_defaults(flip=True)
    args = p.parse_args()

    if args.db:
        backend_db.DB_PATH = args.db

    _run_forever(
        args.flip,
        args.max_rotate,
    )


if __name__ == "__main__":
    main()
