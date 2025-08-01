#!/usr/bin/env python
"""fastai continuous trainer‑predictor (stateful)

Loops forever **continuing** training on the *same* model rather than starting
from scratch each cycle.

Cycle logic
-----------
1. Re‑scan DB → build/refresh ``DataLoaders`` from latest *label* annotations.
2. Fit the existing learner **one epoch** (weights accumulate over cycles).
3. Measure epoch wall‑clock time *T* → predict on ``budget = 2 × T`` *unlabeled*
   images; write predictions via ``DatabaseAPI.set_predictions``.
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
import random
import signal
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

# ––– third‑party –––
import torch
from fastai.vision.all import (
    ImageDataLoaders,
    PILImage,
    Resize,
    accuracy,
    resnet18,
    resnet34,
    vision_learner,
)

# ––– local –––
try:
    from src.database.data import DatabaseAPI
except ModuleNotFoundError:  # script run from repo root
    _root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_root))
    from database.data import DatabaseAPI  # type: ignore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gather_training_items(db: DatabaseAPI) -> List[Tuple[str, str]]:
    """Latest *label* per image as ``(filepath, class)`` list."""
    items: List[Tuple[str, str]] = []
    for fp in db.get_samples():
        labels = [a for a in db.get_annotations(fp) if a.get("type") == "label" and a.get("class")]
        if labels:
            latest = max(labels, key=lambda a: a.get("timestamp", 0))
            items.append((fp, latest["class"]))
    return items


def _build_dls(paths: Sequence[str], labels: Sequence[str]):
    """Create ``DataLoaders`` with fixed transforms so we can reuse each cycle."""
    return ImageDataLoaders.from_lists(
        Path("."), list(paths), list(labels), valid_pct=0.20, seed=42, bs=64, item_tfms=Resize(64)
    )


def _predict_subset(db: DatabaseAPI, learner, unlabeled: List[str], budget: int) -> int:
    """Predict for *budget* images, return #predicted."""
    if budget <= 0 or not unlabeled:
        return 0
    count = 0
    for fp in unlabeled[:budget]:
        pred_class, pred_idx, probs = learner.predict(PILImage.create(fp))
        db.set_predictions(
            fp,
            [
                {
                    "sample_filepath": fp,
                    "type": "label",
                    "class": str(pred_class),
                    "probability": float(probs[pred_idx]),
                }
            ],
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def _run_forever(db_path: str | None, arch: str, sleep_s: int) -> None:
    db = DatabaseAPI(db_path)

    def _exit_handler(*_):
        print("\n[INFO] Exiting…")
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)

    model_arch = resnet18 if arch == "small" else resnet34
    learner = None  # will lazily instantiate when we first have data
    model_path = Path(db.db_path).with_suffix(".pth")
    cycle = 0

    while True:
        cycle += 1
        try:
            train_items = _gather_training_items(db)
            if not train_items:
                print("[WARN] No label annotations available — sleeping…")
                time.sleep(max(1, sleep_s))
                continue

            paths, labels = zip(*train_items)
            dls = _build_dls(paths, labels)

            if learner is None:
                learner = vision_learner(dls, model_arch, metrics=accuracy)
                if model_path.exists():
                    learner.model.load_state_dict(torch.load(model_path))
            else:
                # re‑attach fresh DataLoaders to keep dataset up‑to‑date
                learner.dls = dls

            t0 = time.time()
            learner.fit(1)
            epoch_time = time.time() - t0

            try:
                valid_res = learner.validate()
                valid_loss = float(valid_res[0]) if len(valid_res) > 0 else None
                accuracy_val = float(valid_res[1]) if len(valid_res) > 1 else None
                train_loss = float(learner.recorder.losses[-1]) if learner.recorder.losses else None
            except Exception:
                train_loss = valid_loss = accuracy_val = None

            db.add_training_stat(cycle, train_loss, valid_loss, accuracy_val)
            # Save model checkpoint after each epoch
            try:
                torch.save(learner.model.state_dict(), model_path)
            except Exception as e:
                print(f"[WARN] Failed to save model checkpoint: {e}")

            budget = max(10, int(2 * epoch_time))
            labeled_set = set(paths)
            unlabeled = [fp for fp in db.get_samples() if fp not in labeled_set]
            predicted_n = _predict_subset(db, learner, unlabeled, budget)

            print(
                f"[cycle {cycle}] epoch {epoch_time:.1f}s — predicted {predicted_n}/{len(unlabeled)} "
                f"unlabeled — sleeping {sleep_s}s"
            )
        except Exception as e:
            print(f"[ERR][cycle {cycle}] {e}", file=sys.stderr)
        finally:
            torch.cuda.empty_cache()
            time.sleep(max(1, sleep_s))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", "-d", help="Path to annotation SQLite db", default=None)
    p.add_argument("--arch", "-a", choices=["default", "small"], default="default", help="CNN size")
    p.add_argument("--sleep", "-s", type=int, default=0, help="Seconds between cycles")
    args = p.parse_args()

    _run_forever(args.db, args.arch, args.sleep)


if __name__ == "__main__":
    main()
