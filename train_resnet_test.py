import hashlib
import json
import random
from pathlib import Path

import numpy as np

from config import BATCH_SIZE, FEATURES_DIR
from data_loader import load_combined_dataset, create_data_loaders
from feature_extractor import AudioFeatureExtractor
from model import get_model
from train import Trainer


# ---- Run knobs ------------------------------------------------------------
EPOCHS          = 10      # rough-signal run: enough to see if ResNet is competitive
SUBSAMPLE_FRAC  = None    # fraction of the dataset to actually use (None = all)
BINARY          = True    # binary (real vs AI) vs multi-class
SEED            = 42
RESULTS_DIR     = Path("results") / "resnet_test"
FEATURE_CACHE_DIR = FEATURES_DIR / "cnn_mel_cache"   # delete this dir to force re-extraction
# ---------------------------------------------------------------------------


class CachingFeatureExtractor:
    

    def __init__(self, cache_dir=FEATURE_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._extractor = AudioFeatureExtractor()

    def _cache_path(self, file_path):
        key = hashlib.md5(str(Path(file_path).resolve()).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.npy"

    def extract_for_cnn(self, file_path):
        cp = self._cache_path(file_path)
        if cp.exists():
            try:
                return np.load(cp)
            except Exception:
                pass  # corrupt/partial cache entry — fall through and re-extract

        features = self._extractor.extract_for_cnn(file_path)
        if features is not None:
            try:
                np.save(cp, features)
            except Exception:
                pass  # cache write failure shouldn't break training
        return features


def subsample(file_paths, labels, frac, seed=SEED):
    if not frac or frac >= 1.0:
        return file_paths, labels
    rng = random.Random(seed)
    idx = list(range(len(file_paths)))
    rng.shuffle(idx)
    keep = idx[: max(1, int(len(idx) * frac))]
    keep.sort()
    return [file_paths[i] for i in keep], [labels[i] for i in keep]


def main():
    print("=" * 60)
    print("VOCALUITY - ResNet ROUGH-SIGNAL RUN (short training, cached features)")
    print("=" * 60)

    try:
        file_paths, labels, label_map = load_combined_dataset(binary=BINARY)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    file_paths, labels = subsample(file_paths, labels, SUBSAMPLE_FRAC)
    print(f"\nUsing {len(file_paths)} samples (subsample_frac={SUBSAMPLE_FRAC})")

    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        file_paths, labels, batch_size=BATCH_SIZE
    )

    # Swap each dataset's extractor for the caching one BEFORE any iteration,
    # so the (Windows-spawned) workers pickle the caching version.
    print(f"Feature cache: {FEATURE_CACHE_DIR}  (first epoch warms it, later epochs read it)")
    for loader in (train_loader, val_loader, test_loader):
        loader.dataset.extractor = CachingFeatureExtractor()

    num_classes = len(label_map)
    class_names = list(label_map.keys())

    print(f"\nCreating 'resnet' model with {num_classes} classes: {class_names}")
    model = get_model('resnet', num_classes=num_classes)

    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        num_classes=num_classes,
        class_names=class_names,
        class_weights=class_weights,
    )

    # save_best=False so we don't drop checkpoints into models/ during this test run;
    # early stopping on so it bails early if val_loss has clearly plateaued.
    trainer.train(epochs=EPOCHS, save_best=False, early_stopping_patience=5)

    metrics, preds, labels_eval, _ = trainer.evaluate()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.plot_history(save_path=RESULTS_DIR / "training_history.png")
    trainer.plot_confusion_matrix(labels_eval, preds,
                                  save_path=RESULTS_DIR / "confusion_matrix.png")
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({**metrics, "model": "resnet", "epochs": EPOCHS,
                   "subsample_frac": SUBSAMPLE_FRAC}, f, indent=2)

    print(f"\nResNet rough-signal results saved to {RESULTS_DIR}/")
    return trainer, metrics


if __name__ == "__main__":
    result = main()
    if result is not None:
        trainer, metrics = result
