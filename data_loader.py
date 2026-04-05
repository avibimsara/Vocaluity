import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    FAKEMUSICCAPS_PATH, MUSICCAPS_PATH,
    BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, DEVICE, PIN_MEMORY,
    BINARY_CLASSES, MULTI_CLASSES
)
from feature_extractor import AudioFeatureExtractor


class VocaluityDataset(Dataset):
    

    def __init__(self, file_paths, labels, extractor=None, transform=None, augment=False):
        
        self.file_paths = file_paths
        self.labels = labels
        self.extractor = extractor or AudioFeatureExtractor()
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def _spec_augment(self, features):
        """
        SpecAugment-style time and frequency masking.

        Randomly zeroes out rectangular regions in the mel spectrogram
        to prevent the model from memorising specific patterns.

        Mask sizes (conservative defaults to avoid underfitting):
          - Frequency: 1-2 masks, each up to 20 bins  (~15 % of 128 bins)
          - Time:      1-2 masks, each up to 40 frames (~18 % of ~215 frames)
        """
        features = features.clone()
        _, freq_bins, time_frames = features.shape

        # Frequency masking
        for _ in range(random.randint(1, 2)):
            f = random.randint(1, min(20, freq_bins // 6))
            f0 = random.randint(0, freq_bins - f)
            features[:, f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(random.randint(1, 2)):
            t = random.randint(1, min(40, time_frames // 5))
            t0 = random.randint(0, time_frames - t)
            features[:, :, t0:t0 + t] = 0.0

        return features

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        features = self.extractor.extract_for_cnn(file_path)

        if features is None:
            # Graceful fallback – training loop will not be disrupted
            features = np.zeros((1, 128, 216), dtype=np.float32)

        features = torch.from_numpy(features)

        if self.transform:
            features = self.transform(features)

        if self.augment:
            features = self._spec_augment(features)

        return features, label



# Dataset loaders
def load_fakemusiccaps(data_path=None, binary=True):
    
    data_path = Path(data_path or FAKEMUSICCAPS_PATH)

    if not data_path.exists():
        raise FileNotFoundError(
            f"FakeMusicCaps not found at {data_path}. "
            "Download from: https://zenodo.org/records/15063698"
        )

    file_paths = []
    labels = []

    ai_folder_map = {
        "MusicGen_medium": "MusicGen_medium",
        "musicldm":        "musicldm",
        "audioldm2":       "audioldm2",
        "stable_audio_open": "stable_audio_open",
        "mustango":        "mustango",
    }

    if binary:
        label_map = {"real": 0, "ai_generated": 1}

        real_path = data_path / "real"
        if real_path.exists():
            for f in real_path.glob("*.wav"):
                file_paths.append(str(f))
                labels.append(0)

        real_count = labels.count(0)

        for folder in ai_folder_map.values():
            folder_path = data_path / folder
            if folder_path.exists():
                for f in folder_path.glob("*.wav"):
                    file_paths.append(str(f))
                    labels.append(1)

        ai_count = labels.count(1)

        if real_count == 0:
            raise FileNotFoundError(
                f"No real audio samples found at {real_path}.\n"
                "Run download_musiccaps.py first to download real MusicCaps audio."
            )

        print(f"Loaded {len(file_paths)} samples ({real_count} real, {ai_count} AI)")

    else:
        label_map = {
            "MusicGen_medium":   0,
            "musicldm":          1,
            "audioldm2":         2,
            "stable_audio_open": 3,
            "mustango":          4,
        }

        for folder_name, label_idx in label_map.items():
            folder_path = data_path / folder_name
            if folder_path.exists():
                for f in folder_path.glob("*.wav"):
                    file_paths.append(str(f))
                    labels.append(label_idx)

        print(f"Loaded {len(file_paths)} samples across {len(label_map)} AI-generator classes")
        for name, idx in label_map.items():
            print(f"  {name}: {labels.count(idx)}")

    return file_paths, labels, label_map


def load_musiccaps(data_path=None):
    
    data_path = Path(data_path or MUSICCAPS_PATH)

    if not data_path.exists():
        return [], []

    file_paths = []
    labels = []

    for f in data_path.glob("*.wav"):
        file_paths.append(str(f))
        labels.append(0)

    print(f"Loaded {len(file_paths)} MusicCaps real samples from {data_path}")
    return file_paths, labels


def load_combined_dataset(fakemusiccaps_path=None, musiccaps_path=None, binary=True):
    
    fakemusiccaps_path = Path(fakemusiccaps_path or FAKEMUSICCAPS_PATH)
    musiccaps_path     = Path(musiccaps_path     or MUSICCAPS_PATH)

    if not fakemusiccaps_path.exists():
        raise FileNotFoundError(
            f"FakeMusicCaps not found at {fakemusiccaps_path}. "
            "Download from: https://zenodo.org/records/15063698"
        )

    file_paths = []
    labels = []

    ai_folders = [
        "MusicGen_medium", "musicldm", "audioldm2",
        "stable_audio_open", "mustango", "suno",
    ]

    if binary:
        label_map = {"real": 0, "ai_generated": 1}

        # Real samples: fakemusiccaps/real
        real_path = fakemusiccaps_path / "real"
        fmc_real = 0
        if real_path.exists():
            for f in real_path.glob("*.wav"):
                file_paths.append(str(f))
                labels.append(0)
            fmc_real = labels.count(0)

        # Real samples: standalone musiccaps/ (optional)
        mc_fps, mc_lbls = load_musiccaps(musiccaps_path)
        file_paths.extend(mc_fps)
        labels.extend(mc_lbls)
        mc_real = len(mc_fps)

        real_count = labels.count(0)

        if real_count == 0:
            raise FileNotFoundError(
                f"No real audio found.\n"
                f"  Checked: {real_path}  →  {fmc_real} files\n"
                f"  Checked: {musiccaps_path}  →  {mc_real} files\n"
                "Run download_musiccaps.py to populate fakemusiccaps/real/."
            )

        # AI samples
        for folder in ai_folders:
            folder_path = fakemusiccaps_path / folder
            if folder_path.exists():
                for f in folder_path.glob("*.wav"):
                    file_paths.append(str(f))
                    labels.append(1)

        ai_count = labels.count(1)

        ai_per_folder = {}
        for folder in ai_folders:
            folder_path = fakemusiccaps_path / folder
            if folder_path.exists():
                ai_per_folder[folder] = len(list(folder_path.glob("*.wav")))

        print(f"\nCombined dataset loaded: {len(file_paths)} total samples")
        print(f"  Real (0):         {real_count}  "
              f"[{fmc_real} FakeMusicCaps/real + {mc_real} MusicCaps]")
        print(f"  AI-generated (1): {ai_count}")
        for folder, count in ai_per_folder.items():
            print(f"    {folder}: {count}")
        print(f"  Class ratio real:AI = 1:{ai_count / max(real_count, 1):.2f}")

    else:
        # Multi-class: real=0 + AI generators 
        label_map = {
            "real":              0,
            "MusicGen_medium":   1,
            "musicldm":          2,
            "audioldm2":         3,
            "stable_audio_open": 4,
            "mustango":          5,
        }
        # Add suno class 
        suno_path = fakemusiccaps_path / "suno"
        if suno_path.exists() and any(suno_path.glob("*.wav")):
            label_map["suno"] = 6

        # Real samples
        real_path = fakemusiccaps_path / "real"
        if real_path.exists():
            for f in real_path.glob("*.wav"):
                file_paths.append(str(f))
                labels.append(0)

        mc_fps, mc_lbls = load_musiccaps(musiccaps_path)
        file_paths.extend(mc_fps)
        labels.extend(mc_lbls)

        # AI samples per generator
        for folder, label_idx in [
            ("MusicGen_medium",   1),
            ("musicldm",          2),
            ("audioldm2",         3),
            ("stable_audio_open", 4),
            ("mustango",          5),
            ("suno",              6),
        ]:
            folder_path = fakemusiccaps_path / folder
            if folder_path.exists():
                for f in folder_path.glob("*.wav"):
                    file_paths.append(str(f))
                    labels.append(label_idx)

        print(f"\nCombined multi-class dataset: {len(file_paths)} samples")
        for name, idx in label_map.items():
            print(f"  {name}: {labels.count(idx)}")

    return file_paths, labels, label_map


def load_custom_dataset(real_dir, ai_dir):
    """
    Load a custom dataset with separate real and AI directories.

    Args:
        real_dir: Path to directory containing real audio
        ai_dir:   Path to directory containing AI-generated audio

    Returns:
        file_paths, labels, label_map
    """
    real_dir = Path(real_dir)
    ai_dir   = Path(ai_dir)

    file_paths = []
    labels = []
    label_map = {"real": 0, "ai_generated": 1}

    audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]

    for ext in audio_extensions:
        for f in real_dir.glob(ext):
            file_paths.append(str(f))
            labels.append(0)

    for ext in audio_extensions:
        for f in ai_dir.glob(ext):
            file_paths.append(str(f))
            labels.append(1)

    print(f"Loaded {len(file_paths)} samples ({labels.count(0)} real, {labels.count(1)} AI)")
    return file_paths, labels, label_map



# Data loaders
def create_data_loaders(file_paths, labels, batch_size=BATCH_SIZE,
                        train_split=TRAIN_SPLIT, val_split=VAL_SPLIT):
    
    # ---- Split by ytid (song ID = filename stem) -------------------------
    ytids = [Path(fp).stem for fp in file_paths]
    unique_ytids = list(set(ytids))

    test_val_ratio         = 1.0 - train_split
    val_ratio_of_remainder = val_split / test_val_ratio

    train_ytids, temp_ytids = train_test_split(
        unique_ytids,
        test_size=test_val_ratio,
        random_state=42
    )
    val_ytids, test_ytids = train_test_split(
        temp_ytids,
        test_size=1.0 - val_ratio_of_remainder,
        random_state=42
    )

    train_set = set(train_ytids)
    val_set   = set(val_ytids)
    test_set  = set(test_ytids)

    train_files, train_labels = [], []
    val_files,   val_labels   = [], []
    test_files,  test_labels  = [], []

    for fp, label, ytid in zip(file_paths, labels, ytids):
        if ytid in train_set:
            train_files.append(fp);  train_labels.append(label)
        elif ytid in val_set:
            val_files.append(fp);    val_labels.append(label)
        else:
            test_files.append(fp);   test_labels.append(label)

    print(f"Split by song ID  Train: {len(train_files)}, "
          f"Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"  Unique songs  Train: {len(train_ytids)}, "
          f"Val: {len(val_ytids)}, Test: {len(test_ytids)}")

    # Class weights (computed on training labels only)
    train_labels_array = np.array(train_labels)
    classes = np.unique(train_labels_array)
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels_array
    )
    class_weights = torch.FloatTensor(raw_weights)
    print(f"Class weights: { {int(c): round(float(w), 4) for c, w in zip(classes, raw_weights)} }")

    # Datasets:
    #  augment=True  → SpecAugment applied to training set only
    #  augment=False → clean features for validation and test
    train_dataset = VocaluityDataset(train_files, train_labels, augment=True)
    val_dataset   = VocaluityDataset(val_files,   val_labels,   augment=False)
    test_dataset  = VocaluityDataset(test_files,  test_labels,  augment=False)

    # ---- DataLoaders -----------------------------------------------------
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=2,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_weights


def preprocess_and_cache_features(file_paths, labels, cache_dir, extractor=None):
    """
    Preprocess all audio files and cache features to disk.
    Speeds up subsequent training runs significantly.

    Args:
        file_paths: List of audio file paths
        labels:     List of labels
        cache_dir:  Directory to save cached .npy feature files
        extractor:  AudioFeatureExtractor instance
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = extractor or AudioFeatureExtractor()

    print("Preprocessing and caching features...")

    feature_paths = []
    valid_labels  = []

    for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths))):
        cache_file = cache_dir / f"features_{i:06d}.npy"

        if not cache_file.exists():
            features = extractor.extract_for_cnn(file_path)
            if features is not None:
                np.save(cache_file, features)
                feature_paths.append(str(cache_file))
                valid_labels.append(label)
        else:
            feature_paths.append(str(cache_file))
            valid_labels.append(label)

    metadata = pd.DataFrame({
        'feature_path': feature_paths,
        'label':        valid_labels
    })
    metadata.to_csv(cache_dir / 'metadata.csv', index=False)

    print(f"Cached {len(feature_paths)} feature files to {cache_dir}")
    return feature_paths, valid_labels


class CachedDataset(Dataset):
    """Dataset that loads pre-cached .npy feature files from disk."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.metadata  = pd.read_csv(self.cache_dir / 'metadata.csv')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row      = self.metadata.iloc[idx]
        features = np.load(row['feature_path'])
        features = torch.from_numpy(features)
        label    = row['label']
        return features, label
