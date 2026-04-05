import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("VOCALUITY_BASE_DIR", Path(__file__).parent))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# Dataset paths (these should be organized as RAW_DATA_DIR / dataset_name / class_name / audio_files)
FAKEMUSICCAPS_PATH = RAW_DATA_DIR / "fakemusiccaps"
MUSICCAPS_PATH = RAW_DATA_DIR / "musiccaps"
ASVSPOOF_PATH = RAW_DATA_DIR / "asvspoof"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Audio settings
SAMPLE_RATE = 22050  # Standard for music analysis
DURATION = 5  # seconds - clip length for training
N_MFCC = 40  # Number of MFCC coefficients
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512
N_FFT = 2048

# Model settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Classes
BINARY_CLASSES = ["real", "ai_generated"]

# For multi-class (AI generator attribution)
MULTI_CLASSES = [
    "MusicGen_medium",
    "musicldm",
    "audioldm2",
    "stable_audio_open",
    "mustango"
]

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    print(f"Using device: {DEVICE}")
PIN_MEMORY = DEVICE == "cuda"
