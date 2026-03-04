# Vocaluity Prototype Setup Guide

## Quick Start for AI Vocal Detection Prototype

This guide will help you set up and run the Vocaluity AI vocal detection prototype using the FakeMusicCaps dataset.

---

## 📁 Project Structure

```
vocaluity/
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── processed/              # Preprocessed audio files
│   └── features/               # Extracted features
├── src/
│   ├── data_loader.py          # Dataset loading utilities
│   ├── feature_extractor.py    # Audio feature extraction
│   ├── model.py                # CNN classifier
│   └── train.py                # Training script
├── notebooks/
│   └── exploration.ipynb       # Data exploration
├── app/
│   └── streamlit_app.py        # Web interface
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### Step 1: Create Virtual Environment

```bash
# Create project directory
mkdir -p vocaluity
cd vocaluity

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

#### Option A: FakeMusicCaps (Recommended)
1. Go to: https://zenodo.org/records/15063698
2. Request access (usually instant for academic use)
3. Download and extract to `data/raw/fakemusiccaps/`

#### Option B: Alternative - Use MusicCaps + Generate AI Samples
```bash
# Download MusicCaps metadata
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('google/MusicCaps'); ds.save_to_disk('data/raw/musiccaps')"
```

### Step 4: Run the Prototype

```bash
# Train the model
python src/train.py

# Launch web interface
streamlit run app/streamlit_app.py
```

---

## 📊 Dataset Options Comparison

| Dataset | Ease of Use | Size | Best For |
|---------|-------------|------|----------|
| FakeMusicCaps | ⭐⭐⭐⭐⭐ | 27K tracks | Full music detection |
| ASVspoof 2019 | ⭐⭐⭐⭐ | Large | Voice-only detection |
| Custom (Suno + Real) | ⭐⭐⭐ | Variable | Specific artists |

---

## 🔧 Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **Librosa errors**: Ensure ffmpeg is installed: `sudo apt install ffmpeg`
3. **Dataset not found**: Check paths in `config.py`

---

## 📚 References

- FakeMusicCaps Paper: https://arxiv.org/abs/2409.10684
- Deezer Deepfake Detector: https://github.com/deezer/deepfake-detector
- ASVspoof Challenge: https://www.asvspoof.org/
