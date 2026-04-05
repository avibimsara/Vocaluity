import sys
import os
import io
import base64
import tempfile
from pathlib import Path

import numpy as np
import torch
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path to import config / model / feature_extractor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEVICE, SAMPLE_RATE, MODELS_DIR, BINARY_CLASSES
from feature_extractor import AudioFeatureExtractor
from model import get_model

NUM_CLASSES = 2
CLASS_NAMES = BINARY_CLASSES

app = FastAPI(title="Vocaluity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading (runs once at import / startup)
_model = None
_extractor = None
_model_name = "unknown"
_model_accuracy = -1.0


def _load_model():
    global _model, _extractor, _model_name, _model_accuracy

    model = get_model("simple", num_classes=NUM_CLASSES)

    if not MODELS_DIR.exists() or not list(MODELS_DIR.glob("*.pth")):
        print("[server] No trained model found – using random weights.")
        model.eval()
        _model = model
        _extractor = AudioFeatureExtractor()
        _model_name = "VocaluityCNN (random weights)"
        return

    entries = []
    for mf in MODELS_DIR.glob("*.pth"):
        try:
            parts = mf.stem.split("_")
            timestamp = next(
                (
                    parts[i] + "_" + parts[i + 1]
                    for i in range(len(parts) - 1)
                    if len(parts[i]) == 8
                    and parts[i].isdigit()
                    and len(parts[i + 1]) == 6
                    and parts[i + 1].isdigit()
                ),
                None,
            )
            acc = float(mf.name.split("_acc")[1].split("_")[0])
            if timestamp:
                entries.append((timestamp, acc, mf))
        except Exception:
            continue

    if not entries:
        best_model = max(MODELS_DIR.glob("*.pth"), key=os.path.getctime)
        best_acc = -1.0
    else:
        entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_acc = entries[0][1]
        best_model = entries[0][2]

    checkpoint = torch.load(best_model, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _extractor = AudioFeatureExtractor()
    _model_name = best_model.name
    _model_accuracy = best_acc
    print(f"[server] Loaded model: {_model_name}  (acc={_model_accuracy:.4f})")


_load_model()

# Helpers

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


async def _save_upload(upload: UploadFile) -> str:
    ext = Path(upload.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(await upload.read())
    tmp.close()
    return tmp.name


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# GET:Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": _model is not None,
    }

# GET: Model info
@app.get("/model/info")
def model_info():
    return {
        "model_name": _model_name,
        "accuracy": _model_accuracy,
        "classes": CLASS_NAMES,
    }

# POST: Predict class from uploaded audio file
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tmp_path = await _save_upload(file)
    try:
        features = _extractor.extract_for_cnn(tmp_path)
        if features is None:
            raise HTTPException(status_code=422, detail="Could not extract features from audio file.")

        features_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = _model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            all_probs = probs[0].cpu().numpy().tolist()

        return {
            "prediction": CLASS_NAMES[pred_class],
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {name: prob for name, prob in zip(CLASS_NAMES, all_probs)},
        }
    finally:
        os.unlink(tmp_path)

# POST: Generate visualizations (waveform, spectrogram, etc.) from uploaded audio file
@app.post("/visualize")
async def visualize(file: UploadFile = File(...)):
    tmp_path = await _save_upload(file)
    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=10)

        images: dict[str, str] = {}

        # Waveform
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        images["waveform"] = _fig_to_base64(fig)

        # Mel Spectrogram
        fig, ax = plt.subplots(figsize=(6, 3))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title("Mel Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        images["mel_spectrogram"] = _fig_to_base64(fig)

        # MFCCs
        fig, ax = plt.subplots(figsize=(6, 3))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        img2 = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=ax)
        ax.set_title("MFCCs")
        fig.colorbar(img2, ax=ax)
        images["mfccs"] = _fig_to_base64(fig)

        # Chroma
        fig, ax = plt.subplots(figsize=(6, 3))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img3 = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax)
        ax.set_title("Chroma Features")
        fig.colorbar(img3, ax=ax)
        images["chroma"] = _fig_to_base64(fig)

        return images
    finally:
        os.unlink(tmp_path)
