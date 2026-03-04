# streamlit_app.py - Vocaluity Web Interface

import streamlit as st
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from config import DEVICE, SAMPLE_RATE, MODELS_DIR, BINARY_CLASSES
    from feature_extractor import AudioFeatureExtractor
    from model import get_model
except ImportError:
    # Fallback defaults if imports fail
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_RATE = 22050
    MODELS_DIR = Path("models")
    BINARY_CLASSES = ["real", "ai_generated"]

# Binary classification: real (0) vs AI-generated (1)
NUM_CLASSES = 2
CLASS_NAMES = BINARY_CLASSES

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Vocaluity - AI Vocal Detection",
    page_icon="🎵",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        margin: 20px 0;
    }
    .result-real {
        background-color: #C8E6C9;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
    .result-ai {
        background-color: #FFCDD2;
        color: #C62828;
        border: 2px solid #C62828;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# HELPER FUNCTIONS
# ============================================
@st.cache_resource
def load_model_cached():
    """Load the best trained binary model (cached across Streamlit reruns)."""
    model = get_model('simple', num_classes=NUM_CLASSES)

    if not MODELS_DIR.exists() or not list(MODELS_DIR.glob("*.pth")):
        st.sidebar.warning("No trained model found. Using random weights.")
        model.eval()
        return model

    # Pick the best accuracy model from the MOST RECENT training date.
    # This prevents old models from a different training run (e.g. a previous
    # multi-class run) from outranking a newer binary model on accuracy alone.
    entries = []
    for mf in MODELS_DIR.glob("*.pth"):
        try:
            parts = mf.stem.split("_")
            # Extract YYYYMMDD_HHMMSS timestamp from filename
            timestamp = next(
                (parts[i] + "_" + parts[i + 1]
                 for i in range(len(parts) - 1)
                 if len(parts[i]) == 8 and parts[i].isdigit()
                 and len(parts[i + 1]) == 6 and parts[i + 1].isdigit()),
                None
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
        # Sort by timestamp desc, then accuracy desc → best of latest session first
        entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_acc = entries[0][1]
        best_model = entries[0][2]

    try:
        checkpoint = torch.load(best_model, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"Model: {best_model.name}")
        st.sidebar.caption(f"Val accuracy: {best_acc:.4f}")
    except RuntimeError:
        st.sidebar.error(
            "Model architecture mismatch. The saved model may be from an older "
            "training run. Delete old .pth files from the models/ folder and retrain."
        )

    model.eval()
    return model


@st.cache_resource
def get_feature_extractor():
    """Get feature extractor (cached)."""
    return AudioFeatureExtractor()


def predict_audio(audio_path, model, extractor):
    """Make prediction on audio file."""
    # Extract features
    features = extractor.extract_for_cnn(audio_path)

    if features is None:
        return None, None, None

    # Convert to tensor
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        all_probs = probs[0].cpu().numpy()

    return pred_class, confidence, all_probs


def visualize_audio(audio_path, extractor):
    """Create visualizations for the audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=10)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
    axes[0, 0].set_title('Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                    y_axis='mel', ax=axes[0, 1])
    axes[0, 1].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 0])
    axes[1, 0].set_title('MFCCs')
    fig.colorbar(img2, ax=axes[1, 0])
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img3 = librosa.display.specshow(chroma, sr=sr, x_axis='time', 
                                     y_axis='chroma', ax=axes[1, 1])
    axes[1, 1].set_title('Chroma Features')
    fig.colorbar(img3, ax=axes[1, 1])
    
    plt.tight_layout()
    return fig


# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown('<h1 class="main-header">Vocaluity</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Generated Audio Detection System</p>',
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    show_visualizations = st.sidebar.checkbox("Show Audio Visualizations", value=True)
    show_technical = st.sidebar.checkbox("Show Technical Details", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    Vocaluity uses deep learning to detect whether
    a music clip is real or AI-generated.

    **Trained on:**
    - Real audio from MusicCaps (5,145 clips)
    - AI audio from 5 generators (27,605 clips):
      MusicGen, MusicLDM, AudioLDM2,
      Stable Audio Open, Mustango

    **Supported formats:** WAV, MP3, FLAC, OGG

    **Model Test Accuracy:** 99.82%
    """)
    
    # Load model
    model = load_model_cached()
    extractor = get_feature_extractor()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Upload a music file to analyze"
        )
        
        # Demo files
        st.markdown("---")
        st.markdown("**Or try a demo:**")
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button("🎸 Load Real Sample"):
                st.info("Demo files not included. Upload your own audio.")
        with demo_col2:
            if st.button("🤖 Load AI Sample"):
                st.info("Demo files not included. Upload your own audio.")
    
    with col2:
        st.subheader("Analysis Results")

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Display audio player
                st.audio(uploaded_file, format='audio/wav')

                # Analyze
                with st.spinner("Analyzing audio..."):
                    pred_class, confidence, all_probs = predict_audio(tmp_path, model, extractor)

                if pred_class is not None:
                    is_real = pred_class == 0
                    real_prob = float(all_probs[0])
                    ai_prob   = float(all_probs[1])

                    # Main verdict box
                    if is_real:
                        st.markdown(
                            f'<div class="result-box result-real">'
                            f'<strong>REAL MUSIC DETECTED</strong>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="result-box result-ai">'
                            f'<strong>AI-GENERATED AUDIO DETECTED</strong>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    # Confidence meter
                    st.markdown(f"**Confidence: {confidence*100:.1f}%**")
                    st.progress(confidence)

                    # Real vs AI probability breakdown
                    st.markdown("---")
                    st.markdown("**Probability Breakdown:**")

                    col_real, col_ai = st.columns(2)
                    with col_real:
                        st.metric("Real Music", f"{real_prob*100:.1f}%")
                        st.progress(real_prob)
                    with col_ai:
                        st.metric("AI-Generated", f"{ai_prob*100:.1f}%")
                        st.progress(ai_prob)

                    # Technical details
                    if show_technical:
                        st.markdown("---")
                        st.markdown("**Technical Details:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Predicted Class", CLASS_NAMES[pred_class])
                        with col_b:
                            st.metric("Raw Confidence", f"{confidence:.4f}")
                else:
                    st.error("Could not analyze the audio file. Please try another file.")

            finally:
                # Cleanup temp file
                os.unlink(tmp_path)
        else:
            st.info("Upload an audio file to begin analysis")
    
    # Visualizations
    if uploaded_file is not None and show_visualizations:
        st.markdown("---")
        st.subheader("📈 Audio Visualizations")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("Generating visualizations..."):
                fig = visualize_audio(tmp_path, extractor)
                st.pyplot(fig)
        finally:
            os.unlink(tmp_path)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Vocaluity - BEng Software Engineering Final Year Project</p>
        <p>University of Westminster / IIT Sri Lanka</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
