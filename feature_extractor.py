import numpy as np
import librosa
import librosa.display
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import config
import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    SAMPLE_RATE, DURATION, N_MFCC, N_MELS, 
    HOP_LENGTH, N_FFT
)


class AudioFeatureExtractor:
    """
    Extract audio features for AI vocal detection.
    
    Features extracted:
    - MFCCs (Mel-frequency cepstral coefficients)
    - Mel Spectrogram
    - Chroma features
    - Spectral contrast
    - Zero crossing rate
    """
    
    def __init__(self, sr=SAMPLE_RATE, duration=DURATION):
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        
    def load_audio(self, file_path):
        """Load and preprocess audio file."""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # Pad or trim to target length
            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
            else:
                y = y[:self.target_length]
            
            return y
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mfcc(self, y):
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=self.sr, 
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        return mfcc
    
    def extract_mel_spectrogram(self, y):
        """Extract Mel Spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
        return mel_spec_db
    
    def extract_chroma(self, y):
        """Extract Chroma features."""
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sr,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        return chroma
    
    def extract_spectral_contrast(self, y):
        """Extract Spectral Contrast."""
        contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=self.sr,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
        )
        return contrast
    
    def extract_zero_crossing_rate(self, y):
        """Extract Zero Crossing Rate."""
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        return zcr
    
    def extract_all_features(self, file_path):
        """Extract all features from an audio file."""
        y = self.load_audio(file_path)
        if y is None:
            return None
        
        features = {
            'mfcc': self.extract_mfcc(y),
            'mel_spectrogram': self.extract_mel_spectrogram(y),
            'chroma': self.extract_chroma(y),
            'spectral_contrast': self.extract_spectral_contrast(y),
            'zcr': self.extract_zero_crossing_rate(y)
        }
        
        return features
    
    def extract_for_cnn(self, file_path):
        """
        Extract features formatted for CNN input.
        Returns a single mel spectrogram image suitable for CNN.
        """
        y = self.load_audio(file_path)
        if y is None:
            return None
        
        # Get mel spectrogram (primary feature for CNN)
        mel_spec = self.extract_mel_spectrogram(y)
        
        # Reshape for CNN: (channels, height, width)
        mel_spec = mel_spec[np.newaxis, :, :]  # Add channel dimension
        
        return mel_spec.astype(np.float32)
    
    def extract_combined_features(self, file_path):
        """
        Extract and stack multiple features for enhanced CNN input.
        Combines mel spectrogram, MFCCs, and spectral contrast.
        """
        y = self.load_audio(file_path)
        if y is None:
            return None
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(y)
        mfcc = self.extract_mfcc(y)
        contrast = self.extract_spectral_contrast(y)
        
        # Resize to same dimensions if needed
        # Stack as multiple channels
        min_frames = min(mel_spec.shape[1], mfcc.shape[1], contrast.shape[1])
        
        mel_spec = mel_spec[:, :min_frames]
        mfcc_resized = np.resize(mfcc[:, :min_frames], (N_MELS, min_frames))
        contrast_resized = np.resize(contrast[:, :min_frames], (N_MELS, min_frames))
        
        # Stack as 3-channel input (like RGB image)
        combined = np.stack([mel_spec, mfcc_resized, contrast_resized], axis=0)
        
        return combined.astype(np.float32)


def visualize_features(features, save_path=None):
    """Visualize extracted features."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mel Spectrogram
    img1 = librosa.display.specshow(
        features['mel_spectrogram'], 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH,
        x_axis='time', 
        y_axis='mel', 
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Mel Spectrogram')
    fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')
    
    # MFCCs
    img2 = librosa.display.specshow(
        features['mfcc'], 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH,
        x_axis='time', 
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('MFCCs')
    fig.colorbar(img2, ax=axes[0, 1])
    
    # Chroma
    img3 = librosa.display.specshow(
        features['chroma'], 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH,
        x_axis='time', 
        y_axis='chroma', 
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Chroma Features')
    fig.colorbar(img3, ax=axes[1, 0])
    
    # Spectral Contrast
    img4 = librosa.display.specshow(
        features['spectral_contrast'], 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH,
        x_axis='time', 
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Spectral Contrast')
    fig.colorbar(img4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Features visualization saved to {save_path}")
    
    plt.show()
    return fig


# Test the feature extractor
if __name__ == "__main__":
    print("Testing AudioFeatureExtractor...")
    
    # Create extractor
    extractor = AudioFeatureExtractor()
    
    # Test with a sample file (replace with actual path)
    test_file = "path/to/test_audio.wav"
    
    if Path(test_file).exists():
        features = extractor.extract_all_features(test_file)
        if features:
            print(f"MFCC shape: {features['mfcc'].shape}")
            print(f"Mel Spectrogram shape: {features['mel_spectrogram'].shape}")
            print(f"Chroma shape: {features['chroma'].shape}")
            print(f"Spectral Contrast shape: {features['spectral_contrast'].shape}")
            
            # Visualize
            visualize_features(features)
    else:
        print(f"Test file not found. Feature shapes with default settings:")
        print(f"  - MFCC: ({N_MFCC}, ~{int(SAMPLE_RATE * DURATION / HOP_LENGTH)})")
        print(f"  - Mel Spectrogram: ({N_MELS}, ~{int(SAMPLE_RATE * DURATION / HOP_LENGTH)})")
        print("Place an audio file and update test_file path to test.")
