"""
=====================================
MODULE 3: Feature Extraction Module
=====================================
Extracts acoustic features from audio.
Uses: Librosa
Features: MFCC, Pitch, Amplitude, Duration, Frequency, ZCR, Chroma, Spectral
"""

import numpy as np
import librosa


def extract_features(file_path: str, n_features: int = 64) -> np.ndarray:
    """
    Extract 64 acoustic features from audio file.
    Feature order MATCHES the training CSV dataset format.
    
    Features extracted (in order):
    - Duration & basic stats (4): duration, mean_amp, std_amp, max_amp
    - RMS Energy stats (4): mean, std, max, min
    - ZCR stats (4): mean, std, max, sum
    - Spectral Centroid stats (4): mean, std, max, min
    - Spectral Rolloff stats (4): mean, std, max, min
    - Spectral Bandwidth stats (4): mean, std, max, min
    - MFCCs 13x3 (39): for each coeff: mean, std, max
    - Chroma mean (1)
    
    Args:
        file_path: Audio file path
        n_features: Total features (default 64 - matches dataset)
    
    Returns:
        np.ndarray: Shape (1, 64)
    """
    y, sr = librosa.load(file_path, duration=3, sr=16000)
    
    # Normalize audio (same as training data creation)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    features = []

    # --- Duration & basic stats (4 features) ---
    duration = librosa.get_duration(y=y, sr=sr)
    features.extend([duration, np.mean(y), np.std(y), np.max(np.abs(y))])

    # --- RMS Energy stats (4 features) ---
    rms = librosa.feature.rms(y=y)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])

    # --- Zero Crossing Rate stats (4 features) ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.sum(zcr) / len(zcr)])

    # --- Spectral Centroid stats (4 features) ---
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend([np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent), np.min(spec_cent)])

    # --- Spectral Rolloff stats (4 features) ---
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.extend([np.mean(spec_roll), np.std(spec_roll), np.max(spec_roll), np.min(spec_roll)])

    # --- Spectral Bandwidth stats (4 features) ---
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.extend([np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw), np.min(spec_bw)])

    # --- MFCCs 13 coefficients x 3 stats = 39 features ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features.extend([np.mean(mfcc[i]), np.std(mfcc[i]), np.max(mfcc[i])])

    # --- Chroma mean (1 feature) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma))

    # Total: 4+4+4+4+4+4+39+1 = 64

    # --- Padding to n_features ---
    print(f"[DEBUG] Before padding: {len(features)} features, n_features={n_features}")
    while len(features) < n_features:
        features.append(0.0)
    features = features[:n_features]  # Trim if more than needed
    print(f"[DEBUG] After trim: {len(features)} features")

    feature_array = np.array(features, dtype=np.float32).reshape(1, -1)

    # Store key values for display
    _pitch = _estimate_pitch(y, sr)
    _rms_mean = float(np.mean(rms))
    _zcr_mean = float(np.mean(zcr))
    _rolloff_mean = float(np.mean(spec_roll))

    print(f"[Feature Extraction] Extracted {len(features)} features")
    print(f"  Pitch: {_pitch:.2f} Hz")
    print(f"  Amplitude (RMS): {_rms_mean:.4f}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  ZCR: {_zcr_mean:.4f}")
    print(f"  Spectral Rolloff: {_rolloff_mean:.2f} Hz")

    return feature_array


def get_feature_names() -> list:
    """List of feature names matching CSV dataset format."""
    names = ["duration", "mean_amplitude", "std_amplitude", "max_amplitude"]
    names += ["rms_mean", "rms_std", "rms_max", "rms_min"]
    names += ["zcr_mean", "zcr_std", "zcr_max", "zcr_sum"]
    names += ["spec_cent_mean", "spec_cent_std", "spec_cent_max", "spec_cent_min"]
    names += ["spec_roll_mean", "spec_roll_std", "spec_roll_max", "spec_roll_min"]
    names += ["spec_bw_mean", "spec_bw_std", "spec_bw_max", "spec_bw_min"]
    for i in range(13):
        names += [f"mfcc_{i}_mean", f"mfcc_{i}_std", f"mfcc_{i}_max"]
    names += ["chroma_mean"]
    while len(names) < 64:
        names.append(f"pad_{len(names)}")
    return names[:64]


def extract_features_detailed(file_path: str) -> dict:
    """Return detailed features dict for display (sr=16000 matches model pipeline)."""
    y, sr = librosa.load(file_path, duration=3, sr=16000)

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

    return {
        "pitch_hz": pitch_mean,
        "amplitude_rms": float(np.mean(librosa.feature.rms(y=y))),
        "duration_sec": librosa.get_duration(y=y, sr=sr),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "spectral_rolloff_hz": float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        "sample_rate": sr,
        "mfcc_mean": float(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40))),
    }


def extract_features_engineered(file_path: str) -> np.ndarray:
    """
    Extract features with engineering for enhanced model.
    NOTE: This is NOT used in the main prediction pipeline anymore.
    predict_stress() calls engineer_features_for_prediction() internally.
    Kept for backward compatibility.
    
    Returns:
        np.ndarray: Shape (1, 64) - base features only
    """
    return extract_features(file_path, n_features=64)


def _estimate_pitch(y, sr, fmin=50, fmax=500):
    """Estimate fundamental frequency using autocorrelation."""
    if len(y) < sr:
        return 0.0
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    min_period = sr // fmax
    max_period = sr // fmin
    if max_period < len(autocorr):
        autocorr_subset = autocorr[min_period:max_period]
        if len(autocorr_subset) > 0:
            period = np.argmax(autocorr_subset) + min_period
            return float(sr / period) if period > 0 else 0.0
    return 0.0
