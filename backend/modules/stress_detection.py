"""
=====================================
MODULE 4: Stress Detection Module
=====================================
Classifies stress level from acoustic features.
Uses: Machine Learning ONLY (Random Forest + SVM + Gradient Boosting + XGBoost)
Levels: Low, Medium, High
"""

import numpy as np
import joblib
import os

# Stress mapping from emotion labels
EMOTION_TO_STRESS = {
    "angry":   "High",
    "fear":    "High",
    "disgust": "High",
    "sad":     "Medium",
    "neutral": "Low",
    "happy":   "Low",
}

# Stress numeric scores for calculations
STRESS_SCORES = {
    "Low":    1,
    "Medium": 2,
    "High":   3
}

STRESS_COLORS = {
    "Low":    "#27ae60",   
    "Medium": "#f39c12",   
    "High":   "#e74c3c",   
}


def engineer_features_for_prediction(X):
    """
    Apply the SAME advanced feature engineering used in training.
    This is CRITICAL - features must match what the model was trained on!
    
    CSV Feature Layout (64 features):
    [0-3]   Duration & basic stats
    [4-7]   RMS Energy stats
    [8-11]  ZCR stats
    [12-15] Spectral Centroid stats
    [16-19] Spectral Rolloff stats
    [20-23] Spectral Bandwidth stats
    [24-62] MFCCs 13x3 (39 features)
    [63]    Chroma mean
    
    Args:
        X: shape (1, 64) or (n_samples, 64)
    """
    X_eng = X.copy()
    n_samples = X.shape[0]
    
    # ========== MFCC Statistics (Emotional Spectrum) ==========
   
    mfccs = X[:, 24:63]
    X_eng = np.column_stack([
        X_eng,
        np.mean(mfccs, axis=1),      
        np.std(mfccs, axis=1),      
        np.max(mfccs, axis=1),      
        np.min(mfccs, axis=1),      
        np.median(mfccs, axis=1),      
        np.percentile(mfccs, 25, axis=1),   
        np.percentile(mfccs, 75, axis=1),   
    ])
    
    # ========== Pitch & Energy Features (Critical for Happy vs Sad) ==========
    if X.shape[1] >= 64:
        # Pitch & Energy stats from the 64-feature vector
        pitch_proxy = X[:, 12]      
        pitch_std   = X[:, 13]      
        energy      = X[:, 4]       
        energy_std  = X[:, 5]       
        zcr         = X[:, 8]       
        
        X_eng = np.column_stack([
            X_eng,
            pitch_proxy,                       
            pitch_std,                         
            energy,                           
            energy_std,                        
            np.abs(energy * pitch_proxy),      
            zcr,                               
            pitch_proxy / (energy + 1e-6),     
        ])
    
    # ========== Spectral Features (Happy = More Brightness) ==========
    
    upper_mfcc = X[:, 51:63]
    X_eng = np.column_stack([
        X_eng,
        np.mean(upper_mfcc, axis=1),        
        np.std(upper_mfcc, axis=1),         
    ])
    
    # ========== Temporal Dynamics ==========
    if n_samples > 1:
        mfcc_delta = np.gradient(mfccs, axis=0)  
        X_eng = np.column_stack([
            X_eng,
            np.mean(np.abs(mfcc_delta), axis=1),  
            np.std(mfcc_delta, axis=1),             
        ])
    else:
        # Single sample: no temporal variation, use zeros
        X_eng = np.column_stack([
            X_eng,
            np.zeros(n_samples),  
            np.zeros(n_samples),  
        ])
    
    return X_eng


def load_stress_model(model_dir: str = "models"):
    """
    Load trained ML model, scaler and label encoder.
    
    Returns:
        tuple: (model, label_encoder, scaler)
    """
    model_path = os.path.join(model_dir, "stress_model.pkl")
    le_path    = os.path.join(model_dir, "label_encoder.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    for path in [model_path, le_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Please run 'python train_models.py' first!"
            )

    model  = joblib.load(model_path)
    le     = joblib.load(le_path)
    scaler = joblib.load(scaler_path)

    print(f"[Stress Detection] ML Model loaded: {type(model).__name__}")
    return model, le, scaler


def predict_stress(features: np.ndarray, model, le, scaler) -> dict:
    """
    Detect stress using ML model.
    
    Args:
        features: np.ndarray shape (1, 64) - RAW features from extraction
        model: Trained ML classifier
        le: LabelEncoder
        scaler: StandardScaler
    
    Returns:
        dict: emotion, stress_level, stress_score, confidence, probabilities
    """
    # IMPORTANT: Apply feature engineering BEFORE scaling!
    features_engineered = engineer_features_for_prediction(features)

    # Validate dimensions match scaler
    n_expected = scaler.n_features_in_
    n_actual = features_engineered.shape[1]
    if n_actual != n_expected:
        raise ValueError(
            f"Feature dimension mismatch! "
            f"Model expects {n_expected} features, got {n_actual}. "
            f"Input features shape: {features.shape}, "
            f"Engineered shape: {features_engineered.shape}"
        )

    # Scale features
    features_scaled = scaler.transform(features_engineered)

    # Predict emotion
    pred_encoded = model.predict(features_scaled)[0]
    emotion = le.inverse_transform([pred_encoded])[0]

    # Get probabilities (confidence)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        all_classes = le.classes_
        prob_dict = {cls: round(float(p), 3) for cls, p in zip(all_classes, proba)}
    else:
        confidence = 1.0
        prob_dict = {emotion: 1.0}

    # ========== SMART POST-PROCESSING FOR RECORDED AUDIO ==========
    # Analyze raw features to detect anger indicators
    # Feature layout: [0-3] Duration, [4-7] RMS Energy, [8-11] ZCR, [12-15] Spectral Centroid
    
    rms_mean = float(features[0, 4])    # Energy
    rms_std = float(features[0, 5])     # Energy variation
    zcr_mean = float(features[0, 8])    # Zero crossing rate
    zcr_std = float(features[0, 9])     # ZCR variation
    spec_cent_mean = float(features[0, 12])  # Spectral centroid (pitch proxy)
    spec_cent_std = float(features[0, 13])   # Pitch variation
    
    # Anger indicators:
    # - High energy (RMS)
    # - High ZCR (more vocal intensity)
    # - High spectral centroid (higher pitch)
    # - High variation in these features
    
    print(f"[Feature Analysis]")
    print(f"  RMS Energy: mean={rms_mean:.4f}, std={rms_std:.4f}")
    print(f"  ZCR: mean={zcr_mean:.4f}, std={zcr_std:.4f}")
    print(f"  Spectral Centroid: mean={spec_cent_mean:.2f}, std={spec_cent_std:.2f}")
    
    # Detect anger from features (regardless of model prediction)
    anger_score = 0
    
    # High energy indicates anger/excitement
    if rms_mean > 0.02:
        anger_score += 2
    elif rms_mean > 0.01:
        anger_score += 1
    
    # High energy variation (burst of anger)
    if rms_std > 0.01:
        anger_score += 1
    
    # High ZCR indicates intensity
    if zcr_mean > 0.05:
        anger_score += 2
    elif zcr_mean > 0.03:
        anger_score += 1
    
    # High ZCR variation
    if zcr_std > 0.02:
        anger_score += 1
    
    # High spectral centroid (higher pitch in anger)
    if spec_cent_mean > 2000:
        anger_score += 2
    elif spec_cent_mean > 1500:
        anger_score += 1
    
    # High pitch variation (stress in voice)
    if spec_cent_std > 500:
        anger_score += 1
    
    print(f"  Anger Score (from features): {anger_score}")
    
    # If anger indicators are strong but model predicts happy/neutral
    # This happens often with recorded audio due to mic/quality differences
    if anger_score >= 5 and emotion.lower() in ['happy', 'neutral']:
        # Check if angry has reasonable probability
        if 'angry' in prob_dict and prob_dict['angry'] > 0.15:
            print(f"  [CORRECTION] Strong anger indicators detected, overriding {emotion} -> angry")
            emotion = 'angry'
            confidence = max(confidence, prob_dict['angry'])
    
    # If very calm features but model predicts angry
    elif anger_score <= 2 and emotion.lower() == 'angry':
        if 'neutral' in prob_dict and prob_dict['neutral'] > 0.20:
            print(f"  [CORRECTION] Low anger indicators, adjusting angry -> neutral")
            emotion = 'neutral'
            confidence = max(confidence, prob_dict['neutral'])
    
    # Map emotion to stress level
    stress_level = EMOTION_TO_STRESS.get(emotion.lower(), "Medium")
    stress_score = STRESS_SCORES[stress_level]

    print(f"[Stress Detection]")
    print(f"  Emotion Detected : {emotion}")
    print(f"  Stress Level     : {stress_level}")
    print(f"  ML Confidence    : {confidence:.2%}")
    print(f"  Probabilities    : {prob_dict}")

    return {
        "emotion": emotion,
        "stress_level": stress_level,
        "stress_score": stress_score,
        "stress_color": STRESS_COLORS[stress_level],
        "confidence": round(confidence, 3),
        "emotion_probabilities": prob_dict
    }


def get_stress_description(stress_level: str) -> str:
    """Stress level explanation in English."""
    descriptions = {
        "Low": (
            "No stress detected in your voice. "
            "Calm and relaxed state detected."
        ),
        "Medium": (
            "Some stress detected. "
            "Mild tension or sad emotion present."
        ),
        "High": (
            "High stress detected. "
            "Strong negative emotion (anger/fear) detected."
        )
    }
    return descriptions.get(stress_level, "Unknown stress level")
