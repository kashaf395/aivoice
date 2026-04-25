"""
=====================================
MODULE 5: Lie Detection Module
=====================================
Calculates probability of deception from acoustic features.
Uses: Machine Learning ONLY (Rule-based scoring + ML features)
Output: Low / Medium / High lie probability
"""

import numpy as np
import joblib
import os


# Lie probability mapping from emotion + stress
LIE_PROBABILITY_MAP = {
    ("angry",   "High"):   ("High",   0.85),
    ("fear",    "High"):   ("High",   0.82),
    ("disgust", "High"):   ("High",   0.78),
    ("fear",    "Medium"): ("Medium", 0.60),
    ("angry",   "Medium"): ("Medium", 0.55),
    ("sad",     "Medium"): ("Medium", 0.45),
    ("sad",     "Low"):    ("Low",    0.25),
    ("neutral", "Low"):    ("Low",    0.10),
    ("happy",   "Low"):    ("Low",    0.08),
    ("disgust", "Medium"): ("Medium", 0.50),
    ("neutral", "Medium"): ("Low",    0.20),
    ("happy",   "Medium"): ("Low",    0.15),
    ("neutral", "High"):   ("Medium", 0.40),
    ("happy",   "High"):   ("Medium", 0.35),
}

LIE_COLORS = {
    "Low":    "#27ae60",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
}

LIE_SCORES = {
    "Low":    1,
    "Medium": 2,
    "High":   3
}


def calculate_lie_probability(
    emotion: str,
    stress_level: str,
    features: np.ndarray,
    confidence: float
) -> dict:
    """
    Calculate lie probability from ML features and emotion/stress.
    
    Algorithm:
    1. Get base probability from emotion+stress map
    2. Apply feature-based adjustments (pitch, ZCR, amplitude)
    3. Apply confidence adjustment

    Args:
        emotion: Detected emotion string
        stress_level: "Low" / "Medium" / "High"
        features: np.ndarray (1, 64) - acoustic features
        confidence: ML model confidence (0-1)

    Returns:
        dict: lie_probability, lie_score, lie_numeric, indicators
    """
    key = (emotion.lower(), stress_level)
    base_level, base_prob = LIE_PROBABILITY_MAP.get(key, ("Medium", 0.40))

    # --- Feature-based adjustments ---
    # CSV Feature Layout (64 features):
    # [0-3] Duration stats, [4-7] RMS stats, [8-11] ZCR stats,
    # [12-15] Spectral Centroid, [16-19] Spectral Rolloff, [20-23] Spectral Bandwidth,
    # [24-62] MFCCs 13x3, [63] Chroma mean
    feat = features[0]  # flatten (64,)

    adjustments = []
    adjusted_prob = base_prob

    # ZCR mean (index 8): High ZCR = more nervousness = possible lie
    if len(feat) > 8:
        zcr = feat[8]
        if zcr > 0.15:
            adjusted_prob += 0.05
            adjustments.append("High ZCR detected (+nervousness)")
        elif zcr < 0.05:
            adjusted_prob -= 0.03
            adjustments.append("Low ZCR (calm voice)")

    # Spectral Centroid mean (index 12) as pitch proxy: High = possible stress/lie
    if len(feat) > 12:
        pitch_proxy = feat[12]
        if pitch_proxy > 3000:
            adjusted_prob += 0.06
            adjustments.append("High spectral centroid detected (+stress)")
        elif pitch_proxy < 500:
            adjusted_prob += 0.03
            adjustments.append("Low spectral centroid (monotone)")

    # RMS mean (index 4): Very low = whispering/hiding
    if len(feat) > 4:
        rms = feat[4]
        if rms < 0.01:
            adjusted_prob += 0.04
            adjustments.append("Very low amplitude (possible concealment)")
        elif rms > 0.2:
            adjusted_prob += 0.03
            adjustments.append("Very high amplitude (shouting/anger)")

    # Confidence adjustment: If model is not confident, add uncertainty
    if confidence < 0.5:
        adjusted_prob += 0.05
        adjustments.append("Low model confidence (uncertain behavior)")

    # Clamp probability to [0, 1]
    adjusted_prob = float(np.clip(adjusted_prob, 0.0, 1.0))

    # Convert to level
    if adjusted_prob >= 0.65:
        lie_level = "High"
    elif adjusted_prob >= 0.35:
        lie_level = "Medium"
    else:
        lie_level = "Low"

    print(f"[Lie Detection]")
    print(f"  Base Probability : {base_prob:.2%} ({base_level})")
    print(f"  Adjusted Prob    : {adjusted_prob:.2%} ({lie_level})")
    print(f"  Adjustments      : {adjustments if adjustments else 'None'}")

    return {
        "lie_probability": lie_level,
        "lie_numeric": round(adjusted_prob, 3),
        "lie_percentage": f"{adjusted_prob * 100:.1f}%",
        "lie_score": LIE_SCORES[lie_level],
        "lie_color": LIE_COLORS[lie_level],
        "base_probability": round(base_prob, 3),
        "adjustments": adjustments,
        "indicators": _get_indicators(emotion, stress_level, adjustments)
    }


def _get_indicators(emotion: str, stress_level: str, adjustments: list) -> list:
    """Generate list of lie indicators."""
    indicators = []

    if emotion.lower() in ["fear", "angry"]:
        indicators.append("Negative emotion detected in speech")

    if stress_level == "High":
        indicators.append("High stress level in voice")

    if stress_level == "Medium":
        indicators.append("Moderate stress detected")

    for adj in adjustments:
        indicators.append(adj)

    if not indicators:
        indicators.append("No significant deceptive indicators found")

    return indicators


def get_lie_description(lie_level: str) -> str:
    """Get description for lie level."""
    descriptions = {
        "Low": "Speech sounds natural. No significant deceptive indicators found.",
        "Medium": "Some suspicious patterns detected. Further investigation recommended.",
        "High": "High stress and deceptive indicators present. High lie probability."
    }
    return descriptions.get(lie_level, "Unknown")
