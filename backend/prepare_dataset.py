"""
=====================================
Dataset Preparation Script
=====================================
Extracts features from audio files and creates CSV datasets for training.
Dataset: URDU Emotional Speech (Angry, Happy, Neutral, Sad)
"""

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  URDU Dataset Preparation - Audio to CSV")
print("=" * 70)

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_DIR = "dataset"
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad"]
EMOTION_LABELS = {"Angry": "angry", "Happy": "happy", "Neutral": "neutral", "Sad": "sad"}
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================
# FEATURE EXTRACTION (64 features - matches model)
# ============================================================
def extract_features_from_audio(file_path: str) -> list:
    """
    Extract 64 acoustic features from audio file.
    Same format as used in feature_extraction.py module.
    """
    try:
        # Load audio (16kHz sample rate, max 3 seconds)
        y, sr = librosa.load(file_path, duration=3, sr=16000)
        
        # Normalize audio
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
        
        # Total: 64 features
        return features
        
    except Exception as e:
        print(f"  [ERROR] Failed to process {file_path}: {e}")
        return None


# ============================================================
# STEP 1: Collect all audio files
# ============================================================
print("\n[Step 1] Scanning audio files...")

all_files = []
all_labels = []

for emotion in EMOTIONS:
    emotion_dir = os.path.join(DATASET_DIR, emotion)
    if os.path.exists(emotion_dir):
        files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
        print(f"  {emotion}: {len(files)} files")
        
        for f in files:
            all_files.append(os.path.join(emotion_dir, f))
            all_labels.append(EMOTION_LABELS[emotion])

print(f"\n  Total files: {len(all_files)}")

# ============================================================
# STEP 2: Extract features from all files
# ============================================================
print("\n[Step 2] Extracting features from audio files...")
print("  (This may take a few minutes...)")

all_features = []
failed_files = []

for i, (file_path, label) in enumerate(zip(all_files, all_labels)):
    features = extract_features_from_audio(file_path)
    
    if features is not None:
        # Append label as last column
        features.append(label)
        all_features.append(features)
    else:
        failed_files.append(file_path)
    
    # Progress indicator
    if (i + 1) % 50 == 0:
        print(f"  Processed: {i + 1}/{len(all_files)} files...")

print(f"\n  Successfully processed: {len(all_features)} files")
print(f"  Failed: {len(failed_files)} files")

if len(all_features) == 0:
    print("\n[ERROR] No features extracted! Check audio files.")
    exit(1)

# ============================================================
# STEP 3: Create DataFrame
# ============================================================
print("\n[Step 3] Creating dataset...")

# Create column names (64 features + 1 label)
columns = [f"feat_{i}" for i in range(64)] + ["label"]
df = pd.DataFrame(all_features, columns=columns)

print(f"  Dataset shape: {df.shape}")
print(f"  Emotion distribution:")
print(df['label'].value_counts())

# ============================================================
# STEP 4: Train/Test Split
# ============================================================
print("\n[Step 4] Splitting into train/test sets...")

train_df, test_df = train_test_split(
    df, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=df['label']
)

print(f"  Train set: {len(train_df)} samples")
print(f"  Test set:  {len(test_df)} samples")
print(f"\n  Train emotion distribution:")
print(train_df['label'].value_counts())
print(f"\n  Test emotion distribution:")
print(test_df['label'].value_counts())

# ============================================================
# STEP 5: Save CSV files
# ============================================================
print("\n[Step 5] Saving CSV files...")

train_path = os.path.join(DATASET_DIR, "complete_train_data_64.csv")
test_path = os.path.join(DATASET_DIR, "complete_test_data_64.csv")

train_df.to_csv(train_path, index=False, header=False)
test_df.to_csv(test_path, index=False, header=False)

print(f"  Saved: {train_path}")
print(f"  Saved: {test_path}")

# ============================================================
# STEP 6: Create Male/Female splits (optional)
# ============================================================
print("\n[Step 6] Creating male/female splits...")

# Based on filename: SM = Male, SF = Female
def get_gender(filename):
    """Extract gender from filename (SM = male, SF = female)"""
    basename = os.path.basename(filename)
    if basename.startswith('SM'):
        return 'male'
    elif basename.startswith('SF'):
        return 'female'
    return 'unknown'

# Create gender-based datasets
male_files = []
female_files = []

for file_path, label in zip(all_files, all_labels):
    gender = get_gender(file_path)
    if gender == 'male':
        male_files.append((file_path, label))
    elif gender == 'female':
        female_files.append((file_path, label))

print(f"  Male files: {len(male_files)}")
print(f"  Female files: {len(female_files)}")

# Extract and save male dataset
if len(male_files) > 0:
    male_features = []
    for file_path, label in male_files:
        features = extract_features_from_audio(file_path)
        if features is not None:
            features.append(label)
            male_features.append(features)
    
    if len(male_features) > 0:
        male_df = pd.DataFrame(male_features, columns=columns)
        male_train, male_test = train_test_split(male_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=male_df['label'])
        
        male_train.to_csv(os.path.join(DATASET_DIR, "males_train_data_64.csv"), index=False, header=False)
        male_test.to_csv(os.path.join(DATASET_DIR, "males_test_data_64.csv"), index=False, header=False)
        print(f"  Saved male datasets: {len(male_train)} train, {len(male_test)} test")

# Extract and save female dataset
if len(female_files) > 0:
    female_features = []
    for file_path, label in female_files:
        features = extract_features_from_audio(file_path)
        if features is not None:
            features.append(label)
            female_features.append(features)
    
    if len(female_features) > 0:
        female_df = pd.DataFrame(female_features, columns=columns)
        female_train, female_test = train_test_split(female_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=female_df['label'])
        
        female_train.to_csv(os.path.join(DATASET_DIR, "females_train_data_64.csv"), index=False, header=False)
        female_test.to_csv(os.path.join(DATASET_DIR, "females_test_data_64.csv"), index=False, header=False)
        print(f"  Saved female datasets: {len(female_train)} train, {len(female_test)} test")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  DATASET PREPARATION COMPLETE!")
print("=" * 70)
print(f"  Total samples: {len(df)}")
print(f"  Train samples: {len(train_df)}")
print(f"  Test samples:  {len(test_df)}")
print(f"  Features:      64")
print(f"  Emotions:      {list(df['label'].unique())}")
print(f"\n  Files created:")
print(f"    - complete_train_data_64.csv")
print(f"    - complete_test_data_64.csv")
print("=" * 70)
print("\n  Next step: Run 'python train_models.py' to train the models!")
