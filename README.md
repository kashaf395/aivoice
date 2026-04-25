# Urdu Speech - Stress & Lie Detection System

## Dataset: URDU Emotional Speech

Yeh project URDU Emotional Speech Dataset use karta hai:
- **4 Emotions**: Angry, Happy, Neutral, Sad
- **400 Audio Files**: 100 per emotion
- **38 Speakers**: 27 Male, 11 Female
- **Source**: Urdu Talk Shows (YouTube)

---

## System Components (6 Modules)

| # | Module | File | Library |
|---|--------|------|---------|
| 1 | Audio Input Module | `modules/audio_input.py` | Librosa |
| 2 | Speech to Text Module | `modules/speech_to_text.py` | Whisper |
| 3 | Feature Extraction Module | `modules/feature_extraction.py` | Librosa |
| 4 | Stress Detection Module | `modules/stress_detection.py` | ML (Random Forest/SVM/GB) |
| 5 | Lie Detection Module | `modules/lie_detection.py` | ML (feature-based scoring) |
| 6 | Evaluation & Reporting Module | `modules/evaluation_reporting.py` | Matplotlib + Pandas |

---

## Project Structure

```
urdu_speech_project/
├── backend/
│   ├── app.py                    # Main Flask API
│   ├── prepare_dataset.py        # Audio to CSV conversion (NEW)
│   ├── train_models.py           # ML Model training script
│   ├── modules/
│   │   ├── audio_input.py        # Module 1
│   │   ├── speech_to_text.py     # Module 2
│   │   ├── feature_extraction.py # Module 3
│   │   ├── stress_detection.py   # Module 4
│   │   ├── lie_detection.py      # Module 5
│   │   └── evaluation_reporting.py # Module 6
│   ├── models/
│   │   ├── stress_model.pkl      # Trained ML model (Random Forest)
│   │   ├── label_encoder.pkl     # Label encoder
│   │   └── scaler.pkl            # Feature scaler
│   ├── dataset/                  # URDU audio files + CSV files
│   │   ├── Angry/                # 100 angry audio files
│   │   ├── Happy/                # 100 happy audio files
│   │   ├── Neutral/              # 100 neutral audio files
│   │   ├── Sad/                  # 100 sad audio files
│   │   └── *.csv                 # Generated training data
│   └── reports/                  # Generated charts/reports
├── frontend/
│   └── index.html                # Web UI
└── requirements.txt
```

---

## Setup & Run

### Step 1: Requirements Install Karein
```bash
pip install -r requirements.txt
```

### Step 2: Dataset Prepare Karein (NEW - Audio to CSV)
```bash
cd urdu_speech_project/backend
python prepare_dataset.py
```

Yeh command:
- `dataset/Angry/`, `Happy/`, `Neutral/`, `Sad/` folders scan karta hai
- Audio files se 64 features extract karta hai
- Train/Test split karta hai (80/20)
- CSV files generate karta hai

### Step 3: Models Train Karein
```bash
cd urdu_speech_project/backend
python train_models.py
```

Yeh command:
- CSV dataset load karta hai
- Data Augmentation apply karta hai (4x)
- Random Forest, XGBoost, SVM, Gradient Boosting, AdaBoost train karta hai
- Best model select karta hai (97.52% accuracy achieved!)
- `models/` folder mein save karta hai
- `reports/` mein charts save karta hai

### Step 4: Server Chalayein
```bash
cd urdu_speech_project/backend
python app.py
```

### Step 5: Browser Mein Kholein
```
http://localhost:5000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend (Web UI) |
| `/api/predict` | POST | Audio analyze karo |
| `/api/evaluate` | GET | Model evaluation run karo |
| `/test` | GET | API status check |

---

## ML Models Used (NO Deep Learning)

- **Random Forest Classifier** - n_estimators=800 (Best: 97.52%)
- **XGBoost Classifier** - n_estimators=400 (93.79%)
- **Gradient Boosting Classifier** - n_estimators=350 (97.52%)
- **Support Vector Machine (SVM)** - RBF kernel (97.52%)
- **AdaBoost Classifier** - n_estimators=300 (75.78%)
- **Stacking Classifier** - Ensemble of RF, XGB, GB, SVM (97.52%)
- **Voting Classifier** - Soft voting with weights (97.52%)

### Training Results
| Model | Test Accuracy |
|-------|---------------|
| Random Forest | **97.52%** |
| Gradient Boosting | 97.52% |
| SVM | 97.52% |
| Stacking | 97.52% |
| Voting | 97.52% |
| XGBoost | 93.79% |
| AdaBoost | 75.78% |

---

## Features Extracted (64 base + 18 engineered = 82 total)

### Base Features (64)
- **Duration stats (4)** - Audio length, amplitude stats
- **RMS Energy stats (4)** - Voice energy
- **ZCR stats (4)** - Zero Crossing Rate
- **Spectral Centroid stats (4)** - Brightness
- **Spectral Rolloff stats (4)** - High frequency content
- **Spectral Bandwidth stats (4)** - Frequency spread
- **MFCCs (39)** - 13 coefficients × 3 stats (tone/texture)
- **Chroma mean (1)** - Pitch class energy

### Engineered Features (+18)
- MFCC statistics (mean, std, max, min, median, Q1, Q3)
- Pitch-energy interaction features
- Spectral brightness features
- Temporal dynamics

---

## Reports Generated

After running `train_models.py`:
- `reports/confusion_matrix.png` - Emotion classification matrix
- `reports/performance_metrics.png` - Overall accuracy metrics
- `reports/model_comparison.png` - All models comparison
- `reports/emotion_performance.png` - Per-emotion precision/recall/F1

---

## Note

- Pehle `prepare_dataset.py` chalana **zaroor** hai CSV files ke liye
- Phir `train_models.py` chalana **zaroor** hai models ke liye
- Whisper model pehli baar internet se download hoga (~150MB)
- CPU pe chalane ke liye `fp16=False` already set hai
- Training mein **97.52% accuracy** mili hai!

---

## Stress & Lie Detection Mapping

### Emotion → Stress Level
| Emotion | Stress Level |
|---------|--------------|
| Angry | High |
| Sad | Medium |
| Neutral | Low |
| Happy | Low |

### Lie Probability Calculation
- Base probability from emotion + stress combination
- Feature-based adjustments (ZCR, pitch, amplitude)
- Confidence adjustment from ML model
