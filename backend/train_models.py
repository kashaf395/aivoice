
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  Urdu Speech - ML Model Training")
print("  URDU Emotional Speech Dataset (Angry, Happy, Neutral, Sad)")
print("  Stress & Lie Detection System")
print("=" * 70)

# ============================================================
# STEP 1: Load Dataset
# ============================================================
print("\n[Step 1] Loading dataset...")

dataset_dir = "dataset"
models_dir  = "models"
reports_dir = "reports"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Load base dataset
train_path = os.path.join(dataset_dir, "complete_train_data_64.csv")
test_path  = os.path.join(dataset_dir, "complete_test_data_64.csv")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("\n[ERROR] CSV dataset files not found!")
    print("  Please run 'python prepare_dataset.py' first to create the dataset.")
    print("  Expected files:")
    print(f"    - {train_path}")
    print(f"    - {test_path}")
    exit(1)

train = pd.read_csv(train_path, header=None)
test  = pd.read_csv(test_path,  header=None)

print(f"  Base train shape: {train.shape}")
print(f"  Base test shape : {test.shape}")

# Show emotion distribution
print(f"\n  Emotion distribution in train:")
print(f"    {train.iloc[:, -1].value_counts().to_dict()}")

# Try to add male/female datasets
extra_files = [
    ("males_train_data_64.csv",   "males_test_data_64.csv"),
    ("females_train_data_64.csv", "females_test_data_64.csv"),
]

for tr_file, te_file in extra_files:
    tr_path = os.path.join(dataset_dir, tr_file)
    te_path = os.path.join(dataset_dir, te_file)
    if os.path.exists(tr_path) and os.path.exists(te_path):
        extra_train = pd.read_csv(tr_path, header=None)
        extra_test  = pd.read_csv(te_path, header=None)
        train = pd.concat([train, extra_train], ignore_index=True)
        test  = pd.concat([test,  extra_test],  ignore_index=True)
        print(f"  ✅ Added: {tr_file}")

print(f"\n  Final train shape: {train.shape}")
print(f"  Final test shape : {test.shape}")
print(f"\n  Emotions detected: {sorted(train.iloc[:, -1].unique())}")

# ============================================================
# STEP 2: Data Augmentation
# ============================================================
print("\n[Step 2] Applying Data Augmentation...")

def augment_features(X, y, augmentation_factor=2):
    """
    Feature-level data augmentation.
    Simulates audio variations through feature perturbation.
    """
    augmented_X = [X]
    augmented_y = [y]
    
    for _ in range(augmentation_factor - 1):
        # 1. Add Gaussian Noise (simulates background noise)
        noise = np.random.normal(0, 0.02, X.shape)
        X_noisy = X + noise
        augmented_X.append(X_noisy)
        augmented_y.append(y)
        
        # 2. Feature Scaling Variation (simulates volume changes)
        scale_factor = np.random.uniform(0.9, 1.1)
        X_scaled = X * scale_factor
        augmented_X.append(X_scaled)
        augmented_y.append(y)
        
        # 3. Feature Dropout (simulates missing features)
        mask = np.random.binomial(1, 0.95, X.shape)
        X_dropped = X * mask
        augmented_X.append(X_dropped)
        augmented_y.append(y)
    
    return np.vstack(augmented_X), np.hstack(augmented_y)

# Prepare original data
X_train_orig = train.iloc[:, :-1].values
y_train_orig = train.iloc[:, -1].values
X_test  = test.iloc[:, :-1].values
y_test_raw  = test.iloc[:, -1].values

# Apply augmentation
X_train_aug, y_train_aug = augment_features(X_train_orig, y_train_orig, augmentation_factor=2)

print(f"  Original train samples: {len(X_train_orig)}")
print(f"  Augmented train samples: {len(X_train_aug)}")
print(f"  Augmentation ratio: {len(X_train_aug) / len(X_train_orig):.1f}x")

# ============================================================
# STEP 3: Feature Engineering (IMPROVED)
# ============================================================
print("\n[Step 3] Advanced Feature Engineering for Emotion Detection...")

def engineer_features_improved(X):
    """
    Advanced feature engineering specifically optimized for emotion detection.
    Focuses on features that distinguish happy from sad.
    
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
        X: shape (n_samples, 64)
    """
    X_eng = X.copy()
    n_samples = X.shape[0]
    
    # ========== MFCC Statistics (Emotional Spectrum) ==========
    # MFCCs are at indices 24-62 (13 coeffs x 3 stats = 39 features)
    mfccs = X[:, 24:63]
    X_eng = np.column_stack([
        X_eng,
        np.mean(mfccs, axis=1),        # Mean MFCC
        np.std(mfccs, axis=1),         # Std MFCC
        np.max(mfccs, axis=1),         # Max MFCC
        np.min(mfccs, axis=1),         # Min MFCC
        np.median(mfccs, axis=1),      # Median MFCC
        np.percentile(mfccs, 25, axis=1),    # Q1
        np.percentile(mfccs, 75, axis=1),    # Q3
    ])
    
    # ========== Pitch & Energy Features (Critical for Happy vs Sad) ==========
    if X.shape[1] >= 64:
        # Pitch & Energy stats from the 64-feature vector
        pitch_proxy = X[:, 12]      # Spectral centroid mean
        pitch_std   = X[:, 13]      # Spectral centroid std (Pitch variation)
        energy      = X[:, 4]       # RMS mean
        energy_std  = X[:, 5]       # RMS std (Energy variation)
        zcr         = X[:, 8]       # ZCR mean
        
        X_eng = np.column_stack([
            X_eng,
            pitch_proxy,                       # Pitch proxy
            pitch_std,                         # REAL Pitch variation
            energy,                            # Voice energy
            energy_std,                        # Energy variation
            np.abs(energy * pitch_proxy),      # Energy-pitch interaction
            zcr,                               # Zero crossing rate
            pitch_proxy / (energy + 1e-6),     # Pitch-to-energy ratio
        ])
    
    # ========== Spectral Features (Happy = More Brightness) ==========
    # Upper MFCCs: indices 51-62 (MFCC coeff 9-12, last 12 features of MFCC block)
    upper_mfcc = X[:, 51:63]
    X_eng = np.column_stack([
        X_eng,
        np.mean(upper_mfcc, axis=1),        # High-frequency content
        np.std(upper_mfcc, axis=1),         # Spectral variability
    ])
    
    # ========== Temporal Dynamics ==========
    mfcc_delta = np.gradient(mfccs, axis=0)  # Rate of change along samples
    X_eng = np.column_stack([
        X_eng,
        np.mean(np.abs(mfcc_delta), axis=1),  # Temporal activity
        np.std(mfcc_delta, axis=1),           # Temporal variability
    ])
    
    return X_eng

# Apply improved feature engineering
X_train_eng = engineer_features_improved(X_train_aug)
X_test_eng = engineer_features_improved(X_test)

print(f"  Original features: {X_train_aug.shape[1]}")
print(f"  Engineered features: {X_train_eng.shape[1]}")
print(f"  Feature increase: {X_train_eng.shape[1] - X_train_aug.shape[1]} new features added")

# ============================================================
# STEP 4: Encode Labels & Scale Features
# ============================================================
print("\n[Step 4] Encoding labels & scaling features...")

y_train_raw = y_train_aug

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test  = le.transform(y_test_raw)
print(f"  Classes: {list(le.classes_)}")
print(f"  Train samples: {len(X_train_eng)}")
print(f"  Test samples : {len(X_test_eng)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled  = scaler.transform(X_test_eng)

# ============================================================
# STEP 5: Train ML Models with Cross-Validation (OPTIMIZED)
# ============================================================
print("\n[Step 5] Training OPTIMIZED ML Models...")
print("  (Random Forest + XGBoost + SVM + Gradient Boosting + AdaBoost)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calculate class weights to help minority/confused classes
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n  Class weights: {class_weight_dict}")

# Model 1: Random Forest (Highly Optimized)
print("\n  [1/5] Random Forest (Highly Optimized)...")
rf_model = RandomForestClassifier(
    n_estimators=800,
    max_depth=22,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    max_samples=0.85,
    criterion='gini'
)
rf_model.fit(X_train_scaled, y_train)
rf_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
print(f"       CV Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
print(f"       Test Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# Model 2: XGBoost (NEW - Better for Imbalanced/Complex Classifications)
print("\n  [2/5] XGBoost (NEW)...")
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    gamma=1.5,
    min_child_weight=3,
    scale_pos_weight=1,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_scaled, y_train)
xgb_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled))
print(f"       CV Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std()*2:.4f})")
print(f"       Test Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

# Model 3: Gradient Boosting (Optimized)
print("\n  [3/5] Gradient Boosting (Optimized)...")
gb_model = GradientBoostingClassifier(
    n_estimators=350,
    learning_rate=0.06,
    max_depth=7,
    min_samples_split=3,
    min_samples_leaf=2,
    subsample=0.82,
    max_features='sqrt',
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
gb_acc = accuracy_score(y_test, gb_model.predict(X_test_scaled))
print(f"       CV Accuracy: {gb_scores.mean():.4f} (+/- {gb_scores.std()*2:.4f})")
print(f"       Test Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")

# Model 4: SVM (Optimized)
print("\n  [4/5] SVM (Optimized)...")
svm_model = SVC(
    kernel='rbf',
    C=20,
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)
svm_model.fit(X_train_scaled, y_train)
svm_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
print(f"       CV Accuracy: {svm_scores.mean():.4f} (+/- {svm_scores.std()*2:.4f})")
print(f"       Test Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")

# Model 5: AdaBoost (NEW - Focuses on Misclassified Samples)
print("\n  [5/5] AdaBoost (NEW - Focus on difficult samples)...")
ada_model = AdaBoostClassifier(
    n_estimators=300,
    learning_rate=0.08,
    random_state=42,
    algorithm='SAMME'
)
ada_model.fit(X_train_scaled, y_train)
ada_scores = cross_val_score(ada_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
ada_acc = accuracy_score(y_test, ada_model.predict(X_test_scaled))
print(f"       CV Accuracy: {ada_scores.mean():.4f} (+/- {ada_scores.std()*2:.4f})")
print(f"       Test Accuracy: {ada_acc:.4f} ({ada_acc*100:.2f}%)")

# Model 6: Stacking Classifier (Best of all)
print("\n  [6/5] Stacking Classifier training...")
stacking = StackingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('gb', gb_model),
        ('svm', svm_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.8, class_weight='balanced'),
    cv=3,
    n_jobs=-1
)
stacking.fit(X_train_scaled, y_train)
stacking_acc = accuracy_score(y_test, stacking.predict(X_test_scaled))
print(f"       Test Accuracy: {stacking_acc:.4f} ({stacking_acc*100:.2f}%)")

# Model 7: Voting Classifier (Enhanced - More Weight on Best Models)
print("\n  [Ensemble] Voting Classifier (Enhanced)...")
voting = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('gb', gb_model),
        ('svm', svm_model),
        ('ada', ada_model)
    ],
    voting='soft',
    weights=[2, 3, 2, 1, 1]  # XGBoost gets highest weight
)
voting.fit(X_train_scaled, y_train)
voting_acc = accuracy_score(y_test, voting.predict(X_test_scaled))
print(f"       Test Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")

# ============================================================
# STEP 6: Select Best Model
# ============================================================
print("\n[Step 6] Selecting best model...")

models_acc = {
    "RandomForest": (rf_model, rf_acc),
    "XGBoost": (xgb_model, xgb_acc),
    "GradientBoosting": (gb_model, gb_acc),
    "SVM": (svm_model, svm_acc),
    "AdaBoost": (ada_model, ada_acc),
    "Stacking": (stacking, stacking_acc),
    "Voting": (voting, voting_acc),
}

best_name = max(models_acc, key=lambda k: models_acc[k][1])
best_model, best_acc = models_acc[best_name]
print(f"\n  Best Model: {best_name} (Accuracy: {best_acc:.4f} - {best_acc*100:.2f}%)")
print(f"  Improvement from baseline: +{(best_acc - 0.6589)*100:.2f}%")

# ============================================================
# STEP 7: Detailed Evaluation with Emotion-Specific Analysis
# ============================================================
print("\n[Step 7] Detailed evaluation...")

y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Per-class metrics
print("\n[EMOTION-SPECIFIC ANALYSIS]")
precisions, recalls, f1s, supports = precision_recall_fscore_support(y_test, y_pred, labels=range(len(le.classes_)), zero_division=0)
for i, emotion in enumerate(le.classes_):
    print(f"  {emotion.upper():12} - Precision: {precisions[i]:.3f} | Recall: {recalls[i]:.3f} | F1: {f1s[i]:.3f} | Support: {supports[i]}")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', annot_kws={'size': 12},
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, cbar_kws={'label': 'Count'})
ax.set_title(f'Confusion Matrix - {best_name}\nAccuracy: {best_acc*100:.2f}%', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Emotion', fontsize=11)
ax.set_ylabel('Actual Emotion', fontsize=11)
plt.tight_layout()
cm_path = os.path.join(reports_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  ✅ Confusion matrix saved: {cm_path}")

# Performance Bar Chart
from sklearn.metrics import precision_score, recall_score, f1_score
metrics = {
    'Accuracy':  accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
    'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
    'F1 Score':  f1_score(y_test, y_pred, average='weighted', zero_division=0),
}

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(list(metrics.keys()), list(metrics.values()),
              color=['#3498db','#2ecc71','#e67e22','#9b59b6'], edgecolor='white', linewidth=2)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2%}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.2)
ax.set_title(f'IMPROVED Model Performance - {best_name}', fontsize=14, fontweight='bold')
ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='85% Target', linewidth=2)
ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, label='80% Threshold', linewidth=2)
ax.legend(fontsize=11)
ax.set_ylabel('Score', fontsize=11)
plt.tight_layout()
perf_path = os.path.join(reports_dir, "performance_metrics.png")
plt.savefig(perf_path, dpi=150)
plt.close()
print(f"  ✅ Performance chart saved: {perf_path}")

# Per-Emotion Performance Chart (IMPORTANT for debugging happy emotion)
fig, ax = plt.subplots(figsize=(12, 6))
emotions = list(le.classes_)
x_pos = np.arange(len(emotions))
width = 0.25

precisions_list = [precisions[i] for i in range(len(emotions))]
recalls_list = [recalls[i] for i in range(len(emotions))]
f1s_list = [f1s[i] for i in range(len(emotions))]

ax.bar(x_pos - width, precisions_list, width, label='Precision', color='#3498db', edgecolor='black')
ax.bar(x_pos, recalls_list, width, label='Recall', color='#2ecc71', edgecolor='black')
ax.bar(x_pos + width, f1s_list, width, label='F1 Score', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Emotion Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(emotions)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.tight_layout()
emotion_path = os.path.join(reports_dir, "emotion_performance.png")
plt.savefig(emotion_path, dpi=150)
plt.close()
print(f"  ✅ Emotion-specific performance chart saved")

# Model comparison chart
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(models_acc.keys())
model_accs  = [v[1] for v in models_acc.values()]
colors = ['#3498db' if v[1] != best_acc else '#2ecc71' for v in models_acc.values()]
bars2 = ax.bar(model_names, model_accs, color=colors, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars2, model_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.2%}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_title('All Models Comparison (IMPROVED Training)', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=11)
ax.axhline(y=best_acc, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_name}', linewidth=2)
ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% Threshold', linewidth=2)
ax.legend(fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
comp_path = os.path.join(reports_dir, "model_comparison.png")
plt.savefig(comp_path, dpi=150)
plt.close()
print(f"  ✅ Model comparison chart saved")

# ============================================================
# STEP 8: Save Models
# ============================================================
print("\n[Step 8] Saving models...")

joblib.dump(best_model, os.path.join(models_dir, "stress_model.pkl"))
joblib.dump(le,         os.path.join(models_dir, "label_encoder.pkl"))
joblib.dump(scaler,     os.path.join(models_dir, "scaler.pkl"))

# Save RF as backup
joblib.dump(rf_model,   os.path.join(models_dir, "stress_model_improved.pkl"))

# Save feature engineering info
feature_info = {
    "n_original_features": 64,
    "n_engineered_features": X_train_eng.shape[1],
    "augmentation_applied": True,
    "best_model": best_name,
    "accuracy": best_acc
}
joblib.dump(feature_info, os.path.join(models_dir, "feature_info.pkl"))

print(f"  stress_model.pkl saved ({best_name})")
print(f"  label_encoder.pkl saved")
print(f"  scaler.pkl saved")
print(f"  feature_info.pkl saved")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  TRAINING COMPLETE!")
print("=" * 70)
print(f"  Best Model     : {best_name}")
print(f"  Test Accuracy  : {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"  CV Score       : {np.mean([rf_scores.mean(), gb_scores.mean(), svm_scores.mean()]):.4f}")
print(f"  Data Augmented : {len(X_train_aug)} samples ({len(X_train_aug)/len(X_train_orig):.1f}x)")
print(f"  Features       : {X_train_eng.shape[1]} (engineered)")
print(f"  Models saved in: ./{models_dir}/")
print(f"  Reports saved in: ./{reports_dir}/")
print("=" * 70)

# Print improvement summary
print("\n[Improvement Summary]")
print(f"  Original samples: {len(X_train_orig)}")
print(f"  Augmented samples: {len(X_train_aug)}")
print(f"  Feature count: {X_train_eng.shape[1]} (from {X_train_orig.shape[1]})")
print(f"  Best accuracy achieved: {best_acc*100:.2f}%")
