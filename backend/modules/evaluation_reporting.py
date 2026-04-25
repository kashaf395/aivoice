"""

MODULE 6: Evaluation and Reporting Module

"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# Configure Urdu font for Windows
_urdu_font = None
def get_urdu_font():
    """Find a font that supports Urdu text."""
    global _urdu_font
    if _urdu_font:
        return _urdu_font
    
    # Common Windows fonts with Urdu support
    urdu_fonts = [
        'Jameel Noori Nastaleeq',  
        'Urdu Typesetting',
        'Segoe UI',
        'Arial Unicode MS',
        'Microsoft Sans Serif',
        'Tahoma',
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in urdu_fonts:
        if font_name in available_fonts:
            _urdu_font = font_name
            print(f"[Report] Using Urdu font: {font_name}")
            return _urdu_font
    
    # Fallback to default
    _urdu_font = 'sans-serif'
    return _urdu_font
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from datetime import datetime

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)



#  Model Evaluation on Dataset


def evaluate_model(model, le, scaler, X_test, y_test) -> dict:
    """
    Evaluate trained model on test set.
    
    Returns:
        dict: accuracy, precision, recall, f1, report
    """
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    classes = le.classes_
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

    print("\n[Evaluation Module] Model Performance:")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "report": report,
        "y_pred": y_pred,
        "y_test": y_test,
        "classes": list(classes)
    }



# SECTION B: Confusion Matrix Plot


def plot_confusion_matrix(y_test, y_pred, classes: list, save: bool = True) -> str:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5
    )
    ax.set_title('Confusion Matrix - Stress/Emotion Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Confusion matrix saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION C: Performance Bar Chart
# ============================================================

def plot_performance_metrics(metrics: dict, save: bool = True) -> str:
    """Bar chart for Accuracy, Precision, Recall, F1."""
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.2%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_ylim(0, 1.15)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(REPORTS_DIR, "performance_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Performance chart saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION D: Emotion Distribution Plot
# ============================================================

def plot_emotion_distribution(y_labels, classes: list, save: bool = True) -> str:
    """Show emotion distribution in dataset."""
    if hasattr(y_labels[0], 'item'):
        encoded = y_labels
        counts = np.bincount(encoded, minlength=len(classes))
        label_counts = {cls: int(counts[i]) for i, cls in enumerate(classes)}
    else:
        unique, counts = np.unique(y_labels, return_counts=True)
        label_counts = dict(zip(unique, counts.tolist()))

    colors = ['#e74c3c', '#e67e22', '#27ae60', '#3498db', '#9b59b6',
              '#1abc9c', '#f39c12', '#2c3e50']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(list(label_counts.keys()), list(label_counts.values()),
                   color=colors[:len(label_counts)], edgecolor='white')
    ax1.set_title('Emotion Class Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Count')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(int(bar.get_height())), ha='center', fontsize=9)

    # Pie chart
    ax2.pie(list(label_counts.values()), labels=list(label_counts.keys()),
            colors=colors[:len(label_counts)], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Emotion Distribution (%)', fontsize=12, fontweight='bold')

    plt.suptitle('Dataset Emotion Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(REPORTS_DIR, "emotion_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Distribution chart saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION E: Single Prediction Report
# ============================================================

def generate_prediction_report(prediction_result: dict, save: bool = True) -> str:
    """
    Generate beautiful visual report for single audio prediction.
    
    Args:
        prediction_result: dict from app.py prediction
        save: Save as PNG
    """
    emotion      = prediction_result.get("emotion", "Unknown")
    stress       = prediction_result.get("stress_level", "Unknown")
    lie_prob     = prediction_result.get("lie_probability", "Unknown")
    lie_pct      = prediction_result.get("lie_percentage", "N/A")
    transcription = prediction_result.get("transcription", "")
    confidence   = prediction_result.get("confidence", 0)
    indicators   = prediction_result.get("indicators", [])

    # Clean white theme with accent colors
    colors = {
        "bg_white": "#ffffff",
        "bg_light": "#f8fafc", 
        "accent_purple": "#7c3aed",
        "accent_blue": "#3b82f6",
        "accent_green": "#10b981",
        "accent_orange": "#f59e0b",
        "accent_red": "#ef4444",
        "text_dark": "#1e293b",
        "text_gray": "#64748b",
        "border": "#e2e8f0",
        "card_shadow": "#f1f5f9"
    }
    
    stress_colors = {"Low": colors["accent_green"], "Medium": colors["accent_orange"], "High": colors["accent_red"]}
    lie_colors = {"Low": colors["accent_green"], "Medium": colors["accent_orange"], "High": colors["accent_red"]}
    
    emotion_colors = {
        "angry": "#ef4444", "happy": "#f59e0b", "neutral": "#3b82f6", 
        "sad": "#6366f1", "unknown": "#6b7280"
    }

    # Create figure with white background
    fig = plt.figure(figsize=(14, 8), facecolor=colors["bg_white"])
    
    # Main title
    fig.suptitle('URDU SPEECH ANALYSIS REPORT', fontsize=22, fontweight='bold', 
                 color=colors["text_dark"], y=0.96, fontfamily='sans-serif')
    
    # Subtitle
    fig.text(0.5, 0.92, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
             ha='center', fontsize=10, color=colors["text_gray"])

    # Create grid layout (2 rows only - no transcription)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25, 
                          left=0.05, right=0.95, top=0.88, bottom=0.08)

    # ===== EMOTION CARD =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(colors["bg_light"])
    emotion_color = emotion_colors.get(emotion.lower(), colors["accent_purple"])
    
    for spine in ax1.spines.values():
        spine.set_color(emotion_color)
        spine.set_linewidth(3)
    
    ax1.text(0.5, 0.75, 'EMOTION', ha='center', va='center', fontsize=11, 
             color=colors["text_gray"], fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, emotion.upper(), ha='center', va='center', fontsize=28, 
             color=emotion_color, fontweight='bold', transform=ax1.transAxes)
    ax1.text(0.5, 0.25, f'{confidence:.1%} Confidence', ha='center', va='center', 
             fontsize=12, color=colors["text_dark"], transform=ax1.transAxes)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ===== STRESS CARD =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(colors["bg_light"])
    stress_color = stress_colors.get(stress, colors["accent_blue"])
    
    for spine in ax2.spines.values():
        spine.set_color(stress_color)
        spine.set_linewidth(3)
    
    # Draw stress meter circle
    circle = plt.Circle((0.5, 0.45), 0.25, color=stress_color, transform=ax2.transAxes, alpha=0.9)
    ax2.add_patch(circle)
    circle_border = plt.Circle((0.5, 0.45), 0.28, fill=False, color=stress_color, 
                               linewidth=3, transform=ax2.transAxes)
    ax2.add_patch(circle_border)
    
    ax2.text(0.5, 0.45, stress.upper(), ha='center', va='center', fontsize=18, 
             color='white', fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.85, 'STRESS LEVEL', ha='center', va='center', fontsize=11, 
             color=colors["text_gray"], fontweight='bold', transform=ax2.transAxes)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # ===== LIE DETECTION CARD =====
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(colors["bg_light"])
    lie_color = lie_colors.get(lie_prob, colors["accent_blue"])
    
    for spine in ax3.spines.values():
        spine.set_color(lie_color)
        spine.set_linewidth(3)
    
    ax3.text(0.5, 0.75, 'DECEPTION', ha='center', va='center', fontsize=11, 
             color=colors["text_gray"], fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.5, lie_prob.upper(), ha='center', va='center', fontsize=28, 
             color=lie_color, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.25, lie_pct, ha='center', va='center', fontsize=14, 
             color=colors["text_dark"], transform=ax3.transAxes)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # ===== INDICATORS SECTION =====
    ax_ind = fig.add_subplot(gs[1, :])
    ax_ind.set_facecolor(colors["bg_light"])
    
    for spine in ax_ind.spines.values():
        spine.set_color(colors["border"])
        spine.set_linewidth(2)
    
    ax_ind.text(0.02, 0.85, 'LIE DETECTION INDICATORS', ha='left', va='center', 
                fontsize=12, color=colors["accent_purple"], fontweight='bold', transform=ax_ind.transAxes)
    
    if indicators:
        ind_text = ""
        for i, ind in enumerate(indicators[:5], 1):
            ind_text += f"  {i}. {ind}\n"
    else:
        ind_text = "  No significant deception indicators detected"
    
    ax_ind.text(0.02, 0.4, ind_text, ha='left', va='center', fontsize=11, 
                color=colors["text_dark"], transform=ax_ind.transAxes, 
                fontfamily='sans-serif', linespacing=1.8)
    ax_ind.set_xticks([])
    ax_ind.set_yticks([])

    # Save
    path = ""
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{REPORTS_DIR}/prediction_report_{timestamp}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=colors["bg_white"])
        print(f"[Evaluation] Prediction report saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION F: Save CSV Summary Report
# ============================================================

def save_metrics_csv(metrics: dict) -> str:
    """Save metrics to CSV file."""
    data = {
        "Metric":    ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score":     [metrics['accuracy'], metrics['precision'],
                      metrics['recall'], metrics['f1_score']],
        "Percentage": [f"{v*100:.2f}%" for v in [
            metrics['accuracy'], metrics['precision'],
            metrics['recall'], metrics['f1_score']
        ]]
    }
    df = pd.DataFrame(data)
    path = os.path.join(REPORTS_DIR, "model_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[Evaluation] Metrics CSV saved: {path}")
    return path
