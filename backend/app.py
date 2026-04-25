
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import uuid
import os
import hashlib
import secrets
from datetime import datetime

from modules.database import (
    get_db, close_db,
    create_user, find_user_by_email, find_user_by_id, update_user_stats, update_user_profile,
    save_analysis_report, get_user_reports, get_all_reports, get_admin_stats,
    get_all_users, save_contact_message, get_contact_messages, ensure_admin_exists,
    delete_user, delete_report, delete_message, get_all_reports_with_users
)

# =====================
# Import All Modules
# =====================
from modules.audio_input       import load_audio, validate_audio
from modules.speech_to_text    import speech_to_text, transcribe_audio
from modules.feature_extraction import extract_features, extract_features_detailed
from modules.stress_detection  import load_stress_model, predict_stress, get_stress_description, engineer_features_for_prediction
from modules.lie_detection     import calculate_lie_probability, get_lie_description
from modules.evaluation_reporting import generate_prediction_report

# =====================
# Flask App Setup
# =====================
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max

# Init DB
get_db()


# =====================
# Auth Helper
# =====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_current_user(request):
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return None
    db = get_db()
    session = db.sessions.find_one({"token": token})
    if not session:
        return None
    user = find_user_by_id(session["user_id"])
    return user

# =====================
# Load ML Models (once at startup)
# =====================
print("\n[App Startup] Loading ML Models...")
try:
    ml_model, label_encoder, feature_scaler = load_stress_model("models")
    print("[App Startup] Models ready!")
except FileNotFoundError as e:
    print(f"[App Startup] {e}")
    print("[App Startup] Please run 'python train_models.py' first!")
    ml_model, label_encoder, feature_scaler = None, None, None

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Ensure admin account exists
ensure_admin_exists()


# =====================
# Protected Routes Helper
# =====================

def check_auth():
    """Check if user is authenticated. Returns user or None"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return None
    db = get_db()
    session = db.sessions.find_one({"token": token})
    if not session:
        return None
    user = find_user_by_id(session["user_id"])
    return user


def check_admin():
    """Check if user is admin. Returns user or None"""
    user = check_auth()
    if not user or user.get("role") != "admin":
        return None
    return user


# =====================
# Routes
# =====================

# PUBLIC ROUTES
@app.route("/")
def home():
    return send_from_directory('../frontend', 'home.html')

@app.route("/home")
@app.route("/home.html")
def home_page():
    return send_from_directory('../frontend', 'home.html')

@app.route("/login")
@app.route("/login.html")
def login_page():
    return send_from_directory('../frontend', 'login.html')

@app.route("/signup")
@app.route("/signup.html")
def signup_page():
    return send_from_directory('../frontend', 'signup.html')


# PROTECTED ROUTES - Frontend handles protection via localStorage
# Backend serves the HTML, frontend JavaScript checks if user is logged in
@app.route("/analysis")
@app.route("/analysis.html")
def analysis_page():
    return send_from_directory('../frontend', 'analysis.html')


# PROTECTED ROUTES - PROFILE (requires login)
@app.route("/profile")
@app.route("/profile.html")
def profile_page():
    return send_from_directory('../frontend', 'profile.html')


# PROTECTED ROUTES - ADMIN (requires admin role)
@app.route("/admin")
@app.route("/admin.html")
def admin_page():
    return send_from_directory('../frontend', 'admin.html')


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory('reports', filename)


@app.route("/test")
def test():
    model_status = "✅ Loaded" if ml_model else "❌ Not loaded (run train_models.py)"
    
    # Check dependencies
    deps = {}
    try:
        import imageio_ffmpeg
        deps["imageio-ffmpeg"] = "✅"
    except:
        deps["imageio-ffmpeg"] = "❌"
    
    try:
        import pydub
        deps["pydub"] = "✅"
    except:
        deps["pydub"] = "❌"
    
    try:
        import librosa
        deps["librosa"] = "✅"
    except:
        deps["librosa"] = "❌"
    
    try:
        import whisper
        deps["whisper"] = "✅"
    except:
        deps["whisper"] = "❌"
    
    return jsonify({
        "status": "API is running",
        "model": model_status,
        "dependencies": deps,
        "modules": [
            "Audio Input Module",
            "Speech to Text Module",
            "Feature Extraction Module",
            "Stress Detection Module",
            "Lie Detection Module",
            "Evaluation & Reporting Module"
        ]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Upload audio → all 6 modules run → result return
    """
    temp_path = None
    audio_info = None

    try:
        # ---- Check Models ----
        if ml_model is None:
            return jsonify({"error": "ML Models not loaded. Please run 'python train_models.py' first."}), 503

        # ---- Get Audio File ----
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "No audio file uploaded"}), 400

        filename = audio_file.filename or "audio.wav"
        temp_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}_{filename}")
        audio_file.save(temp_path)
        print(f"\n[API] New request - file: {filename}")

        # =====================
        # MODULE 1: Audio Input
        # =====================
        try:
            audio_info = load_audio(temp_path)
            is_valid = validate_audio(audio_info)
            if not is_valid:
                return jsonify({"error": "Invalid audio (too short or silent)"}), 400
        except Exception as e:
            return jsonify({"error": f"Audio load error: {str(e)}"}), 400

        # =====================
        # MODULE 3: Feature Extraction
        # =====================
        try:
            # Use converted path if available (for webm files)
            audio_path = audio_info.get("converted_path") or temp_path
            # Convert to absolute path (fixes Windows path issues)
            audio_path = os.path.abspath(audio_path)
            
            # Verify file exists before processing
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found at: {audio_path}")
            
            features = extract_features(audio_path)
            features_detail = extract_features_detailed(audio_path)
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 400

        # =====================
        # MODULE 4: Stress Detection (ML)
        # =====================
        try:
            stress_result = predict_stress(features, ml_model, label_encoder, feature_scaler)
            emotion       = stress_result["emotion"]
            stress_level  = stress_result["stress_level"]
            confidence    = stress_result["confidence"]
            stress_desc   = get_stress_description(stress_level)
        except Exception as e:
            return jsonify({"error": f"Stress detection failed: {str(e)}"}), 500

        # =====================
        # MODULE 5: Lie Detection (ML)
        # =====================
        try:
            lie_result  = calculate_lie_probability(emotion, stress_level, features, confidence)
            lie_desc    = get_lie_description(lie_result["lie_probability"])
        except Exception as e:
            return jsonify({"error": f"Lie detection failed: {str(e)}"}), 500

        # =====================
        # MODULE 2: Speech to Text
        # =====================
        transcription = ""
        transcription_confidence = 0.0
        try:
            # Use converted path if available (for webm files)
            stt_path = audio_info.get("converted_path") or temp_path
            # Convert to absolute path (fixes Windows path issues)
            stt_path = os.path.abspath(stt_path)
            
            # Verify file exists before transcribing
            if not os.path.exists(stt_path):
                raise FileNotFoundError(f"Audio file not found at: {stt_path}")
            
            # Try auto-detect first, then fallback to Urdu
            stt_result = transcribe_audio(stt_path, language="ur")
            transcription = stt_result["text"]
            transcription_confidence = stt_result["confidence"]
        except Exception as e:
            print(f"[API] Speech to text error (non-critical): {e}")

        # =====================
        # MODULE 6: Generate Report
        # =====================
        report_path = ""
        try:
            prediction_for_report = {
                "emotion":       emotion,
                "stress_level":  stress_level,
                "lie_probability": lie_result["lie_probability"],
                "lie_percentage":  lie_result["lie_percentage"],
                "confidence":    confidence,
                "indicators":    lie_result["indicators"]
            }
            report_path = generate_prediction_report(prediction_for_report, save=True)
        except Exception as e:
            print(f"[API] Report generation error (non-critical): {e}")

        # =====================
        # Build Response
        # =====================
        response = {
            # Core results
            "emotion":          emotion,
            "stress_level":     stress_level,
            "lie_probability":  lie_result["lie_probability"],
            "transcription":    transcription,

            # Detailed stress info
            "stress_score":     stress_result["stress_score"],
            "stress_description": stress_desc,
            "stress_color":     stress_result["stress_color"],
            "emotion_probabilities": stress_result["emotion_probabilities"],

            # Detailed lie info
            "lie_numeric":      lie_result["lie_numeric"],
            "lie_percentage":   lie_result["lie_percentage"],
            "lie_color":        lie_result["lie_color"],
            "lie_description":  lie_desc,
            "lie_indicators":   lie_result["indicators"],

            # Audio info
            "audio_duration":   round(audio_info["duration"], 2),
            "audio_size_kb":    audio_info["file_size_kb"],
            "audio_format":     audio_info.get("format", "WAV"),

            # Acoustic features summary
            "features": {
                "pitch_hz":         round(features_detail["pitch_hz"], 2),
                "amplitude_rms":    round(features_detail["amplitude_rms"], 4),
                "duration_sec":     round(features_detail["duration_sec"], 2),
                "zcr":              round(features_detail["zcr"], 4),
                "spectral_rolloff": round(features_detail["spectral_rolloff_hz"], 2),
            },

            # Model info
            "ml_confidence":    f"{confidence:.1%}",
            "transcription_confidence": f"{transcription_confidence:.1%}",
            "report_generated": bool(report_path),
            "report_path": "/" + report_path if report_path else "",
        }

        print(f"[API] ✅ Prediction complete: emotion={emotion}, stress={stress_level}, lie={lie_result['lie_probability']}")

        # Save report to DB for the logged-in user
        try:
            current_user = get_current_user(request)
            if current_user:
                report_data = {
                    "emotion": emotion,
                    "stress_level": stress_level,
                    "lie_probability": lie_result["lie_probability"],
                    "lie_percentage": lie_result["lie_percentage"],
                    "transcription": transcription,
                    "ml_confidence": f"{confidence:.1%}",
                    "features": features_detail,
                    "audio_duration": round(audio_info["duration"], 2),
                    "audio_size_kb": audio_info["file_size_kb"],
                }
                save_analysis_report(str(current_user["_id"]), report_data)
                update_user_stats(str(current_user["_id"]), stress_level)
        except Exception as e:
            print(f"[API] Report save error (non-critical): {e}")

        return jsonify(response), 200

    except Exception as e:
        print(f"[API] ❌ Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Temp file clean up
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        # Clean up converted webm temp file
        if audio_info and audio_info.get("converted_path"):
            try:
                os.remove(audio_info["converted_path"])
            except Exception:
                pass


@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    """
    Evaluate model on dataset and return metrics.
    MODULE 6: Evaluation & Reporting
    """
    if ml_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        import pandas as pd
        import numpy as np
        from modules.evaluation_reporting import (
            evaluate_model, plot_confusion_matrix,
            plot_performance_metrics, plot_emotion_distribution,
            save_metrics_csv
        )

        # Load test data
        test = pd.read_csv("dataset/complete_test_data_64.csv", header=None)
        X_test = test.iloc[:, :-1].values
        y_test_raw = test.iloc[:, -1].values
        y_test = label_encoder.transform(y_test_raw)

        # Apply feature engineering (same as training pipeline)
        X_test_eng = engineer_features_for_prediction(X_test)

        # Evaluate
        metrics = evaluate_model(ml_model, label_encoder, feature_scaler, X_test_eng, y_test)

        # Generate charts
        plot_confusion_matrix(metrics["y_test"], metrics["y_pred"], metrics["classes"])
        plot_performance_metrics(metrics)
        plot_emotion_distribution(y_test_raw, metrics["classes"])
        save_metrics_csv(metrics)

        return jsonify({
            "accuracy":   metrics["accuracy"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "f1_score":   metrics["f1_score"],
            "report":     metrics["report"],
            "charts_saved": ["reports/confusion_matrix.png",
                             "reports/performance_metrics.png",
                             "reports/emotion_distribution.png",
                             "reports/model_metrics.csv"]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# Auth Routes
# =====================

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "All fields required"}), 400

    existing = find_user_by_email(email)
    if existing:
        return jsonify({"error": "Email already registered"}), 409

    user = create_user(name, email, hash_password(password))
    if not user:
        return jsonify({"error": "Signup failed"}), 500

    token = secrets.token_hex(32)
    db = get_db()
    db.sessions.insert_one({"token": token, "user_id": str(user["_id"]), "created_at": datetime.utcnow()})

    return jsonify({
        "token": token,
        "user": {"id": str(user["_id"]), "name": user["name"], "email": user["email"], "role": user["role"]}
    }), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = find_user_by_email(email)
    if not user or user["password"] != hash_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = secrets.token_hex(32)
    db = get_db()
    db.sessions.insert_one({"token": token, "user_id": str(user["_id"]), "created_at": datetime.utcnow()})

    return jsonify({
        "token": token,
        "user": {"id": str(user["_id"]), "name": user["name"], "email": user["email"], "role": user["role"]}
    }), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token:
        db = get_db()
        db.sessions.delete_one({"token": token})
    return jsonify({"message": "Logged out"}), 200


@app.route("/api/profile", methods=["GET"])
def profile():
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    reports = get_user_reports(str(user["_id"]), limit=50)
    for r in reports:
        r["_id"] = str(r["_id"])
        r["user_id"] = str(r["user_id"])
        r["created_at"] = r["created_at"].isoformat()

    return jsonify({
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "analyses_count": user.get("analyses_count", 0),
            "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
        },
        "reports": reports
    }), 200


@app.route("/api/profile/update", methods=["POST"])
def update_profile():
    """Update user name and/or password."""
    user = get_current_user(request)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    new_name = data.get("name", "").strip()
    new_password = data.get("password", "").strip()
    current_password = data.get("current_password", "").strip()
    
    # Validate input
    if not new_name and not new_password:
        return jsonify({"error": "No changes provided"}), 400
    
    # If changing password, verify current password
    if new_password:
        if not current_password:
            return jsonify({"error": "Current password required to change password"}), 400
        
        import bcrypt
        if not bcrypt.checkpw(current_password.encode(), user["password"].encode()):
            return jsonify({"error": "Current password is incorrect"}), 400
        
        if len(new_password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        # Hash new password
        new_password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    else:
        new_password_hash = None
    
    # Update profile
    success = update_user_profile(str(user["_id"]), name=new_name if new_name else None, password_hash=new_password_hash)
    
    if success:
        # Get updated user
        updated_user = find_user_by_id(str(user["_id"]))
        return jsonify({
            "message": "Profile updated successfully",
            "user": {
                "id": str(updated_user["_id"]),
                "name": updated_user["name"],
                "email": updated_user["email"],
                "role": updated_user["role"]
            }
        }), 200
    else:
        return jsonify({"error": "Failed to update profile"}), 500


@app.route("/api/admin/stats", methods=["GET"])
def admin_stats():
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Admin access required"}), 403

    stats = get_admin_stats()
    recent = get_all_reports_with_users(limit=20)
    for r in recent:
        r["_id"] = str(r["_id"])
        if "user_id" in r and r["user_id"]:
            r["user_id"] = str(r["user_id"])
        if "created_at" in r:
            r["created_at"] = r["created_at"].isoformat()

    return jsonify({"stats": stats, "recent": recent}), 200


@app.route("/api/admin/users/<user_id>", methods=["DELETE"])
def admin_delete_user(user_id):
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Admin access required"}), 403

    # Prevent admin from deleting themselves
    if str(user["_id"]) == user_id:
        return jsonify({"error": "Cannot delete your own account"}), 400

    success = delete_user(user_id)
    if success:
        return jsonify({"message": "User deleted successfully"}), 200
    else:
        return jsonify({"error": "Failed to delete user"}), 500


@app.route("/api/admin/reports/<report_id>", methods=["DELETE"])
def admin_delete_report(report_id):
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Admin access required"}), 403

    success = delete_report(report_id)
    if success:
        return jsonify({"message": "Report deleted successfully"}), 200
    else:
        return jsonify({"error": "Failed to delete report"}), 500


@app.route("/api/admin/messages/<message_id>", methods=["DELETE"])
def admin_delete_message(message_id):
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Admin access required"}), 403

    success = delete_message(message_id)
    if success:
        return jsonify({"message": "Message deleted successfully"}), 200
    else:
        return jsonify({"error": "Failed to delete message"}), 500


@app.route("/api/admin/users", methods=["GET"])
def admin_users():
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Admin access required"}), 403

    users = get_all_users()
    for u in users:
        u["_id"] = str(u["_id"])
        if "created_at" in u:
            u["created_at"] = u["created_at"].isoformat()
    return jsonify({"users": users}), 200


@app.route("/api/check-email", methods=["POST"])
def check_email():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email is required"}), 400

    existing = find_user_by_email(email)
    if existing:
        return jsonify({"exists": True, "name": existing.get("name", "")}), 200
    else:
        return jsonify({"exists": False}), 200


@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    new_password = data.get("new_password", "")

    if not email or not new_password:
        return jsonify({"error": "Email and new password required"}), 400

    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    existing = find_user_by_email(email)
    if not existing:
        return jsonify({"error": "No account found with this email"}), 404

    new_hash = hash_password(new_password)
    success = update_user_password(email, new_hash)

    if success:
        return jsonify({"message": "Password updated successfully"}), 200
    else:
        return jsonify({"error": "Failed to update password"}), 500


@app.route("/api/contact", methods=["POST"])
def contact_form():
    data = request.get_json()
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    subject = data.get("subject", "").strip()
    message = data.get("message", "").strip()

    if not name or not email or not subject or not message:
        return jsonify({"error": "All fields are required"}), 400

    save_contact_message(name, email, subject, message)
    return jsonify({"message": "Message sent successfully"}), 200


@app.route("/api/admin/messages", methods=["GET"])
def admin_messages():
    user = get_current_user(request)
    if not user or user.get("role") != "admin":
        return jsonify({"error": "Admin access required"}), 403

    messages = get_contact_messages(limit=50)
    for m in messages:
        m["_id"] = str(m["_id"])
        if "created_at" in m:
            m["created_at"] = m["created_at"].isoformat()
    return jsonify({"messages": messages}), 200


if __name__ == "__main__":
    import os

    print("\n" + "=" * 50)
    print("  Urdu Speech Detection System")
    print("  Stress + Lie Detection (ML Only)")
    print("  MongoDB: Connected")
    print("=" * 50)

    try:
        port = int(os.environ.get("PORT", 10000))  # Render auto port
        app.run(host="0.0.0.0", port=port)
    finally:
        close_db()
