import os
import librosa
import numpy as np
import soundfile as sf
import subprocess
import tempfile


SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm']
MAX_DURATION_SECONDS = 60
MIN_DURATION_SECONDS = 0.5


def get_ffmpeg_path():
    """Get ffmpeg executable path, auto-download if needed."""
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return 'ffmpeg'
    except FileNotFoundError:
        pass
    
    
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    
    return None


def convert_webm_to_wav(webm_path: str) -> str:
    """
    Convert WebM audio to WAV format using ffmpeg or pydub.
    Returns path to converted WAV file.
    """
    # Create temp wav file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()
    
    # Ensure paths are absolute
    webm_path = os.path.abspath(webm_path)
    temp_wav_path = os.path.abspath(temp_wav_path)
    
    # Try imageio-ffmpeg first
    try:
        # Dynamic import - only when needed
        import imageio_ffmpeg 
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[Audio Input] Using imageio-ffmpeg from: {ffmpeg_exe}")
        
        result = subprocess.run(
            [ffmpeg_exe, '-y', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', temp_wav_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            print(f"[Audio Input] ✅ WebM converted to WAV successfully")
            return temp_wav_path
        else:
            print(f"[Audio Input] FFmpeg failed: {result.stderr}")
    except Exception as e:
        print(f"[Audio Input] imageio-ffmpeg not available: {e}")
    
    # Try system ffmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"[Audio Input] Using system ffmpeg")
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', webm_path, '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', temp_wav_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                print(f"[Audio Input] ✅ WebM converted to WAV using system ffmpeg")
                return temp_wav_path
    except Exception as e:
        print(f"[Audio Input] System ffmpeg not available: {e}")
    
    # Try pydub as fallback
    try:
        print(f"[Audio Input] Trying pydub fallback...")
        from pydub import AudioSegment
        
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        
        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            print(f"[Audio Input] ✅ WebM converted using pydub")
            return temp_wav_path
    except Exception as e:
        print(f"[Audio Input] Pydub conversion failed: {e}")
    
    # If all else fails, raise error
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)
    
    raise RuntimeError(
        f"❌ WebM conversion failed. Please install ffmpeg:\n"
        f"Windows: Download from https://ffmpeg.org/download.html or run: choco install ffmpeg\n"
        f"Mac: brew install ffmpeg\n"
        f"Linux: sudo apt-get install ffmpeg\n"
        f"Or Python fallback: pip install imageio-ffmpeg pydub"
    )


def load_audio(file_path: str, target_sr: int = 22050):
    """
    Load audio file and return basic info.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default 22050 Hz)
    
    Returns:
        dict: audio data, sample rate, duration, channels info
    """
    # Ensure absolute path
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {SUPPORTED_FORMATS}")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("Audio file is empty (0 bytes)")

    # Convert WebM to WAV first
    converted_path = None
    if ext == '.webm':
        print("[Audio Input] Converting WebM to WAV...")
        converted_path = convert_webm_to_wav(file_path)
        file_path = converted_path
        ext = '.wav'
        print(f"[Audio Input] Converted path: {file_path}")
        print(f"[Audio Input] File exists after conversion: {os.path.exists(file_path)}")

    # Load audio using librosa
    print(f"[Audio Input] Loading audio from: {file_path}")
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)

    if duration < MIN_DURATION_SECONDS:
        raise ValueError(f"Audio too short ({duration:.2f}s). Minimum: {MIN_DURATION_SECONDS}s")

    if duration > MAX_DURATION_SECONDS:
        print(f"[WARNING] Audio too long ({duration:.2f}s). Using first {MAX_DURATION_SECONDS}s only.")
        y = y[:MAX_DURATION_SECONDS * sr]
        duration = MAX_DURATION_SECONDS

    # Basic audio stats
    amplitude_max = float(np.max(np.abs(y)))
    amplitude_mean = float(np.mean(np.abs(y)))
    is_silent = amplitude_max < 0.01

    print(f"[Audio Input Module]")
    print(f"  File     : {os.path.basename(file_path)}")
    print(f"  Duration : {duration:.2f} seconds")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Amplitude: max={amplitude_max:.4f}, mean={amplitude_mean:.4f}")
    print(f"  Silent?  : {is_silent}")

    # Detect original format
    original_ext = os.path.splitext(file_path)[1].upper().replace('.', '')
    
    return {
        "y": y,
        "sr": sr,
        "duration": duration,
        "file_path": file_path,
        "amplitude_max": amplitude_max,
        "amplitude_mean": amplitude_mean,
        "is_silent": is_silent,
        "file_size_kb": round(file_size / 1024, 2),
        "format": original_ext if original_ext else "WAV",
        "converted_path": converted_path  # For cleanup
    }


def validate_audio(audio_info: dict) -> bool:
    """Check if audio is valid."""
    if audio_info["is_silent"]:
        print("[WARNING] Audio is silent (no voice detected)")
        return False
    if audio_info["duration"] < MIN_DURATION_SECONDS:
        print("[WARNING] Audio too short")
        return False
    return True


def get_audio_info(file_path: str) -> dict:
    """Get audio info without processing."""
    audio_info = load_audio(file_path)
    is_valid = validate_audio(audio_info)
    audio_info["is_valid"] = is_valid
    return audio_info
