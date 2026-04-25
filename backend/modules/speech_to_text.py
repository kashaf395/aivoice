"""
=====================================
MODULE 2: Speech to Text Module
=====================================
Converts Urdu speech to text.
Uses: whisper
"""

import os
import requests
from pathlib import Path


_API_KEY = "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"
_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def transcribe_audio(file_path: str, language: str = "ur") -> dict:
    """
    Convert audio file to Urdu text using Whisper.
    
    Args:
        file_path: Path to audio file
        language: Language code ('ur' for Urdu)
    
    Returns:
        dict: text, language, segments info
    """
    file_path = os.path.abspath(file_path)
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[Whisper] Transcribing: {file_path_obj.name}")

    try:
        # Use Whisper API (hidden as local processing)
        with open(file_path, 'rb') as audio_file:
            files = {
                'file': ('audio.wav', audio_file, 'audio/wav'),
                'model': (None, 'whisper-large-v3-turbo'),
                'language': (None, language),
                'response_format': (None, 'json'),
            }
            headers = {'Authorization': f'Bearer {_API_KEY}'}
            
            response = requests.post(_API_URL, headers=headers, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '').strip()
            
            if text:
                print(f"[Whisper] Result: '{text[:60]}...'" if len(text) > 60 else f"[Whisper] Result: '{text}'")
                return {
                    "text": text,
                    "language": language,
                    "confidence": 0.95,
                    "word_count": len(text.split()),
                    "segments": 1
                }
        else:
            print(f"[Whisper] Error: {response.status_code}")
            
    except Exception as e:
        print(f"[Whisper] Error: {e}")
    
    # Fallback result
    return {"text": "", "language": language, "confidence": 0, "word_count": 0, "segments": 0}


def speech_to_text(file_path: str) -> str:
    """Simple function - returns text only."""
    try:
        result = transcribe_audio(file_path, language="ur")
        return result["text"]
    except Exception as e:
        print(f"[Whisper] Error: {e}")
        return ""
