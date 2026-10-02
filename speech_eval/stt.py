"""Speech-to-Text (STT) transcription and text normalization for Turkish."""

import re
from typing import Optional, Union, Tuple
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel


def normalize_turkish_text(text: str) -> str:
    """Standardizes Turkish text: lowers casing accurately and removes punctuation.
    
    Handles Turkish dotted/dotless I rules correctly:
    'İSTANBUL' -> 'istanbul'
    'IŞIK' -> 'ışık'
    """
    if not text:
        return ""

    # Accurate Turkish lowercasing
    tr_map = {
        'İ': 'i',
        'I': 'ı',
        'Ğ': 'ğ',
        'Ü': 'ü',
        'Ş': 'ş',
        'Ö': 'ö',
        'Ç': 'ç',
    }
    for upper, lower in tr_map.items():
        text = text.replace(upper, lower)
    
    text = text.lower()
    # Remove punctuation except letters and spaces
    text = re.sub(r'[^abcçdefgğhıijklmnoöprsştuüvyz\s]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TurkishSTT:
    """Wrapper around faster-whisper for Turkish speech recognition."""

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.device = device
        compute_type = "float16" if device == "cuda" else "int8"
        try:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type, local_files_only=True)
            except Exception:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            # Fallback to CPU if CUDA fails
            self.device = "cpu"
            try:
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
            except Exception:
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, target_text: Optional[str] = None) -> Tuple[str, str]:
        """Transcribes audio or normalizes provided target text.
        
        Args:
            audio: 16kHz mono float32 numpy array
            target_text: Optional known prompt read by user
            
        Returns:
            Tuple of (raw_transcript, normalized_text)
        """
        if target_text and target_text.strip():
            norm = normalize_turkish_text(target_text)
            return target_text.strip(), norm

        segments, _ = self.model.transcribe(
            audio,
            language="tr",
            task="transcribe",
            beam_size=5,
            vad_filter=True
        )
        
        raw_text = " ".join([seg.text for seg in segments]).strip()
        norm_text = normalize_turkish_text(raw_text)
        return raw_text, norm_text
