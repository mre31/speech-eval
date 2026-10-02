"""Audio preprocessing pipeline for Turkish speech evaluation.

Standardizes audio input:
- Converts to mono 16 kHz PCM float32
- Gently filters low-frequency rumble below 50 Hz
- Trims non-speech silence boundaries
- Normalizes volume to standard target RMS/peak
- Avoids aggressive noise suppression that distorts phonetic articulation
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


def butter_highpass_filter(data: np.ndarray, cutoff: float = 60.0, fs: int = 16000, order: int = 5) -> np.ndarray:
    """Removes subsonic rumble and DC offset below cutoff frequency."""
    sos = butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    return sosfilt(sos, data).astype(np.float32)


def normalize_volume(audio: np.ndarray, target_peak: float = 0.90, target_rms: float = 0.08) -> np.ndarray:
    """Normalizes audio volume to standard level without clipping."""
    if len(audio) == 0:
        return audio
    
    current_peak = np.max(np.abs(audio))
    if current_peak < 1e-6:
        return audio

    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 1e-6:
        scale = target_rms / current_rms
        audio = audio * scale

    # Prevent hard clipping
    peak_after = np.max(np.abs(audio))
    if peak_after > target_peak:
        audio = audio * (target_peak / peak_after)

    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int = 16000, top_db: float = 30.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trims leading and trailing silence using energy threshold with margin."""
    trimmed, index = librosa.effects.trim(audio, top_db=top_db, frame_length=512, hop_length=128)
    
    # Keep 50ms padding at edges to preserve initial/final consonants
    pad_samples = int(0.05 * sr)
    start = max(0, index[0] - pad_samples)
    end = min(len(audio), index[1] + pad_samples)
    
    return audio[start:end], (start, end)


def preprocess_audio(
    audio_input: Union[str, Path, np.ndarray],
    sr: int = 16000,
    target_peak: float = 0.90,
    trim: bool = True
) -> np.ndarray:
    """Loads and standardizes input audio.
    
    Args:
        audio_input: File path or raw numpy waveform
        sr: Desired sample rate (default 16000)
        target_peak: Target peak level for scaling
        trim: Whether to trim lead/trail silence
        
    Returns:
        mono float32 numpy array sampled at 16000 Hz
    """
    if isinstance(audio_input, (str, Path)):
        audio, orig_sr = librosa.load(str(audio_input), sr=sr, mono=True)
    else:
        audio = np.asarray(audio_input, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

    # Filter DC offset and subsonic hum
    audio = butter_highpass_filter(audio, cutoff=60.0, fs=sr)

    # Trim silence
    if trim and len(audio) > sr * 0.2:
        audio, _ = trim_silence(audio, sr=sr, top_db=32.0)

    # Normalize gain
    audio = normalize_volume(audio, target_peak=target_peak)

    return audio
