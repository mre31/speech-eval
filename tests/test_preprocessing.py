"""Unit tests for preprocessing."""

import numpy as np
from speech_eval.preprocessing import normalize_volume, butter_highpass_filter, trim_silence


def test_normalize_volume():
    audio = np.array([0.01, -0.02, 0.05, -0.04], dtype=np.float32)
    normalized = normalize_volume(audio, target_peak=0.90)
    assert np.max(np.abs(normalized)) <= 0.9001
    assert np.max(np.abs(normalized)) > 0.05


def test_highpass_filter():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 20 Hz hum + 440 Hz tone
    sig = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 440 * t)
    filtered = butter_highpass_filter(sig, cutoff=60.0, fs=sr)
    assert len(filtered) == len(sig)
    # The 20Hz component should be heavily attenuated
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    idx_20 = np.argmin(np.abs(freqs - 20))
    idx_440 = np.argmin(np.abs(freqs - 440))
    assert fft_filt[idx_20] < fft_orig[idx_20] * 0.1
    assert fft_filt[idx_440] > fft_orig[idx_440] * 0.8


def test_trim_silence():
    sr = 16000
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))).astype(np.float32)
    full = np.concatenate([silence, tone, silence])
    trimmed, _ = trim_silence(full, sr=sr, top_db=25.0)
    assert len(trimmed) < len(full)
    assert len(trimmed) >= len(tone)
