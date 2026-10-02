"""Unit tests for acoustic feature extraction and speaker normalization."""

import numpy as np
import pytest

from speech_eval.acoustics import (
    AcousticFeatureExtractor,
    normalize_utterance_vowels,
    compare_vowel_acoustics,
    compare_consonant_acoustics
)
from speech_eval.g2p import PhonemeItem
from speech_eval.aligner import AlignedSegment


def test_nearey_vowel_normalization():
    """Verify that Nearey log-mean normalization aligns female and male formants."""
    # Simulated male /a/ and /i/ (lower absolute formants)
    male_vowels = [
        {"grapheme": "a", "is_vowel": True, "is_valid_formant": True, "log_f1": np.log(700.0), "log_f2": np.log(1200.0)},
        {"grapheme": "i", "is_vowel": True, "is_valid_formant": True, "log_f1": np.log(300.0), "log_f2": np.log(2200.0)}
    ]
    normalize_utterance_vowels(male_vowels)

    # Simulated female /a/ and /i/ (higher absolute formants, ~15-20% shift)
    female_vowels = [
        {"grapheme": "a", "is_vowel": True, "is_valid_formant": True, "log_f1": np.log(840.0), "log_f2": np.log(1440.0)},
        {"grapheme": "i", "is_vowel": True, "is_valid_formant": True, "log_f1": np.log(360.0), "log_f2": np.log(2640.0)}
    ]
    normalize_utterance_vowels(female_vowels)

    # In Nearey space, the normalized values (F1*, F2*) should be nearly identical
    assert abs(male_vowels[0]["f1_star"] - female_vowels[0]["f1_star"]) < 0.05
    assert abs(male_vowels[0]["f2_star"] - female_vowels[0]["f2_star"]) < 0.05
    assert abs(male_vowels[1]["f1_star"] - female_vowels[1]["f1_star"]) < 0.05
    assert abs(male_vowels[1]["f2_star"] - female_vowels[1]["f2_star"]) < 0.05


def test_invalid_formant_handling():
    """Verify that failed formant extraction returns None without penalizing the speaker."""
    feat = {
        "grapheme": "e",
        "is_vowel": True,
        "is_valid_formant": False,
        "f1": None,
        "f2": None,
        "f1_star": None,
        "f2_star": None
    }
    score, note = compare_vowel_acoustics(feat, [])
    assert score is None
    assert "Formant ölçümü yetersiz" in note


def test_consonant_manner_specific_scoring():
    """Verify that stops and nasals are never tagged with frication warnings."""
    # Stop /t/
    stop_feat = {
        "grapheme": "t",
        "consonant_manner": "plosive",
        "consonant_voiced": False,
        "spectral_centroid": 4500.0,
        "spectral_flatness": 0.02,
        "voicing_ratio": 0.15,
        "nasal_murmur_ratio": 0.05
    }
    ref_stops = [{
        "spectral_centroid": 3200.0,
        "spectral_flatness": 0.01,
        "voicing_ratio": 0.10,
        "nasal_murmur_ratio": 0.05
    }]
    score, note = compare_consonant_acoustics(stop_feat, ref_stops)
    assert score >= 80.0
    assert "sürtünme" not in note.lower()

    # Nasal /m/
    nasal_feat = {
        "grapheme": "m",
        "consonant_manner": "nasal",
        "consonant_voiced": True,
        "spectral_centroid": 600.0,
        "spectral_flatness": 0.005,
        "voicing_ratio": 0.95,
        "nasal_murmur_ratio": 0.75
    }
    score_m, note_m = compare_consonant_acoustics(nasal_feat, [nasal_feat])
    assert score_m >= 88.0
    assert "sürtünme" not in note_m.lower()
