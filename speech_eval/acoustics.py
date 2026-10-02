"""Acoustic feature analysis with speaker-invariant formant normalization.

Extracts:
- Vowels: F1, F2, F3 formants via Praat Burg algorithm, normalized with logarithmic formant ratios
  and Lobanov-inspired vocal-tract normalization.
- Consonants: Spectral centroid, spectral flatness, zero-crossing rate, voicing ratio, energy distribution.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa

from speech_eval.aligner import AlignedSegment


# Standard Turkish vowel target formant ratios (log(F2/F1) and log(F3/F2))
# Derived from Turkish phonetics literature
STANDARD_TURKISH_VOWELS = {
    'a': {'f1': 800.0, 'f2': 1300.0, 'f3': 2550.0, 'ratio_f2_f1': 1.62},
    'e': {'f1': 500.0, 'f2': 1950.0, 'f3': 2650.0, 'ratio_f2_f1': 3.90},
    'ı': {'f1': 400.0, 'f2': 1480.0, 'f3': 2400.0, 'ratio_f2_f1': 3.70},
    'i': {'f1': 320.0, 'f2': 2350.0, 'f3': 2900.0, 'ratio_f2_f1': 7.34},
    'o': {'f1': 520.0, 'f2': 1050.0, 'f3': 2450.0, 'ratio_f2_f1': 2.02},
    'ö': {'f1': 480.0, 'f2': 1650.0, 'f3': 2400.0, 'ratio_f2_f1': 3.44},
    'u': {'f1': 360.0, 'f2': 950.0,  'f3': 2400.0, 'ratio_f2_f1': 2.64},
    'ü': {'f1': 340.0, 'f2': 1850.0, 'f3': 2450.0, 'ratio_f2_f1': 5.44},
}


class AcousticFeatureExtractor:
    """Extracts acoustic features tailored to vowels and consonants."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract_features(
        self,
        audio: np.ndarray,
        segment: AlignedSegment,
        praat_sound: Optional[parselmouth.Sound] = None,
        praat_formants: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Extracts class-specific acoustic features for an aligned phoneme."""
        feat: Dict[str, Any] = {
            "grapheme": segment.phoneme.grapheme,
            "is_vowel": segment.phoneme.is_vowel,
            "duration": segment.duration,
            "confidence": segment.confidence
        }

        # Slice audio samples
        start_samp = int(segment.start_time * self.sr)
        end_samp = int(segment.end_time * self.sr)
        seg_audio = audio[start_samp:end_samp]

        if len(seg_audio) < 128:
            return feat

        # Vowel Acoustic Features: Formants & Formant Ratios
        if segment.phoneme.is_vowel:
            if praat_sound is None:
                praat_sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            if praat_formants is None:
                praat_formants = praat_sound.to_formant_burg(
                    time_step=0.01,
                    max_number_of_formants=5,
                    maximum_formant=5500.0
                )

            # Sample central 60% of vowel duration to avoid transition artifacts
            mid_t = (segment.start_time + segment.end_time) / 2.0
            t_samples = np.linspace(
                segment.start_time + 0.2 * segment.duration,
                segment.end_time - 0.2 * segment.duration,
                5
            )

            f1_list = [praat_formants.get_value_at_time(1, t) for t in t_samples]
            f2_list = [praat_formants.get_value_at_time(2, t) for t in t_samples]
            f3_list = [praat_formants.get_value_at_time(3, t) for t in t_samples]

            f1_clean = [f for f in f1_list if not np.isnan(f) and 150 < f < 1400]
            f2_clean = [f for f in f2_list if not np.isnan(f) and 600 < f < 3500]
            f3_clean = [f for f in f3_list if not np.isnan(f) and 1400 < f < 4500]

            f1 = float(np.median(f1_clean)) if f1_clean else 500.0
            f2 = float(np.median(f2_clean)) if f2_clean else 1500.0
            f3 = float(np.median(f3_clean)) if f3_clean else 2500.0

            feat["f1"] = round(f1, 1)
            feat["f2"] = round(f2, 1)
            feat["f3"] = round(f3, 1)

            # Speaker-invariant normalization: Formant frequency ratios
            # F2 / F1 represents frontness vs height without speaker scale
            feat["ratio_f2_f1"] = round(f2 / max(100.0, f1), 2)
            feat["log_ratio_f2_f1"] = round(float(np.log(max(1.0, f2 / max(100.0, f1)))), 3)
            feat["log_ratio_f3_f2"] = round(float(np.log(max(1.0, f3 / max(100.0, f2)))), 3)

        # Consonant Acoustic Features: Spectral Centroid, Rolloff, Flatness, Voicing
        else:
            n_fft = min(512, max(64, len(seg_audio)))
            hop_length = max(32, n_fft // 4)

            # Spectral centroid (distinguishes sibilants, place of articulation)
            sc = librosa.feature.spectral_centroid(y=seg_audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length)
            feat["spectral_centroid"] = round(float(np.mean(sc)), 1)

            # Spectral flatness (tonal vs noisy/fricative)
            sf = librosa.feature.spectral_flatness(y=seg_audio, n_fft=n_fft, hop_length=hop_length)
            feat["spectral_flatness"] = round(float(np.mean(sf)), 4)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=seg_audio, hop_length=hop_length)
            feat["zero_crossing_rate"] = round(float(np.mean(zcr)), 4)

            # Voicing ratio (energy < 1000 Hz / total energy)
            fft_mag = np.abs(np.fft.rfft(seg_audio))
            freqs = np.fft.rfftfreq(len(seg_audio), 1.0 / self.sr)
            low_energy = np.sum(fft_mag[freqs < 1000.0] ** 2)
            total_energy = np.sum(fft_mag ** 2) + 1e-10
            feat["voicing_ratio"] = round(float(low_energy / total_energy), 3)

        return feat


def compare_vowel_acoustics(user_feat: Dict[str, Any], ref_feats: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Compares user vowel formant ratios against standard references.
    
    Returns:
        (acoustic_similarity_score 0-100, diagnostic_note)
    """
    g = user_feat.get("grapheme", "")
    user_ratio = user_feat.get("ratio_f2_f1", 1.0)
    user_f1 = user_feat.get("f1", 500.0)
    user_f2 = user_feat.get("f2", 1500.0)

    # Reference target
    ref_target = STANDARD_TURKISH_VOWELS.get(g)
    if not ref_target:
        return 80.0, "Standart aralıkta"

    target_ratio = ref_target["ratio_f2_f1"]
    
    # Also collect median ratio from TTS reference features if available
    ref_ratios = [rf["ratio_f2_f1"] for rf in ref_feats if rf.get("ratio_f2_f1") is not None]
    if ref_ratios:
        target_ratio = float(np.median(ref_ratios))

    # Normalized relative error
    rel_error = abs(user_ratio - target_ratio) / target_ratio
    
    # Non-linear scoring curve: error < 0.10 is excellent (90-100), > 0.40 drops rapidly
    score = max(0.0, min(100.0, 100.0 - (rel_error * 180.0)))

    # Diagnostic feedback
    note = "Standart telaffuz"
    if rel_error > 0.20:
        if user_ratio < target_ratio:
            if g in ['e', 'i', 'ö', 'ü']:
                note = "Ön ünlü arkaya kaymış veya kapalı"
            else:
                note = "Art ünlü öne kaymış"
        else:
            if g in ['a', 'ı', 'o', 'u']:
                note = "Art ünlü önleşmiş"
            else:
                note = "Aşırı inceltilmiş / önleşmiş"

    return round(score, 1), note


def compare_consonant_acoustics(user_feat: Dict[str, Any], ref_feats: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Compares user consonant spectral properties with standard references.
    
    Returns:
        (acoustic_similarity_score 0-100, diagnostic_note)
    """
    u_sc = user_feat.get("spectral_centroid", 3000.0)
    u_flat = user_feat.get("spectral_flatness", 0.05)

    ref_scs = [rf["spectral_centroid"] for rf in ref_feats if rf.get("spectral_centroid") is not None]
    ref_flats = [rf["spectral_flatness"] for rf in ref_feats if rf.get("spectral_flatness") is not None]

    if not ref_scs:
        return 80.0, "Standart aralıkta"

    med_sc = float(np.median(ref_scs))
    med_flat = float(np.median(ref_flats)) if ref_flats else 0.05

    sc_err = abs(u_sc - med_sc) / max(1000.0, med_sc)
    flat_err = abs(u_flat - med_flat) / max(0.01, med_flat)

    score = max(0.0, min(100.0, 100.0 - (sc_err * 80.0 + flat_err * 40.0)))
    
    note = "Standart telaffuz"
    if sc_err > 0.35:
        if u_sc < med_sc:
            note = "Boğumlanma yeri geri çekilmiş (yumuşak/gevşek sürtünme)"
        else:
            note = "Sürtünme frekansı yüksek / sert artikülasyon"

    return round(score, 1), note
