"""Duration and prosody (F0 contour, rhythm, intonation) analysis module.

Normalizes speech rate and speaker pitch to evaluate rhythm and intonation
without penalizing individual speaker natural voice traits.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import parselmouth

from speech_eval.aligner import AlignedSegment


class ProsodyDurationAnalyzer:
    """Analyzes normalized phoneme duration, speech rate, and intonation contour."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze_durations(
        self,
        user_segments: List[AlignedSegment],
        ref_segments_list: List[List[AlignedSegment]]
    ) -> List[Dict[str, Any]]:
        """Calculates rate-normalized relative duration scores for each phoneme.
        
        duration_ratio = phoneme_duration / speaker_average_phoneme_duration
        """
        if not user_segments:
            return []

        # Calculate user overall speaking rate
        user_total_dur = sum(s.duration for s in user_segments)
        user_mean_dur = user_total_dur / max(1, len(user_segments))

        # Calculate reference average phoneme durations
        ref_means = []
        for ref_segs in ref_segments_list:
            if ref_segs:
                ref_means.append(sum(s.duration for s in ref_segs) / len(ref_segs))
        overall_ref_mean = float(np.mean(ref_means)) if ref_means else 0.080

        duration_results = []

        for i, u_seg in enumerate(user_segments):
            # User relative duration
            u_rel_dur = u_seg.duration / max(0.01, user_mean_dur)

            # Reference relative durations for this specific phoneme index
            ref_rel_durs = []
            for ref_segs in ref_segments_list:
                if i < len(ref_segs):
                    ref_rel = ref_segs[i].duration / max(0.01, overall_ref_mean)
                    ref_rel_durs.append(ref_rel)

            med_ref_rel = float(np.median(ref_rel_durs)) if ref_rel_durs else 1.0

            # Error ratio
            dur_error = abs(u_rel_dur - med_ref_rel) / max(0.3, med_ref_rel)
            dur_score = max(0.0, min(100.0, 100.0 - (dur_error * 70.0)))

            duration_results.append({
                "raw_duration": u_seg.duration,
                "relative_duration": round(u_rel_dur, 2),
                "ref_relative_duration": round(med_ref_rel, 2),
                "duration_score": round(dur_score, 1)
            })

        return duration_results

    def analyze_intonation(
        self,
        audio: np.ndarray,
        user_segments: List[AlignedSegment]
    ) -> Tuple[float, List[float]]:
        """Analyzes sentence intonation contour normalized to semitones from median pitch.
        
        Returns:
            (overall_intonation_score 0-100, normalized_pitch_contour)
        """
        if len(audio) < self.sr * 0.3:
            return 80.0, []

        sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
        pitch = sound.to_pitch(time_step=0.02)
        f0_values = pitch.selected_array['frequency']
        voiced_f0 = f0_values[f0_values > 0]

        if len(voiced_f0) < 5:
            return 80.0, []

        median_f0 = float(np.median(voiced_f0))

        # Convert to semitones relative to speaker's own median pitch
        # Semitones = 12 * log2(F0 / median_F0)
        norm_pitch = []
        for f in f0_values:
            if f > 0 and median_f0 > 0:
                st = 12.0 * np.log2(f / median_f0)
                norm_pitch.append(round(float(st), 2))
            else:
                norm_pitch.append(0.0)

        # In natural Turkish statement sentences:
        # Standard pitch contour has moderate dynamic range (typically ±4-6 semitones)
        # and non-monotone declination. Unnatural robotic flat speech or extreme wild octave jumps reduce score.
        voiced_st = [st for st, f in zip(norm_pitch, f0_values) if f > 0]
        if not voiced_st:
            return 80.0, norm_pitch

        st_std = float(np.std(voiced_st))
        
        # Optimal standard Turkish speech standard deviation in semitones is around 1.8 to 3.5 st
        if 1.5 <= st_std <= 4.0:
            intonation_score = 90.0 + (10.0 - abs(st_std - 2.5) * 5.0)
        elif st_std < 1.5:
            # Monotone / flat
            intonation_score = max(50.0, 90.0 - (1.5 - st_std) * 30.0)
        else:
            # Excessive pitch fluctuation
            intonation_score = max(50.0, 90.0 - (st_std - 4.0) * 15.0)

        return round(float(np.clip(intonation_score, 0.0, 100.0)), 1), norm_pitch
