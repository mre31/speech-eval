"""Duration and prosody (speech rate, rhythm, F0 contour, intonation) analysis module.

Normalizes speech rate and speaker pitch to evaluate rhythm and intonation
against standard Turkish references without penalizing speaker pitch register or voice type.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import parselmouth

from speech_eval.aligner import AlignedSegment


class ProsodyDurationAnalyzer:
    """Analyzes rate-normalized phoneme duration, rhythm, and intonation contour."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze_durations(
        self,
        user_segments: List[AlignedSegment],
        ref_segments_list: List[List[AlignedSegment]]
    ) -> List[Dict[str, Any]]:
        """Calculates rate-normalized relative duration scores for each phoneme.
        
        duration_ratio = phoneme_duration / speaker_average_phoneme_duration
        Each reference speaker is normalized by that specific speaker's own speech rate.
        """
        if not user_segments:
            return []

        # User overall average phoneme duration
        user_total_dur = sum(s.duration for s in user_segments)
        user_mean_dur = user_total_dur / max(1, len(user_segments))

        # Each reference speaker's individual average phoneme duration
        ref_speaker_means = []
        for ref_segs in ref_segments_list:
            if ref_segs:
                spk_mean = sum(s.duration for s in ref_segs) / len(ref_segs)
                ref_speaker_means.append(max(0.01, spk_mean))
            else:
                ref_speaker_means.append(0.080)

        duration_results = []

        for i, u_seg in enumerate(user_segments):
            # User rate-normalized relative duration
            u_rel_dur = u_seg.duration / max(0.01, user_mean_dur)

            # Reference relative durations (each divided by that speaker's own mean)
            ref_rel_durs = []
            for r_idx, ref_segs in enumerate(ref_segments_list):
                if i < len(ref_segs):
                    ref_spk_mean = ref_speaker_means[r_idx]
                    ref_rel = ref_segs[i].duration / ref_spk_mean
                    ref_rel_durs.append(ref_rel)

            med_ref_rel = float(np.median(ref_rel_durs)) if ref_rel_durs else 1.0
            sigma_ref_rel = max(0.20, float(np.std(ref_rel_durs))) if len(ref_rel_durs) > 1 else 0.25

            # Error relative to standard reference median, with variance tolerance
            dur_error = abs(u_rel_dur - med_ref_rel)
            
            # Errors within 0.35 ratio difference or 1.5 sigma are completely natural
            z_dur = max(0.0, dur_error - 0.25) / sigma_ref_rel
            dur_score = max(40.0, min(100.0, 100.0 - z_dur * 30.0))

            duration_results.append({
                "raw_duration": round(u_seg.duration, 4),
                "relative_duration": round(u_rel_dur, 2),
                "ref_relative_duration": round(med_ref_rel, 2),
                "duration_score": round(dur_score, 1)
            })

        return duration_results

    def extract_phoneme_pitch(
        self,
        audio: np.ndarray,
        segments: List[AlignedSegment]
    ) -> List[Optional[float]]:
        """Extracts median pitch in semitones (relative to utterance median) for each phoneme."""
        if len(audio) < self.sr * 0.2:
            return [None] * len(segments)

        try:
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch = sound.to_pitch(time_step=0.01)
            f0_arr = pitch.selected_array['frequency']
            voiced = f0_arr[f0_arr > 0]
            if len(voiced) < 5:
                return [None] * len(segments)
            
            median_f0 = float(np.median(voiced))
            
            phoneme_pitches: List[Optional[float]] = []
            for seg in segments:
                t_mid = (seg.start_time + seg.end_time) / 2.0
                p_val = pitch.get_value_at_time(t_mid)
                if not np.isnan(p_val) and p_val > 50.0:
                    st = 12.0 * np.log2(p_val / median_f0)
                    phoneme_pitches.append(float(st))
                else:
                    phoneme_pitches.append(None)
            return phoneme_pitches
        except Exception:
            return [None] * len(segments)

    def analyze_intonation(
        self,
        user_audio: np.ndarray,
        user_segments: List[AlignedSegment],
        ref_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, List[float]]:
        """Analyzes sentence intonation contour compared with standard Turkish reference trajectories.
        
        Evaluates:
        1. Pitch contour similarity (rising/falling sentence intonation tune)
        2. Mean semitone trajectory deviation
        3. Dynamic range naturalness
        
        Returns:
            (overall_intonation_score 0-100, normalized_pitch_contour)
        """
        if len(user_audio) < self.sr * 0.3:
            return 85.0, []

        sound = parselmouth.Sound(user_audio, sampling_frequency=self.sr)
        pitch = sound.to_pitch(time_step=0.02)
        f0_values = pitch.selected_array['frequency']
        voiced_f0 = f0_values[f0_values > 0]

        if len(voiced_f0) < 5:
            return 85.0, []

        median_f0 = float(np.median(voiced_f0))

        # Normalized pitch contour in semitones relative to utterance median
        norm_pitch = []
        for f in f0_values:
            if f > 0 and median_f0 > 0:
                st = 12.0 * np.log2(f / median_f0)
                norm_pitch.append(round(float(st), 2))
            else:
                norm_pitch.append(0.0)

        voiced_st = [st for st, f in zip(norm_pitch, f0_values) if f > 0]
        st_std = float(np.std(voiced_st)) if voiced_st else 2.5

        # 1. Base range score (standard Turkish statement range is ~1.5 to 4.5 semitones std)
        if 1.4 <= st_std <= 4.2:
            range_score = 95.0
        elif st_std < 1.4:
            range_score = max(55.0, 95.0 - (1.4 - st_std) * 35.0)
        else:
            range_score = max(60.0, 95.0 - (st_std - 4.2) * 20.0)

        # 2. Reference contour comparison (if references are provided)
        contour_score = range_score
        if ref_data and user_segments:
            user_ph_pitches = self.extract_phoneme_pitch(user_audio, user_segments)
            
            # Extract phoneme pitches across reference audios
            all_ref_pitches = []
            for r in ref_data:
                if "audio" in r and "segments" in r:
                    r_pitches = self.extract_phoneme_pitch(r["audio"], r["segments"])
                    all_ref_pitches.append(r_pitches)

            # Compute median reference pitch per phoneme
            med_ref_pitches: List[Optional[float]] = []
            for i in range(len(user_segments)):
                vals = [rp[i] for rp in all_ref_pitches if i < len(rp) and rp[i] is not None]
                med_ref_pitches.append(float(np.median(vals)) if vals else None)

            # Compare pairs where both user and reference have voiced pitch
            valid_pairs = [
                (u, r) for u, r in zip(user_ph_pitches, med_ref_pitches)
                if u is not None and r is not None
            ]

            if len(valid_pairs) >= 3:
                u_arr = np.array([p[0] for p in valid_pairs])
                r_arr = np.array([p[1] for p in valid_pairs])

                # Mean absolute semitone error
                mae = float(np.mean(np.abs(u_arr - r_arr)))
                mae_score = max(50.0, min(100.0, 100.0 - max(0.0, mae - 2.0) * 18.0))

                # Pearson correlation of contour direction
                if np.std(u_arr) > 0.3 and np.std(r_arr) > 0.3:
                    corr = float(np.corrcoef(u_arr, r_arr)[0, 1])
                    if np.isnan(corr):
                        corr = 0.5
                    corr_score = max(50.0, min(100.0, 75.0 + corr * 25.0))
                else:
                    corr_score = 88.0

                contour_score = 0.5 * mae_score + 0.5 * corr_score

        # Final intonation score combines dynamic range naturalness and contour trajectory
        final_score = 0.4 * range_score + 0.6 * contour_score
        return round(float(np.clip(final_score, 0.0, 100.0)), 1), norm_pitch
