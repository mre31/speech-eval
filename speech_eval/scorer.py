"""Multi-metric scoring engine with outlier filtering and robust aggregation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from speech_eval.config import EvalConfig
from speech_eval.g2p import PhonemeItem


@dataclass
class PhonemeScoreDetail:
    grapheme: str
    ipa: str
    phoneme_key: str
    word: str
    word_index: int
    start_time: float
    end_time: float
    duration: float
    alignment_confidence: float
    is_valid: bool
    embedding_score: float
    acoustic_score: Optional[float]
    acoustic_available: bool
    effective_score: float
    duration_score: float
    prosody_score: float
    total_score: float
    diagnostic_note: str


@dataclass
class WordScoreDetail:
    word: str
    word_index: int
    score: float
    phonemes: List[PhonemeScoreDetail] = field(default_factory=list)


@dataclass
class EvaluationResult:
    overall_score: float
    pronunciation_score: float
    vowel_score: float
    consonant_score: float
    rhythm_score: float
    intonation_score: float
    recognized_text: str
    expected_text: str
    problematic_phonemes: List[Dict[str, Any]]
    word_scores: List[WordScoreDetail]
    phoneme_scores: List[PhonemeScoreDetail]


class SpeechScorer:
    """Combines phonetic, acoustic, duration, and prosodic metrics into weighted scores."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()

    def score_single_phoneme(
        self,
        phoneme: PhonemeItem,
        duration: float,
        confidence: float,
        embedding_score: float,
        acoustic_score: Optional[float],
        duration_score: float,
        prosody_score: float,
        diagnostic_note: str = "",
        score_multiplier: float = 1.0
    ) -> PhonemeScoreDetail:
        """Computes weighted single phoneme score and assesses validity."""
        # Quality & outlier checks
        is_valid = True
        notes = []

        if confidence < self.config.min_alignment_confidence:
            is_valid = False
            notes.append("Düşük hizalama güveni")

        if duration < self.config.min_phoneme_duration_sec:
            is_valid = False
            notes.append("Çok kısa ses segmenti")

        # If acoustic score is None (e.g. formants could not be measured by Praat),
        # redistribute its weight to phonetic embedding score to avoid penalizing the user.
        # UI data retains acoustic_score = None and acoustic_available = False for honest reporting.
        if acoustic_score is None:
            eff_acoustic = embedding_score
            acoustic_available = False
            raw_acoustic = None
        else:
            eff_acoustic = acoustic_score
            acoustic_available = True
            raw_acoustic = round(acoustic_score, 1)

        # Weighted score combination uses eff_acoustic
        raw_total = (
            self.config.embedding_weight * embedding_score +
            self.config.acoustic_weight * eff_acoustic +
            self.config.duration_weight * duration_score +
            self.config.prosody_weight * prosody_score
        )

        # Apply observed pronunciation multiplier (e.g. dropped/elided sounds or substitutions)
        total = raw_total * score_multiplier
        if score_multiplier < 0.95:
            embedding_score = embedding_score * score_multiplier

        final_note = diagnostic_note if diagnostic_note else ("; ".join(notes) if notes else "Standart telaffuz")

        return PhonemeScoreDetail(
            grapheme=phoneme.grapheme,
            ipa=phoneme.ipa,
            phoneme_key=phoneme.phoneme_key,
            word=phoneme.word,
            word_index=phoneme.word_index,
            start_time=0.0,
            end_time=0.0,
            duration=round(duration, 4),
            alignment_confidence=round(confidence, 3),
            is_valid=is_valid,
            embedding_score=round(embedding_score, 1),
            acoustic_score=raw_acoustic,
            acoustic_available=acoustic_available,
            effective_score=round(eff_acoustic, 1),
            duration_score=round(duration_score, 1),
            prosody_score=round(prosody_score, 1),
            total_score=round(float(np.clip(total, 0.0, 100.0)), 1),
            diagnostic_note=final_note
        )

    def aggregate_results(
        self,
        phoneme_scores: List[PhonemeScoreDetail],
        intonation_score: float,
        recognized_text: str,
        expected_text: str
    ) -> EvaluationResult:
        """Performs confidence-weighted robust aggregation over all phonemes."""
        valid_items = [p for p in phoneme_scores if p.is_valid]
        if not valid_items:
            # Fallback to all items if all were flagged
            valid_items = phoneme_scores

        if not valid_items:
            return EvaluationResult(
                overall_score=0.0,
                pronunciation_score=0.0,
                vowel_score=0.0,
                consonant_score=0.0,
                rhythm_score=0.0,
                intonation_score=0.0,
                recognized_text=recognized_text,
                expected_text=expected_text,
                problematic_phonemes=[],
                word_scores=[],
                phoneme_scores=phoneme_scores
            )

        # Robust trimmed mean calculation helper
        def robust_weighted_mean(scores: List[float], weights: List[float]) -> float:
            if not scores:
                return 80.0
            if len(scores) <= 4:
                return float(np.average(scores, weights=weights))
            
            # Trim extreme top and bottom outliers according to config trim ratio
            trim_k = max(1, int(len(scores) * self.config.outlier_trim_ratio))
            sorted_pairs = sorted(zip(scores, weights), key=lambda x: x[0])
            trimmed_pairs = sorted_pairs[trim_k:-trim_k]
            
            if not trimmed_pairs:
                trimmed_pairs = sorted_pairs

            sub_scores = [p[0] for p in trimmed_pairs]
            sub_weights = [p[1] for p in trimmed_pairs]
            return float(np.average(sub_scores, weights=sub_weights))

        all_scores = [p.total_score for p in valid_items]
        all_weights = [max(0.1, p.alignment_confidence) for p in valid_items]

        overall_raw = robust_weighted_mean(all_scores, all_weights)

        # Category sub-scores
        emb_scores = [p.embedding_score for p in valid_items]
        pronunciation = robust_weighted_mean(emb_scores, all_weights)

        # Subscore aggregation filters only available acoustic measurements
        vowel_items = [p for p in valid_items if p.grapheme in {'a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü'}]
        vowel_meas = [p for p in vowel_items if p.acoustic_available and p.acoustic_score is not None]
        if vowel_meas:
            vowel_scores = [p.acoustic_score for p in vowel_meas]
            vowel_weights = [max(0.1, p.alignment_confidence) for p in vowel_meas]
            vowels = robust_weighted_mean(vowel_scores, vowel_weights)
        elif vowel_items:
            vowels = robust_weighted_mean(
                [p.embedding_score for p in vowel_items],
                [max(0.1, p.alignment_confidence) for p in vowel_items]
            )
        else:
            vowels = pronunciation

        cons_items = [p for p in valid_items if p.grapheme not in {'a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü'}]
        cons_meas = [p for p in cons_items if p.acoustic_available and p.acoustic_score is not None]
        if cons_meas:
            cons_scores = [p.acoustic_score for p in cons_meas]
            cons_weights = [max(0.1, p.alignment_confidence) for p in cons_meas]
            consonants = robust_weighted_mean(cons_scores, cons_weights)
        elif cons_items:
            consonants = robust_weighted_mean(
                [p.embedding_score for p in cons_items],
                [max(0.1, p.alignment_confidence) for p in cons_items]
            )
        else:
            consonants = pronunciation


        rhythm_scores = [p.duration_score for p in valid_items]
        rhythm = robust_weighted_mean(rhythm_scores, all_weights)

        # Word-level aggregation
        word_map: Dict[int, List[PhonemeScoreDetail]] = {}
        for p in phoneme_scores:
            word_map.setdefault(p.word_index, []).append(p)

        word_scores_list: List[WordScoreDetail] = []
        for w_idx in sorted(word_map.keys()):
            w_phonemes = word_map[w_idx]
            w_valid = [p for p in w_phonemes if p.is_valid] or w_phonemes
            w_score = float(np.mean([p.total_score for p in w_valid]))
            word_str = w_phonemes[0].word
            word_scores_list.append(WordScoreDetail(
                word=word_str,
                word_index=w_idx,
                score=round(w_score, 1),
                phonemes=w_phonemes
            ))

        # Problematic phonemes: group by phoneme key, find phonemes with lowest scores
        phoneme_groups: Dict[str, List[PhonemeScoreDetail]] = {}
        for p in valid_items:
            phoneme_groups.setdefault(p.phoneme_key, []).append(p)

        problematic: List[Dict[str, Any]] = []
        for p_key, p_list in phoneme_groups.items():
            avg_score = float(np.mean([p.total_score for p in p_list]))
            if avg_score < 75.0:  # Threshold for noticeable deviation
                # Collect most common diagnostic notes
                notes = [p.diagnostic_note for p in p_list if p.diagnostic_note and p.diagnostic_note != "Standart telaffuz"]
                note = notes[0] if notes else "Standart dağılımdan sapma"
                problematic.append({
                    "phoneme": p_key,
                    "grapheme": p_list[0].grapheme,
                    "score": round(avg_score, 1),
                    "count": len(p_list),
                    "diagnostic": note
                })

        # Sort problematic phonemes by score ascending
        problematic.sort(key=lambda x: x["score"])

        return EvaluationResult(
            overall_score=round(overall_raw, 1),
            pronunciation_score=round(pronunciation, 1),
            vowel_score=round(vowels, 1),
            consonant_score=round(consonants, 1),
            rhythm_score=round(rhythm, 1),
            intonation_score=round(intonation_score, 1),
            recognized_text=recognized_text,
            expected_text=expected_text,
            problematic_phonemes=problematic,
            word_scores=word_scores_list,
            phoneme_scores=phoneme_scores
        )
