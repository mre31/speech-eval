"""Unit tests for scoring and robust aggregation."""

from speech_eval.scorer import SpeechScorer
from speech_eval.g2p import PhonemeItem


def test_scorer_aggregation():
    scorer = SpeechScorer()
    
    p_a = PhonemeItem(
        grapheme="a", ipa="a", phoneme_key="/a/", word="bak", word_index=0,
        char_index_in_word=1, is_vowel=True
    )
    p_b = PhonemeItem(
        grapheme="b", ipa="b", phoneme_key="/b/", word="bak", word_index=0,
        char_index_in_word=0, is_vowel=False
    )

    detail_a = scorer.score_single_phoneme(
        phoneme=p_a,
        duration=0.08,
        confidence=0.95,
        embedding_score=90.0,
        acoustic_score=88.0,
        duration_score=92.0,
        prosody_score=85.0
    )

    detail_b = scorer.score_single_phoneme(
        phoneme=p_b,
        duration=0.07,
        confidence=0.90,
        embedding_score=85.0,
        acoustic_score=82.0,
        duration_score=88.0,
        prosody_score=85.0
    )

    result = scorer.aggregate_results(
        phoneme_scores=[detail_b, detail_a],
        intonation_score=85.0,
        recognized_text="bak",
        expected_text="bak"
    )

    assert result.overall_score >= 80.0
    assert len(result.word_scores) == 1
    assert result.word_scores[0].word == "bak"
    assert result.vowel_score > 0.0
    assert result.consonant_score > 0.0


def test_scorer_outlier_filtering():
    scorer = SpeechScorer()
    
    p = PhonemeItem(
        grapheme="e", ipa="e", phoneme_key="/e/", word="ev", word_index=0,
        char_index_in_word=0, is_vowel=True
    )

    # Low alignment confidence should be marked invalid
    detail_bad = scorer.score_single_phoneme(
        phoneme=p,
        duration=0.05,
        confidence=0.05,  # below min threshold 0.15
        embedding_score=50.0,
        acoustic_score=50.0,
        duration_score=50.0,
        prosody_score=50.0
    )

    assert detail_bad.is_valid is False
    assert "Düşük hizalama güveni" in detail_bad.diagnostic_note


def test_scorer_unavailable_acoustic():
    """Verify that when acoustic measurement fails, acoustic_score is None, acoustic_available is False,
    and effective_score redistributes weight to embedding_score without penalizing the speaker."""
    scorer = SpeechScorer()
    
    p = PhonemeItem(
        grapheme="a", ipa="a", phoneme_key="/a/", word="at", word_index=0,
        char_index_in_word=0, is_vowel=True
    )

    detail = scorer.score_single_phoneme(
        phoneme=p,
        duration=0.08,
        confidence=0.95,
        embedding_score=94.0,
        acoustic_score=None,  # Formant measurement failed / Praat couldn't track
        duration_score=92.0,
        prosody_score=90.0
    )

    assert detail.acoustic_score is None
    assert detail.acoustic_available is False
    assert detail.effective_score == 94.0
    assert detail.total_score >= 90.0
