"""Unit tests for the Observed Pronunciation layer."""

import pytest
from speech_eval.g2p import TurkishG2P
from speech_eval.observed import ObservedPhonemeAnalyzer, ObservedToken


def test_observed_exact_match():
    g2p = TurkishG2P()
    analyzer = ObservedPhonemeAnalyzer(tokenizer=None)

    phonemes = g2p.convert("bugün")
    tokens = [ObservedToken(p.grapheme, i, 0.95) for i, p in enumerate(phonemes)]
    diags = analyzer.align_sequences(phonemes, tokens)

    assert len(diags) == len(phonemes)
    for d in diags:
        assert d.status == "match"
        assert d.score_multiplier == 1.0


def test_observed_elision_detection():
    """Verify that dropped phonemes (e.g. 'gidecem' for 'gideceğim') are detected as deletions."""
    g2p = TurkishG2P()
    analyzer = ObservedPhonemeAnalyzer(tokenizer=None)

    phonemes = g2p.convert("gideceğim")
    # Spoken: g-i-d-e-c-e-m (omits ğ and i)
    tokens = [
        ObservedToken("g", 0, 0.95),
        ObservedToken("i", 1, 0.95),
        ObservedToken("d", 2, 0.95),
        ObservedToken("e", 3, 0.95),
        ObservedToken("c", 4, 0.95),
        ObservedToken("e", 5, 0.95),
        ObservedToken("m", 6, 0.95),
    ]
    diags = analyzer.align_sequences(phonemes, tokens)

    # Check that deleted 'i' in suffix was detected as deletion with elision note
    deleted_items = [d for d in diags if d.status == "deletion"]
    assert len(deleted_items) >= 1
    assert any("elision" in d.note or "düşmesi" in d.note for d in deleted_items)


def test_observed_confidence_gating():
    """Verify that low-confidence (<0.50) CTC tokens are not penalized, and 0.50-0.70 are diagnostic only."""
    g2p = TurkishG2P()
    analyzer = ObservedPhonemeAnalyzer(tokenizer=None)

    phonemes = g2p.convert("para")
    
    # 1. Low confidence substitution (<0.50) e.g. token 'k' with conf=0.35
    tokens_low = [
        ObservedToken("k", 0, 0.35),
        ObservedToken("a", 1, 0.95),
        ObservedToken("r", 2, 0.95),
        ObservedToken("a", 3, 0.95),
    ]
    diags_low = analyzer.align_sequences(phonemes, tokens_low)
    # The first phoneme 'p' should have score_multiplier == 1.0 (no penalty for noisy frame)
    assert diags_low[0].score_multiplier == 1.0
    assert "yetersiz kanıt" in diags_low[0].note

    # 2. Mid confidence substitution (0.50 - 0.70) e.g. token 'k' with conf=0.60
    tokens_mid = [
        ObservedToken("k", 0, 0.60),
        ObservedToken("a", 1, 0.95),
        ObservedToken("r", 2, 0.95),
        ObservedToken("a", 3, 0.95),
    ]
    diags_mid = analyzer.align_sequences(phonemes, tokens_mid)
    # Diagnostic only, multiplier remains 1.0
    assert diags_mid[0].score_multiplier == 1.0
    assert diags_mid[0].status == "substitution"
    assert "gözlem" in diags_mid[0].note or "Olası" in diags_mid[0].note

    # 3. High confidence substitution (>0.85) e.g. token 'k' with conf=0.92
    tokens_high = [
        ObservedToken("k", 0, 0.92),
        ObservedToken("a", 1, 0.95),
        ObservedToken("r", 2, 0.95),
        ObservedToken("a", 3, 0.95),
    ]
    diags_high = analyzer.align_sequences(phonemes, tokens_high)
    assert diags_high[0].status == "substitution"
    assert diags_high[0].score_multiplier < 0.90


def test_observed_yumusak_g_standard_lengthening():
    """Verify that yumuşak g ('ğ') omitted in greedy CTC is treated as standard vowel lengthening."""
    g2p = TurkishG2P()
    analyzer = ObservedPhonemeAnalyzer(tokenizer=None)

    phonemes = g2p.convert("dağ")  # d - a - ğ
    # In standard speech, greedy CTC often outputs only 'd' and 'a'
    tokens = [
        ObservedToken("d", 0, 0.95),
        ObservedToken("a", 1, 0.95),
    ]
    diags = analyzer.align_sequences(phonemes, tokens)
    g_diag = diags[2]
    assert g_diag.expected_char == 'ğ'
    assert g_diag.score_multiplier == 1.0
    assert "uzatma" in g_diag.note
