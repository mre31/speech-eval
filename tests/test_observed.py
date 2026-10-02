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
