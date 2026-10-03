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


def test_observed_empty_tokens_uncertain():
    """Verify that if observed_tokens is completely empty, it returns status='uncertain' with multiplier=1.0."""
    g2p = TurkishG2P()
    analyzer = ObservedPhonemeAnalyzer(tokenizer=None)

    phonemes = g2p.convert("merhaba")
    diags = analyzer.align_sequences(phonemes, [])

    assert len(diags) == len(phonemes)
    for d in diags:
        assert d.status == "uncertain"
        assert d.score_multiplier == 1.0
        assert "yetersiz" in d.note or "belirsiz" in d.note


def test_observed_allophonic_and_close_classes():
    """Verify distinct behavior between allophonic equivalents, phonetically close pairs, and real substitutions."""
    g2p = TurkishG2P()
    analyzer = ObservedPhonemeAnalyzer(tokenizer=None)

    # 1. Allophonic equivalent: 'ğ' -> 'y' in 'değil' (or 'k' -> 'c')
    phonemes_degil = g2p.convert("değil")
    tokens_degil = [
        ObservedToken("d", 0, 0.95),
        ObservedToken("e", 1, 0.95),
        ObservedToken("y", 2, 0.95),  # ğ pronounced as glide y
        ObservedToken("i", 3, 0.95),
        ObservedToken("l", 4, 0.95),
    ]
    diags_degil = analyzer.align_sequences(phonemes_degil, tokens_degil)
    assert diags_degil[2].status == "match"
    assert diags_degil[2].score_multiplier == 1.0
    assert "allofon" in diags_degil[2].note

    # 2. Phonetically close: 't' -> 'd' (voicing variation in 'at')
    phonemes_at = g2p.convert("at")
    tokens_at = [
        ObservedToken("a", 0, 0.95),
        ObservedToken("d", 1, 0.90),
    ]
    diags_at = analyzer.align_sequences(phonemes_at, tokens_at)
    assert diags_at[1].status == "similar"
    assert diags_at[1].score_multiplier == 0.90  # mild close penalty

    # 3. Real substitution: 'm' -> 'n' (tam -> tan)
    phonemes_tam = g2p.convert("tam")
    tokens_tan = [
        ObservedToken("t", 0, 0.95),
        ObservedToken("a", 1, 0.95),
        ObservedToken("n", 2, 0.90),  # m replaced by n
    ]
    diags_tam = analyzer.align_sequences(phonemes_tam, tokens_tan)
    assert diags_tam[2].status == "substitution"
    assert diags_tam[2].score_multiplier == 0.75  # full real substitution penalty

    # 4. Real substitution: 'a' -> 'u'
    phonemes_al = g2p.convert("al")
    tokens_ul = [
        ObservedToken("u", 0, 0.90),
        ObservedToken("l", 1, 0.95),
    ]
    diags_al = analyzer.align_sequences(phonemes_al, tokens_ul)
    assert diags_al[0].status == "substitution"
    assert diags_al[0].score_multiplier == 0.75

    # 5. Orthographic 'c' for expected 'k' is a real substitution (/dʒ/ vs /k/), NOT an allophone!
    phonemes_kar = g2p.convert("kar")
    tokens_car = [
        ObservedToken("c", 0, 0.95),  # spoken as "car"
        ObservedToken("a", 1, 0.95),
        ObservedToken("r", 2, 0.95),
    ]
    diags_car = analyzer.align_sequences(phonemes_kar, tokens_car)
    assert diags_car[0].status == "substitution"
    assert diags_car[0].score_multiplier == 0.75


def test_observed_unconstrained_deletion_no_circularity():
    """Verify that deletion detection does NOT falsely rely on circular forced alignment seg_conf.
    When unconstrained logits lack the expected token and a competing token dominates,
    deletion is correctly detected even if forced alignment seg_conf was arbitrarily high."""
    import torch
    from types import SimpleNamespace

    # Mock tokenizer mapping
    class MockTokenizer:
        def convert_tokens_to_ids(self, char):
            vocab = {"pad": 0, "g": 1, "i": 2, "d": 3, "e": 4, "c": 5, "m": 6}
            return vocab.get(char, 99)

        unk_token_id = 99

    analyzer = ObservedPhonemeAnalyzer(tokenizer=MockTokenizer())
    g2p = TurkishG2P()
    # Expected: "gideceğim" -> phonemes include 'i' (char 7)
    phonemes = g2p.convert("gideceğim")

    # Spoken audio only contains "gidecem" with unconstrained frame spans
    tokens = [
        ObservedToken("g", 0, 0.95, end_frame_idx=1),
        ObservedToken("i", 1, 0.95, end_frame_idx=2),
        ObservedToken("d", 2, 0.95, end_frame_idx=3),
        ObservedToken("e", 3, 0.95, end_frame_idx=4),
        ObservedToken("c", 4, 0.95, end_frame_idx=5),
        ObservedToken("e", 5, 0.95, end_frame_idx=8),
        ObservedToken("m", 8, 0.95, end_frame_idx=10),
    ]

    # Create dummy logits [1, T=10, vocab=20] where token 'i' (id=2) has ZERO presence,
    # and token 'e' (id=4) and 'm' (id=6) dominate with 0.90 posterior
    logits = torch.full((1, 10, 20), -10.0)
    logits[0, 5:8, 4] = 10.0   # token 'e' strongly active
    logits[0, 8:10, 6] = 10.0  # token 'm' strongly active

    # Mock forced-alignment segment with artificially high confidence (the circular trap!)
    mock_segments = [
        SimpleNamespace(start_frame=i, end_frame=i+1, duration=0.06, confidence=0.98)
        for i in range(len(phonemes))
    ]

    diags = analyzer.align_sequences(
        expected_phonemes=phonemes,
        observed_tokens=tokens,
        segments=mock_segments,
        logits=logits
    )

    # The 'i' phoneme must be detected as deletion despite mock_segments confidence=0.98!
    deleted_i = [d for d in diags if d.expected_char == 'i' and d.status == "deletion"]
    assert len(deleted_i) >= 1
    assert deleted_i[0].score_multiplier == 0.75


