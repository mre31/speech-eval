"""Unit tests for Turkish G2P."""

from speech_eval.g2p import TurkishG2P, FRONT_VOWELS, BACK_VOWELS


def test_g2p_basic_vowels():
    g2p = TurkishG2P()
    phonemes = g2p.convert("bugün eve gidiyorum")
    graphemes = [p.grapheme for p in phonemes]
    assert "b" in graphemes
    assert "ü" in graphemes
    assert "e" in graphemes

    u_phonemes = [p for p in phonemes if p.grapheme == "ü"]
    assert len(u_phonemes) == 1
    assert u_phonemes[0].ipa == "y"
    assert u_phonemes[0].is_vowel is True
    assert u_phonemes[0].vowel_frontness == "front"


def test_g2p_allophones():
    g2p = TurkishG2P()
    # 'kar' has back vowel -> velar [k]
    # 'kedi' has front vowel -> palatal [c]
    p_kar = g2p.convert("kar")
    p_kedi = g2p.convert("kedi")

    k_kar = [p for p in p_kar if p.grapheme == "k"][0]
    k_kedi = [p for p in p_kedi if p.grapheme == "k"][0]

    assert k_kar.ipa == "k"
    assert k_kar.consonant_place == "velar"

    assert k_kedi.ipa == "c"
    assert k_kedi.consonant_place == "palatal"


def test_g2p_yumusak_g():
    g2p = TurkishG2P()
    # 'dağ' has back vowel -> lengthening /ː/
    p_dag = g2p.convert("dağ")
    g_dag = [p for p in p_dag if p.grapheme == "ğ"][0]
    assert g_dag.ipa == "ː"

    # 'eğri' has front vowel -> glide /j/
    p_egri = g2p.convert("eğri")
    g_egri = [p for p in p_egri if p.grapheme == "ğ"][0]
    assert g_egri.ipa == "j"
