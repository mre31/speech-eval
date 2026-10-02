"""Grapheme-to-Phoneme (G2P) converter for Turkish.

Converts standard Turkish orthographic text into detailed phonemic/IPA representations.
Maintains phonological properties (front/back vowel, vowel height, rounding, consonant manner & place).
Provides alignment tokens for forced alignment and phonetic keys for acoustic/embedding comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class PhonemeItem:
    grapheme: str
    ipa: str
    phoneme_key: str
    word: str
    word_index: int
    char_index_in_word: int
    is_vowel: bool
    vowel_frontness: Optional[str] = None  # 'front', 'back', 'central'
    vowel_height: Optional[str] = None     # 'close', 'mid', 'open'
    vowel_rounded: Optional[bool] = None
    consonant_manner: Optional[str] = None  # 'plosive', 'fricative', 'affricate', 'nasal', 'liquid', 'glide'
    consonant_place: Optional[str] = None   # 'bilabial', 'labiodental', 'alveolar', 'palatal', 'velar', 'glottal'
    consonant_voiced: Optional[bool] = None


FRONT_VOWELS: Set[str] = {'e', 'i', 'ö', 'ü'}
BACK_VOWELS: Set[str] = {'a', 'ı', 'o', 'u'}
ALL_VOWELS: Set[str] = FRONT_VOWELS | BACK_VOWELS


# Standard Turkish IPA mapping table
VOWEL_PROPERTIES = {
    'a': {'ipa': 'a', 'frontness': 'central', 'height': 'open', 'rounded': False},
    'e': {'ipa': 'e', 'frontness': 'front', 'height': 'mid', 'rounded': False},
    'ı': {'ipa': 'ɯ', 'frontness': 'back', 'height': 'close', 'rounded': False},
    'i': {'ipa': 'i', 'frontness': 'front', 'height': 'close', 'rounded': False},
    'o': {'ipa': 'o', 'frontness': 'back', 'height': 'mid', 'rounded': True},
    'ö': {'ipa': 'ø', 'frontness': 'front', 'height': 'mid', 'rounded': True},
    'u': {'ipa': 'u', 'frontness': 'back', 'height': 'close', 'rounded': True},
    'ü': {'ipa': 'y', 'frontness': 'front', 'height': 'close', 'rounded': True},
}

CONSONANT_PROPERTIES = {
    'b': {'ipa': 'b', 'manner': 'plosive', 'place': 'bilabial', 'voiced': True},
    'c': {'ipa': 'd͡ʒ', 'manner': 'affricate', 'place': 'palato-alveolar', 'voiced': True},
    'ç': {'ipa': 't͡ʃ', 'manner': 'affricate', 'place': 'palato-alveolar', 'voiced': False},
    'd': {'ipa': 'd', 'manner': 'plosive', 'place': 'alveolar', 'voiced': True},
    'f': {'ipa': 'f', 'manner': 'fricative', 'place': 'labiodental', 'voiced': False},
    'g': {'manner': 'plosive', 'voiced': True},  # allophonic: [ɟ] or [ɡ]
    'ğ': {'manner': 'approximant', 'place': 'palatal', 'voiced': True},  # vowel lengthening or glide
    'h': {'ipa': 'h', 'manner': 'fricative', 'place': 'glottal', 'voiced': False},
    'j': {'ipa': 'ʒ', 'manner': 'fricative', 'place': 'palato-alveolar', 'voiced': True},
    'k': {'manner': 'plosive', 'voiced': False},  # allophonic: [c] or [k]
    'l': {'ipa': 'l', 'manner': 'liquid', 'place': 'alveolar', 'voiced': True},
    'm': {'ipa': 'm', 'manner': 'nasal', 'place': 'bilabial', 'voiced': True},
    'n': {'ipa': 'n', 'manner': 'nasal', 'place': 'alveolar', 'voiced': True},
    'p': {'ipa': 'p', 'manner': 'plosive', 'place': 'bilabial', 'voiced': False},
    'r': {'ipa': 'ɾ', 'manner': 'liquid', 'place': 'alveolar', 'voiced': True},
    's': {'ipa': 's', 'manner': 'fricative', 'place': 'alveolar', 'voiced': False},
    'ş': {'ipa': 'ʃ', 'manner': 'fricative', 'place': 'palato-alveolar', 'voiced': False},
    't': {'ipa': 't', 'manner': 'plosive', 'place': 'alveolar', 'voiced': False},
    'v': {'ipa': 'v', 'manner': 'fricative', 'place': 'labiodental', 'voiced': True},
    'y': {'ipa': 'j', 'manner': 'glide', 'place': 'palatal', 'voiced': True},
    'z': {'ipa': 'z', 'manner': 'fricative', 'place': 'alveolar', 'voiced': True},
}


class TurkishG2P:
    """Converts Turkish text into phoneme sequence with IPA representations and allophones."""

    def convert(self, normalized_text: str) -> List[PhonemeItem]:
        words = normalized_text.split()
        phonemes: List[PhonemeItem] = []

        for w_idx, word in enumerate(words):
            word_len = len(word)
            # Check vowel context in word to determine palatalization of k, g, l
            front_vowel_count = sum(1 for ch in word if ch in FRONT_VOWELS)
            back_vowel_count = sum(1 for ch in word if ch in BACK_VOWELS)
            has_front_vowels = front_vowel_count > back_vowel_count

            for c_idx, ch in enumerate(word):
                if ch in VOWEL_PROPERTIES:
                    v_props = VOWEL_PROPERTIES[ch]
                    phonemes.append(PhonemeItem(
                        grapheme=ch,
                        ipa=v_props['ipa'],
                        phoneme_key=f"/{v_props['ipa']}/",
                        word=word,
                        word_index=w_idx,
                        char_index_in_word=c_idx,
                        is_vowel=True,
                        vowel_frontness=v_props['frontness'],
                        vowel_height=v_props['height'],
                        vowel_rounded=v_props['rounded']
                    ))
                elif ch in CONSONANT_PROPERTIES:
                    c_props = CONSONANT_PROPERTIES[ch]
                    
                    # Context allophone handling for g, k, ğ
                    prev_char = word[c_idx - 1] if c_idx > 0 else ""
                    next_char = word[c_idx + 1] if c_idx < word_len - 1 else ""
                    
                    is_neighbor_front = (prev_char in FRONT_VOWELS) or (next_char in FRONT_VOWELS) or has_front_vowels

                    if ch == 'k':
                        ipa = 'c' if is_neighbor_front else 'k'
                        place = 'palatal' if is_neighbor_front else 'velar'
                    elif ch == 'g':
                        ipa = 'ɟ' if is_neighbor_front else 'ɡ'
                        place = 'palatal' if is_neighbor_front else 'velar'
                    elif ch == 'ğ':
                        # Yumuşak g in standard Istanbul Turkish:
                        # front vowel context: weak palatal glide /j/ or lengthening /ː/
                        # back vowel context: vowel lengthening /ː/
                        if is_neighbor_front:
                            ipa = 'j'
                            place = 'palatal'
                        else:
                            ipa = 'ː'
                            place = 'glottal'
                    else:
                        ipa = c_props['ipa']
                        place = c_props['place']

                    phonemes.append(PhonemeItem(
                        grapheme=ch,
                        ipa=ipa,
                        phoneme_key=f"/{ipa}/",
                        word=word,
                        word_index=w_idx,
                        char_index_in_word=c_idx,
                        is_vowel=False,
                        consonant_manner=c_props['manner'],
                        consonant_place=place,
                        consonant_voiced=c_props['voiced']
                    ))

        return phonemes
