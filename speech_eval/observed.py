"""Observed pronunciation layer using unconstrained CTC decoding and sequence alignment.

Detects phonological elisions (ses düşmesi / yutulması), dialectal sound substitutions,
and epenthetic insertions by aligning expected G2P phonemes against raw acoustic CTC predictions.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch

from speech_eval.g2p import PhonemeItem


@dataclass
class ObservedToken:
    char: str
    frame_idx: int
    confidence: float


@dataclass
class AlignmentDiagnostic:
    status: str            # 'match', 'substitution', 'deletion', 'similar'
    expected_char: str
    observed_char: Optional[str]
    score_multiplier: float  # Multiplier applied to phoneme score (1.0 = full score, 0.3 = deleted)
    note: str


# Allophonic and phonetic similarity clusters in Turkish
PHONETIC_SIMILAR_PAIRS = {
    ('k', 'c'), ('c', 'k'),
    ('g', 'ɟ'), ('ɟ', 'g'),
    ('k', 'g'), ('g', 'k'),
    ('p', 'b'), ('b', 'p'),
    ('t', 'd'), ('d', 't'),
    ('ç', 'c'), ('c', 'ç'),
    ('s', 'z'), ('z', 's'),
    ('ş', 'j'), ('j', 'ş'),
    ('ğ', 'y'), ('y', 'ğ'),
    ('ğ', 'ː'), ('ː', 'ğ'),
    ('ı', 'i'), ('i', 'ı'),
    ('u', 'ü'), ('ü', 'u'),
    ('o', 'ö'), ('ö', 'o'),
    ('e', 'i'), ('i', 'e'),
    ('a', 'e'), ('e', 'a'),
}


class ObservedPhonemeAnalyzer:
    """Performs unconstrained CTC decoding and aligns observed sounds with expected G2P phonemes."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def decode_logits(self, logits: torch.Tensor) -> List[ObservedToken]:
        """Performs unconstrained greedy CTC decoding from raw acoustic frame logits.
        
        Args:
            logits: Tensor of shape [1, T, vocab_size]
            
        Returns:
            List of ObservedToken (collapsed duplicates, blanks removed)
        """
        # Argmax along vocabulary dimension
        pred_ids = torch.argmax(logits[0], dim=-1).detach().cpu().numpy()
        probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

        observed: List[ObservedToken] = []
        prev_id = -1

        for t, tok_id in enumerate(pred_ids):
            # 0 is Wav2Vec2 CTC blank / pad
            if tok_id != 0 and tok_id != prev_id:
                token_str = self.tokenizer.convert_ids_to_tokens(int(tok_id))
                # Skip CTC word delimiters '|' in character token comparison
                if token_str and token_str != "|":
                    prob = float(probs[t, tok_id])
                    observed.append(ObservedToken(
                        char=token_str.lower(),
                        frame_idx=t,
                        confidence=prob
                    ))
            prev_id = tok_id

        return observed

    def align_sequences(
        self,
        expected_phonemes: List[PhonemeItem],
        observed_tokens: List[ObservedToken]
    ) -> List[AlignmentDiagnostic]:
        """Aligns expected phonemes with observed CTC tokens using Needleman-Wunsch algorithm.
        
        Returns:
            List of AlignmentDiagnostic matched 1-to-1 with expected_phonemes.
        """
        # If no observed tokens, all expected phonemes are marked as unobserved
        if not observed_tokens:
            return [
                AlignmentDiagnostic(
                    status="deletion",
                    expected_char=p.grapheme,
                    observed_char=None,
                    score_multiplier=0.4,
                    note="Ses algılanamadı (sessiz / eksik artikülasyon)"
                ) for p in expected_phonemes
            ]

        M = len(expected_phonemes)
        N = len(observed_tokens)

        # Scoring parameters for dynamic programming
        MATCH_SCORE = 3.0
        SIMILAR_SCORE = 1.8
        SUB_PENALTY = -2.0
        GAP_EXP = -2.5      # Expected phoneme omitted (deletion / elision)
        GAP_OBS = -1.5      # Spurious observed sound (insertion)

        def similarity_score(e_ch: str, o_ch: str) -> float:
            if e_ch == o_ch:
                return MATCH_SCORE
            if (e_ch, o_ch) in PHONETIC_SIMILAR_PAIRS:
                return SIMILAR_SCORE
            return SUB_PENALTY

        # DP score matrix: dp[i][j]
        dp = np.zeros((M + 1, N + 1), dtype=np.float32)
        traceback = np.zeros((M + 1, N + 1), dtype=np.int32)
        # 0: diag, 1: up (e deletion), 2: left (obs insertion)

        for i in range(1, M + 1):
            dp[i][0] = dp[i - 1][0] + GAP_EXP
            traceback[i][0] = 1

        for j in range(1, N + 1):
            dp[0][j] = dp[0][j - 1] + GAP_OBS
            traceback[0][j] = 2

        for i in range(1, M + 1):
            e_char = expected_phonemes[i - 1].grapheme
            for j in range(1, N + 1):
                o_char = observed_tokens[j - 1].char

                s_diag = dp[i - 1][j - 1] + similarity_score(e_char, o_char)
                s_up = dp[i - 1][j] + GAP_EXP
                s_left = dp[i][j - 1] + GAP_OBS

                best = max(s_diag, s_up, s_left)
                dp[i][j] = best

                if best == s_diag:
                    traceback[i][j] = 0
                elif best == s_up:
                    traceback[i][j] = 1
                else:
                    traceback[i][j] = 2

        # Traceback from (M, N)
        i, j = M, N
        aligned_pairs: List[Tuple[Optional[int], Optional[int]]] = []

        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i][j] == 0:
                aligned_pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or traceback[i][j] == 1):
                aligned_pairs.append((i - 1, None))
                i -= 1
            else:
                aligned_pairs.append((None, j - 1))
                j -= 1

        aligned_pairs.reverse()

        # Map results back to expected_phonemes
        diagnostics: List[AlignmentDiagnostic] = []
        for exp_idx, obs_idx in aligned_pairs:
            if exp_idx is None:
                # Extra observed insertion (doesn't correspond to expected phoneme)
                continue

            exp_item = expected_phonemes[exp_idx]
            exp_char = exp_item.grapheme

            if obs_idx is None:
                # Deletion / Elision: expected phoneme was dropped!
                # Special case: yumuşak g in standard Istanbul Turkish often lengthens vowel
                if exp_char == 'ğ':
                    diagnostics.append(AlignmentDiagnostic(
                        status="similar",
                        expected_char=exp_char,
                        observed_char=None,
                        score_multiplier=0.92,
                        note="Standart uzatma / hafif kayma artikülasyonu"
                    ))
                else:
                    diagnostics.append(AlignmentDiagnostic(
                        status="deletion",
                        expected_char=exp_char,
                        observed_char=None,
                        score_multiplier=0.45,
                        note=f"Ses yutulmuş / telaffuz edilmedi ('{exp_char}' düşmesi/elision)"
                    ))
            else:
                obs_tok = observed_tokens[obs_idx]
                obs_char = obs_tok.char

                if exp_char == obs_char:
                    diagnostics.append(AlignmentDiagnostic(
                        status="match",
                        expected_char=exp_char,
                        observed_char=obs_char,
                        score_multiplier=1.0,
                        note="Standart telaffuz"
                    ))
                elif (exp_char, obs_char) in PHONETIC_SIMILAR_PAIRS:
                    diagnostics.append(AlignmentDiagnostic(
                        status="similar",
                        expected_char=exp_char,
                        observed_char=obs_char,
                        score_multiplier=0.90,
                        note=f"Hafif ses değişimi ('{obs_char}')"
                    ))
                else:
                    diagnostics.append(AlignmentDiagnostic(
                        status="substitution",
                        expected_char=exp_char,
                        observed_char=obs_char,
                        score_multiplier=0.60,
                        note=f"'{obs_char}' olarak telaffuz edildi (beklenen: '{exp_char}')"
                    ))

        return diagnostics
