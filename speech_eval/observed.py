"""Observed pronunciation layer using unconstrained CTC decoding and sequence alignment.

Detects phonological elisions (ses düşmesi / yutulması), dialectal sound substitutions,
and epenthetic insertions by aligning expected G2P phonemes against raw acoustic CTC predictions.

NOTE ON CTC TOKENS VS PHONEMES:
Standard Wav2Vec2 Turkish models are character-level CTC models, outputting orthographic
letters (a, b, c, ç...) rather than true IPA phonetic symbols. While Turkish orthography
is largely phonemic, allophonic variations ([c] vs [k], [ɟ] vs [ɡ], [j] vs [ː] for 'ğ')
and dialectal contractions map through character tokens.
The observed layer serves as an acoustic surface evidence transducer with confidence gating
and multi-frame segment verification, rather than an unconstrained phonemic decoder.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
import torch

from speech_eval.g2p import PhonemeItem


@dataclass
class ObservedToken:
    char: str
    frame_idx: int
    confidence: float
    end_frame_idx: int = 0


@dataclass
class AlignmentDiagnostic:
    status: str            # 'match', 'similar', 'substitution', 'deletion', 'uncertain'
    expected_char: str
    observed_char: Optional[str]
    score_multiplier: float  # Multiplier applied to phoneme score
    note: str
    confidence: float = 1.0


# Allophonic, phonetic similarity clusters and standard colloquial variations in Turkish
PHONETIC_SIMILAR_PAIRS: Set[Tuple[str, str]] = {
    # Palatal / Velar allophones
    ('k', 'c'), ('c', 'k'),
    ('g', 'ɟ'), ('ɟ', 'g'),
    ('k', 'g'), ('g', 'k'),
    # Voicing / Devoicing pairs
    ('p', 'b'), ('b', 'p'),
    ('t', 'd'), ('d', 't'),
    ('ç', 'c'), ('c', 'ç'),
    ('s', 'z'), ('z', 's'),
    ('ş', 'j'), ('j', 'ş'),
    ('f', 'v'), ('v', 'f'),
    # Yumuşak G glide / vowel lengthening (standard Istanbul Turkish)
    ('ğ', 'y'), ('y', 'ğ'),
    ('ğ', 'v'), ('v', 'ğ'),
    ('ğ', 'ː'), ('ː', 'ğ'),
    ('y', 'i'), ('i', 'y'),
    ('v', 'u'), ('u', 'v'),
    # Vowel raising & colloquial Turkish variation (e.g. diye, başlıyor, geliyom)
    ('e', 'i'), ('i', 'e'),
    ('a', 'ı'), ('ı', 'a'),
    ('a', 'u'), ('u', 'a'),
    ('e', 'ü'), ('ü', 'e'),
    # Vowel frontness / rounding closeness
    ('ı', 'i'), ('i', 'ı'),
    ('u', 'ü'), ('ü', 'u'),
    ('o', 'ö'), ('ö', 'o'),
    ('a', 'e'), ('e', 'a'),
    # Liquid lenition / final coda devoicing (Turkish /r/ devoicing [ɾ̥] in fast speech)
    ('r', 'ş'), ('ş', 'r'),
    ('r', 'h'), ('h', 'r'),
    ('l', 'r'), ('r', 'l'),
    ('m', 'n'), ('n', 'm'),
}


class ObservedPhonemeAnalyzer:
    """Performs unconstrained CTC decoding and aligns observed sounds with expected G2P phonemes."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def decode_logits(self, logits: torch.Tensor) -> List[ObservedToken]:
        """Performs unconstrained greedy CTC decoding with span-level confidence pooling.
        
        Args:
            logits: Tensor of shape [1, T, vocab_size]
            
        Returns:
            List of ObservedToken (consecutive identical frames pooled, blanks and '|' removed)
        """
        pred_ids = torch.argmax(logits[0], dim=-1).detach().cpu().numpy()
        probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

        observed: List[ObservedToken] = []
        T = len(pred_ids)
        t = 0

        while t < T:
            tok_id = int(pred_ids[t])
            if tok_id == 0:  # CTC blank / pad
                t += 1
                continue

            start_t = t
            while t < T and int(pred_ids[t]) == tok_id:
                t += 1
            end_t = t  # exclusive

            token_str = self.tokenizer.convert_ids_to_tokens(tok_id) if self.tokenizer else str(tok_id)
            if token_str and token_str != "|":
                span_probs = [float(probs[f, tok_id]) for f in range(start_t, end_t)]
                peak_prob = max(span_probs)
                mean_prob = float(np.mean(span_probs))
                # Span confidence: heavily weights the peak activation frame
                confidence = float(0.75 * peak_prob + 0.25 * mean_prob)
                observed.append(ObservedToken(
                    char=token_str.lower(),
                    frame_idx=start_t,
                    confidence=round(confidence, 3),
                    end_frame_idx=end_t
                ))

        return observed

    def align_sequences(
        self,
        expected_phonemes: List[PhonemeItem],
        observed_tokens: List[ObservedToken],
        segments: Optional[List[Any]] = None,
        logits: Optional[torch.Tensor] = None
    ) -> List[AlignmentDiagnostic]:
        """Aligns expected phonemes with observed CTC tokens using Needleman-Wunsch algorithm.
        
        Applies confidence gating and multi-frame acoustic evidence verification:
        - Confidence < 0.50: Insufficient evidence, no penalty applied (score_multiplier = 1.0)
        - Confidence 0.50 - 0.70: Diagnostic observation only (score_multiplier = 1.0)
        - Confidence 0.70 - 0.85: Mild penalty (score_multiplier = 0.88 - 0.95)
        - Confidence >= 0.85: Calibrated penalty (score_multiplier = 0.75 - 0.90)
        
        Args:
            expected_phonemes: Canonical phoneme sequence from G2P
            observed_tokens: Greedy decoded tokens from acoustic CTC logits
            segments: Optional aligned segments with frame boundaries for multi-frame deletion check
            logits: Optional raw CTC logits tensor [1, T, vocab_size]
            
        Returns:
            List of AlignmentDiagnostic matched 1-to-1 with expected_phonemes.
        """
        M = len(expected_phonemes)
        N = len(observed_tokens)

        # Pre-calculate probabilities if logits tensor is supplied
        probs_np = None
        if logits is not None:
            probs_np = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

        # If no observed tokens at all, check segment evidence or mark unknown
        if not observed_tokens:
            diagnostics: List[AlignmentDiagnostic] = []
            for p in expected_phonemes:
                e_char = p.grapheme.lower()
                if e_char == 'ğ':
                    diagnostics.append(AlignmentDiagnostic(
                        status="similar",
                        expected_char=e_char,
                        observed_char=None,
                        score_multiplier=1.0,
                        note="Standart uzatma / hafif kayma artikülasyonu",
                        confidence=0.95
                    ))
                else:
                    diagnostics.append(AlignmentDiagnostic(
                        status="deletion",
                        expected_char=e_char,
                        observed_char=None,
                        score_multiplier=0.75,
                        note=f"Ses yutulmuş / telaffuz edilmedi ('{e_char}' düşmesi/elision)",
                        confidence=0.85
                    ))
            return diagnostics

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
            e_char = expected_phonemes[i - 1].grapheme.lower()
            for j in range(1, N + 1):
                o_char = observed_tokens[j - 1].char.lower()

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

        # Map results back to expected_phonemes with confidence gating
        diagnostics: List[AlignmentDiagnostic] = []
        for exp_idx, obs_idx in aligned_pairs:
            if exp_idx is None:
                # Extra observed insertion (doesn't correspond to expected phoneme)
                continue

            exp_item = expected_phonemes[exp_idx]
            exp_char = exp_item.grapheme.lower()

            if obs_idx is None:
                # Expected phoneme did not produce an unconstrained greedy CTC token.
                # Check for standard Turkish phonological rules:
                if exp_char == 'ğ':
                    # Yumuşak g in standard Istanbul Turkish lengthens vowel or forms glide [j]
                    diagnostics.append(AlignmentDiagnostic(
                        status="similar",
                        expected_char=exp_char,
                        observed_char=None,
                        score_multiplier=1.0,
                        note="Standart uzatma / hafif kayma artikülasyonu",
                        confidence=0.95
                    ))
                    continue

                # Check multi-frame acoustic evidence from segment and logits
                has_acoustic_evidence = False
                del_confidence = 0.85

                if segments is not None and exp_idx < len(segments):
                    seg = segments[exp_idx]
                    seg_conf = getattr(seg, 'confidence', 0.5)
                    seg_dur = getattr(seg, 'duration', 0.05)

                    if probs_np is not None and self.tokenizer is not None:
                        tok_id = self.tokenizer.convert_tokens_to_ids(exp_char)
                        if tok_id is not None and tok_id != getattr(self.tokenizer, 'unk_token_id', -1):
                            s_frame = max(0, getattr(seg, 'start_frame', 0) - 1)
                            e_frame = min(probs_np.shape[0], getattr(seg, 'end_frame', s_frame + 1) + 2)
                            if s_frame < e_frame:
                                span_p = probs_np[s_frame:e_frame, tok_id]
                                max_p = float(np.max(span_p)) if len(span_p) > 0 else 0.0
                                
                                # If the acoustic model shows distinct presence (> 0.22) or
                                # forced alignment confidence is healthy, greedy CTC blank is not a deletion!
                                if max_p >= 0.22 or seg_conf >= 0.50 or (seg_dur >= 0.05 and seg_conf >= 0.35):
                                    has_acoustic_evidence = True
                                else:
                                    del_confidence = float(np.clip(1.0 - max(max_p, seg_conf), 0.0, 1.0))
                    else:
                        if seg_conf >= 0.55 or (seg_dur >= 0.06 and seg_conf >= 0.35):
                            has_acoustic_evidence = True

                if has_acoustic_evidence:
                    diagnostics.append(AlignmentDiagnostic(
                        status="match",
                        expected_char=exp_char,
                        observed_char=None,
                        score_multiplier=1.0,
                        note="Standart telaffuz",
                        confidence=round(1.0 - del_confidence, 2)
                    ))
                else:
                    # Gated deletion penalty based on absence evidence strength
                    if del_confidence < 0.50:
                        diagnostics.append(AlignmentDiagnostic(
                            status="uncertain",
                            expected_char=exp_char,
                            observed_char=None,
                            score_multiplier=1.0,
                            note=f"Belirsiz artikülasyon ('{exp_char}', yetersiz kanıt)",
                            confidence=round(del_confidence, 2)
                        ))
                    elif del_confidence < 0.70:
                        diagnostics.append(AlignmentDiagnostic(
                            status="deletion",
                            expected_char=exp_char,
                            observed_char=None,
                            score_multiplier=1.0,
                            note=f"Olası ses düşmesi / zayıf artikülasyon ('{exp_char}', gözlem)",
                            confidence=round(del_confidence, 2)
                        ))
                    elif del_confidence < 0.85:
                        diagnostics.append(AlignmentDiagnostic(
                            status="deletion",
                            expected_char=exp_char,
                            observed_char=None,
                            score_multiplier=0.90,
                            note=f"Ses zayıf / hafif elision ('{exp_char}')",
                            confidence=round(del_confidence, 2)
                        ))
                    else:
                        diagnostics.append(AlignmentDiagnostic(
                            status="deletion",
                            expected_char=exp_char,
                            observed_char=None,
                            score_multiplier=0.75,
                            note=f"Ses yutulmuş / telaffuz edilmedi ('{exp_char}' düşmesi/elision)",
                            confidence=round(del_confidence, 2)
                        ))
            else:
                obs_tok = observed_tokens[obs_idx]
                obs_char = obs_tok.char.lower()
                conf = obs_tok.confidence

                if exp_char == obs_char:
                    diagnostics.append(AlignmentDiagnostic(
                        status="match",
                        expected_char=exp_char,
                        observed_char=obs_char,
                        score_multiplier=1.0,
                        note="Standart telaffuz",
                        confidence=conf
                    ))
                elif (exp_char, obs_char) in PHONETIC_SIMILAR_PAIRS:
                    # Phonetic / allophonic cluster
                    if conf < 0.50:
                        diagnostics.append(AlignmentDiagnostic(
                            status="match",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=1.0,
                            note=f"Standart telaffuz (akustik varyans: '{obs_char}')",
                            confidence=conf
                        ))
                    elif conf < 0.70:
                        diagnostics.append(AlignmentDiagnostic(
                            status="similar",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=1.0,
                            note=f"Hafif ses değişimi ('{obs_char}', gözlem)",
                            confidence=conf
                        ))
                    elif conf < 0.85:
                        diagnostics.append(AlignmentDiagnostic(
                            status="similar",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=0.95,
                            note=f"Hafif ses değişimi ('{obs_char}')",
                            confidence=conf
                        ))
                    else:
                        diagnostics.append(AlignmentDiagnostic(
                            status="similar",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=0.90,
                            note=f"Hafif ses değişimi ('{obs_char}')",
                            confidence=conf
                        ))
                else:
                    # Substitution candidate with confidence gating
                    if conf < 0.50:
                        diagnostics.append(AlignmentDiagnostic(
                            status="uncertain",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=1.0,
                            note=f"Belirsiz artikülasyon ('{obs_char}', yetersiz kanıt)",
                            confidence=conf
                        ))
                    elif conf < 0.70:
                        diagnostics.append(AlignmentDiagnostic(
                            status="substitution",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=1.0,
                            note=f"Olası ses değişimi ('{obs_char}', beklenen: '{exp_char}')",
                            confidence=conf
                        ))
                    elif conf < 0.85:
                        diagnostics.append(AlignmentDiagnostic(
                            status="substitution",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=0.88,
                            note=f"'{obs_char}' olarak telaffuz edildi (beklenen: '{exp_char}')",
                            confidence=conf
                        ))
                    else:
                        diagnostics.append(AlignmentDiagnostic(
                            status="substitution",
                            expected_char=exp_char,
                            observed_char=obs_char,
                            score_multiplier=0.78,
                            note=f"'{obs_char}' olarak telaffuz edildi (beklenen: '{exp_char}')",
                            confidence=conf
                        ))

        return diagnostics

