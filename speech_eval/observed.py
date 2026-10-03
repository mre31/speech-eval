"""Observed pronunciation layer using unconstrained CTC decoding and sequence alignment.

Detects phonological elisions (ses düşmesi / yutulması), dialectal sound substitutions,
and epenthetic insertions by aligning expected G2P phonemes against raw acoustic CTC predictions.

ORTHOGRAPHIC CTC VOCABULARY vs IPA REPRESENTATION:
The Wav2Vec2 CTC model outputs standard Turkish orthographic characters (a, b, c, ç... z).
It does not emit IPA symbols like [c], [ɟ], or [ː].
In Turkish orthography:
- The letter 'c' is the voiced affricate /dʒ/ (e.g. 'cam', 'cep'). It is NOT palatal k!
  If CTC outputs 'c' when 'k' was expected, this is a real substitution, not an allophone.
- The letter 'ğ' is the only letter in standard Turkish orthography that consistently undergoes
  allophonic realization as a palatal glide ('y' in front vowel contexts like 'değil' -> [dejil])
  or labial glide ('v' in rounded contexts like 'öğün' -> [øvyn]) or complete vowel lengthening.
  Palatal vs velar k/g allophones ([c] vs [k]) both map to orthographic 'k' in CTC space and
  are properly evaluated in the acoustic/embedding layer.

THREE PHONETIC SOUND RELATION CLASSES (in CTC token space):
1. ALLOPHONIC_EQUIVALENT_PAIRS:
   Turkish orthographic allophones: 'ğ' ↔ 'y', 'ğ' ↔ 'v'.
   Treated as standard pronunciation (no penalty, multiplier = 1.0).
2. PHONETICALLY_CLOSE_PAIRS (Context-Aware):
   Subtle acoustic shifts and natural phonological processes (daralma, coda devoicing).
   Evaluated with context awareness: natural within standard phonological environments
   (e.g. daralma before 'y'), but treated as dialectal substitution in bare stems.
3. REAL_SUBSTITUTION:
   All other sound pairs (e.g. k ↔ c, m ↔ n, a ↔ u, e ↔ ü, r ↔ ş, l ↔ r).
   Evaluated with full substitution penalty when supported by high confidence (>0.85).

UNCONSTRAINED DELETION VERIFICATION:
To completely avoid circular dependence on forced alignment (which forces expected phonemes onto audio),
the deletion search window is bounded temporally by the preceding and succeeding MATCHED observed tokens
from the unconstrained CTC decoding. Acoustic presence within this window is evaluated using
duration-normalized multi-frame posterior statistics (peak posterior, mean posterior, and sustained rankings),
avoiding length-inflated sums or single-frame noise.
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


# 1. Allophonic equivalents strictly in Turkish orthographic CTC character space.
# In Turkish orthography, 'ğ' is realized as 'y' (değil) or 'v' (öğün).
# Note: 'k' ↔ 'c' is NOT in this set because orthographic 'c' is /dʒ/, not IPA [c]!
ALLOPHONIC_EQUIVALENT_PAIRS: Set[Tuple[str, str]] = {
    ('ğ', 'y'), ('y', 'ğ'),
    ('ğ', 'v'), ('v', 'ğ'),
}

# 2. Phonetically close pairs (subject to context validation in is_contextually_close)
PHONETICALLY_CLOSE_PAIRS: Set[Tuple[str, str]] = {
    # Standard Turkish vowel raising before y (daralma, e.g. diye, yiyen, başlıyor, geliyom)
    ('e', 'i'), ('i', 'e'),
    ('a', 'ı'), ('ı', 'a'),
    # Close vowel frontness / rounding neighbors
    ('ı', 'i'), ('i', 'ı'),
    ('u', 'ü'), ('ü', 'u'),
    ('o', 'ö'), ('ö', 'o'),
    ('a', 'e'), ('e', 'a'),
    # Natural voicing / devoicing pairs (voicing assimilation or partial devoicing in fast speech)
    ('p', 'b'), ('b', 'p'),
    ('t', 'd'), ('d', 't'),
    ('k', 'g'), ('g', 'k'),
    ('ç', 'c'), ('c', 'ç'),
    ('s', 'z'), ('z', 's'),
    ('ş', 'j'), ('j', 'ş'),
    ('f', 'v'), ('v', 'f'),
}

# Unified set for backwards compatibility
PHONETIC_SIMILAR_PAIRS: Set[Tuple[str, str]] = ALLOPHONIC_EQUIVALENT_PAIRS | PHONETICALLY_CLOSE_PAIRS


def is_contextually_close(
    exp_item: PhonemeItem,
    exp_char: str,
    obs_char: str
) -> bool:
    """Evaluates whether a phonetically close variation is natural in Turkish linguistic context.
    
    Distinguishes natural standard Turkish phonological processes from regional dialect shifts:
    - Vowel raising (daralma e->i, a->ı): Natural before glide 'y' or in '-iyor'; dialectal otherwise.
    - Consonant voicing (p->b, t->d, k->g): Initial voicing ('para'->'bara', 'taş'->'daş') is dialectal;
      coda devoicing or intervocalic lenition is natural.
    """
    pair = (exp_char, obs_char)
    if pair not in PHONETICALLY_CLOSE_PAIRS:
        return False

    word = getattr(exp_item, 'word', '').lower()
    char_idx = getattr(exp_item, 'char_index_in_word', -1)

    # 1. Vowel raising / lowering: ('e', 'i'), ('i', 'e'), ('a', 'ı'), ('ı', 'a')
    if (exp_char in ('e', 'a') and obs_char in ('i', 'ı')) or (exp_char in ('i', 'ı') and obs_char in ('e', 'a')):
        # Check if adjacent to 'y' in the same word or in progressive suffixes
        prev_ch = word[char_idx - 1] if char_idx > 0 else ""
        next_ch = word[char_idx + 1] if 0 <= char_idx < len(word) - 1 else ""
        if prev_ch == 'y' or next_ch == 'y' or "iyor" in word or "ıyor" in word:
            return True
        # Without 'y' glide context, e->i or a->ı is regional dialect shift (e.g. 'elma'->'ilma')
        return False

    # 2. Stop voicing: ('p', 'b'), ('t', 'd'), ('k', 'g'), ('ç', 'c')
    if (exp_char in ('p', 't', 'k', 'ç') and obs_char in ('b', 'd', 'g', 'c')):
        # Word-initial stop voicing ('para' -> 'bara', 'taş' -> 'daş') is a regional dialect trait
        if char_idx == 0:
            return False
        return True

    # 3. Stop devoicing: ('b', 'p'), ('d', 't'), ('g', 'k'), ('c', 'ç')
    # Coda / word-final devoicing is standard Turkish phonology (Auslautverhärtung)
    if (exp_char in ('b', 'd', 'g', 'c') and obs_char in ('p', 't', 'k', 'ç')):
        return True

    # Other close vowel neighbors ('ı', 'i'), ('u', 'ü'), ('o', 'ö'), ('s', 'z'), ('ş', 'j'), ('f', 'v')
    return True


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
        
        Applies confidence gating, context-aware phonetic categorization, and neighbor-anchored unconstrained deletion check:
        - Allophonic equivalents (ğ->y, ğ->v): multiplier = 1.0 (no penalty)
        - Contextually close pairs: mild penalty when confirmed
        - Real substitutions: full substitution penalty when confidence > 0.85
        - Deletion verification: search window bounded by neighboring observed tokens; evaluated via duration-normalized multi-frame posteriors
        
        Args:
            expected_phonemes: Canonical phoneme sequence from G2P
            observed_tokens: Greedy decoded tokens from acoustic CTC logits
            segments: Optional aligned segments (unused in window definition to maintain independence)
            logits: Optional raw CTC logits tensor [1, T, vocab_size]
            
        Returns:
            List of AlignmentDiagnostic matched 1-to-1 with expected_phonemes.
        """
        M = len(expected_phonemes)
        N = len(observed_tokens)

        # Bug fix: If observed_tokens is completely empty (e.g. CTC decoder failure or all blanks),
        # do NOT penalize the utterance by 25%. Mark layer as unavailable / uncertain with multiplier 1.0.
        if not observed_tokens:
            return [
                AlignmentDiagnostic(
                    status="uncertain",
                    expected_char=p.grapheme.lower(),
                    observed_char=None,
                    score_multiplier=1.0,
                    note="Gözlem katmanı verisi yetersiz / belirsiz",
                    confidence=0.0
                ) for p in expected_phonemes
            ]

        # Pre-calculate probabilities if logits tensor is supplied
        probs_np = None
        if logits is not None:
            probs_np = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()

        # Scoring parameters for dynamic programming
        MATCH_SCORE = 3.0
        ALLOPHONE_SCORE = 2.8
        CLOSE_SCORE = 1.8
        SUB_PENALTY = -2.0
        GAP_EXP = -2.5      # Expected phoneme omitted (deletion / elision)
        GAP_OBS = -1.5      # Spurious observed sound (insertion)

        def similarity_score(e_ch: str, o_ch: str) -> float:
            if e_ch == o_ch:
                return MATCH_SCORE
            if (e_ch, o_ch) in ALLOPHONIC_EQUIVALENT_PAIRS:
                return ALLOPHONE_SCORE
            if (e_ch, o_ch) in PHONETICALLY_CLOSE_PAIRS:
                return CLOSE_SCORE
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
        num_pairs = len(aligned_pairs)

        for pair_k, (exp_idx, obs_idx) in enumerate(aligned_pairs):
            if exp_idx is None:
                # Extra observed insertion (doesn't correspond to expected phoneme)
                continue

            exp_item = expected_phonemes[exp_idx]
            exp_char = exp_item.grapheme.lower()

            if obs_idx is None:
                # Expected phoneme did not produce an unconstrained greedy CTC token.
                # 1. Yumuşak G in standard Istanbul Turkish:
                if exp_char == 'ğ':
                    diagnostics.append(AlignmentDiagnostic(
                        status="match",
                        expected_char=exp_char,
                        observed_char=None,
                        score_multiplier=1.0,
                        note="Standart uzatma / hafif kayma artikülasyonu",
                        confidence=0.95
                    ))
                    continue

                # 2. Independent unconstrained acoustic verification:
                # Determine search window from neighboring MATCHED observed tokens
                # completely decoupling deletion detection from forced alignment segment bias.
                has_acoustic_evidence = False
                del_confidence = 0.85

                if probs_np is not None and self.tokenizer is not None:
                    T_total = probs_np.shape[0]

                    # Find closest preceding matched observed token
                    left_frame = 0
                    for prev_k in range(pair_k - 1, -1, -1):
                        _, prev_obs = aligned_pairs[prev_k]
                        if prev_obs is not None and prev_obs < len(observed_tokens):
                            prev_tok = observed_tokens[prev_obs]
                            left_frame = getattr(prev_tok, 'end_frame_idx', 0)
                            if left_frame <= prev_tok.frame_idx:
                                left_frame = prev_tok.frame_idx + 1
                            break

                    # Find closest succeeding matched observed token
                    right_frame = T_total
                    for next_k in range(pair_k + 1, num_pairs):
                        _, next_obs = aligned_pairs[next_k]
                        if next_obs is not None and next_obs < len(observed_tokens):
                            next_tok = observed_tokens[next_obs]
                            right_frame = next_tok.frame_idx
                            break

                    # Bound unconstrained search window with a small transition margin
                    s_frame = max(0, left_frame - 1)
                    e_frame = min(T_total, right_frame + 1)
                    if s_frame >= e_frame:
                        s_frame = max(0, min(left_frame, T_total - 2))
                        e_frame = min(T_total, s_frame + 2)

                    tok_id = self.tokenizer.convert_tokens_to_ids(exp_char)
                    if tok_id is not None and tok_id != getattr(self.tokenizer, 'unk_token_id', -1) and s_frame < e_frame:
                        span_p = probs_np[s_frame:e_frame, tok_id]
                        W = len(span_p)
                        peak_p = float(np.max(span_p)) if W > 0 else 0.0
                        mean_p = float(np.mean(span_p)) if W > 0 else 0.0
                        frames_above_18 = int(np.sum(span_p >= 0.18))

                        # Evaluate frame rankings and competing dominant tokens
                        expected_is_top_nb_count = 0
                        competing_dominant_frames = 0

                        for f in range(s_frame, e_frame):
                            f_probs = probs_np[f]
                            top1_id = int(np.argmax(f_probs))

                            if top1_id == tok_id:
                                expected_is_top_nb_count += 1
                            elif top1_id == 0:
                                # CTC blank is top1; inspect non-blank candidate
                                f_nb = f_probs.copy()
                                f_nb[0] = -1.0
                                top_nb_id = int(np.argmax(f_nb))
                                if top_nb_id == tok_id and f_nb[top_nb_id] >= 0.20:
                                    expected_is_top_nb_count += 1
                            else:
                                if f_probs[top1_id] >= 0.70:
                                    competing_dominant_frames += 1

                        # Duration-normalized multi-frame evidence criteria:
                        # Rejects length-inflated sums and isolated single-frame noise.
                        if (
                            (peak_p >= 0.35) or
                            (mean_p >= 0.15 and frames_above_18 >= 2) or
                            (expected_is_top_nb_count >= 2) or
                            (W <= 2 and peak_p >= 0.28 and expected_is_top_nb_count >= 1)
                        ):
                            has_acoustic_evidence = True
                        else:
                            del_confidence = float(np.clip(1.0 - peak_p, 0.50, 1.0))
                            if competing_dominant_frames >= 2:
                                del_confidence = max(del_confidence, 0.90)
                else:
                    del_confidence = 0.85

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
                    # Gated deletion penalty based on independent absence evidence strength
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

                # 1. Exact match
                if exp_char == obs_char:
                    diagnostics.append(AlignmentDiagnostic(
                        status="match",
                        expected_char=exp_char,
                        observed_char=obs_char,
                        score_multiplier=1.0,
                        note="Standart telaffuz",
                        confidence=conf
                    ))
                # 2. Allophonic equivalent strictly in CTC orthographic space (e.g. ğ->y, ğ->v)
                elif (exp_char, obs_char) in ALLOPHONIC_EQUIVALENT_PAIRS:
                    diagnostics.append(AlignmentDiagnostic(
                        status="match",
                        expected_char=exp_char,
                        observed_char=obs_char,
                        score_multiplier=1.0,
                        note="Standart allofonik varyant",
                        confidence=conf
                    ))
                # 3. Contextually close pair (checked against Turkish linguistic context)
                elif is_contextually_close(exp_item, exp_char, obs_char):
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
                # 4. Real substitution (e.g. k ↔ c, m ↔ n, a ↔ u, e ↔ ü, r ↔ ş, l ↔ r, non-contextual shifts)
                else:
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
                            score_multiplier=0.75,
                            note=f"'{obs_char}' olarak telaffuz edildi (beklenen: '{exp_char}')",
                            confidence=conf
                        ))

        return diagnostics


