"""Forced alignment module for Turkish speech using Wav2Vec2 CTC."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, AutoModelForCTC

from speech_eval.g2p import PhonemeItem


@dataclass
class AlignedSegment:
    phoneme: PhonemeItem
    start_time: float      # Start time in seconds
    end_time: float        # End time in seconds
    duration: float        # Duration in seconds
    confidence: float      # Alignment probability (0.0 to 1.0)
    start_frame: int       # Frame index in Wav2Vec2 representation
    end_frame: int         # End frame index in Wav2Vec2 representation


class TurkishAligner:
    """Performs CTC-based forced alignment on Turkish audio."""

    def __init__(self, model_id: str = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id, local_files_only=True)
            self.model = AutoModelForCTC.from_pretrained(model_id, local_files_only=True).to(self.device)
        except Exception:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def align(
        self,
        audio: np.ndarray,
        phonemes: List[PhonemeItem],
        sr: int = 16000,
        return_hidden_states: bool = True
    ) -> Tuple[List[AlignedSegment], Optional[torch.Tensor]]:
        """Aligns phonemes to audio and returns aligned segments with exact timestamps.
        
        Args:
            audio: 16kHz mono float32 numpy array
            phonemes: List of PhonemeItem from TurkishG2P
            sr: Sampling rate (must be 16000)
            return_hidden_states: Whether to return layer hidden states for embedding extraction
            
        Returns:
            Tuple of (aligned_segments, hidden_states)
        """
        if len(audio) == 0 or not phonemes:
            return [], None

        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs.input_values, output_hidden_states=return_hidden_states)
            logits = outputs.logits  # [1, T, vocab_size]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            hidden_states = outputs.hidden_states if return_hidden_states else None

        num_frames = log_probs.shape[1]
        total_duration = len(audio) / sr
        time_per_frame = total_duration / num_frames

        # Group phonemes by word to construct CTC targets with delimiter '|'
        words: List[List[PhonemeItem]] = []
        current_w_idx = -1
        current_word_phonemes = []
        for p in phonemes:
            if p.word_index != current_w_idx:
                if current_word_phonemes:
                    words.append(current_word_phonemes)
                current_word_phonemes = [p]
                current_w_idx = p.word_index
            else:
                current_word_phonemes.append(p)
        if current_word_phonemes:
            words.append(current_word_phonemes)

        # Build token list and index mapping
        target_tokens: List[int] = []
        token_to_phoneme: List[PhonemeItem] = []

        delimiter_id = self.processor.tokenizer.convert_tokens_to_ids("|")

        for w_i, w_phonemes in enumerate(words):
            if w_i > 0:
                target_tokens.append(delimiter_id)
            for p in w_phonemes:
                tok_id = self.processor.tokenizer.convert_tokens_to_ids(p.grapheme)
                if tok_id is None or tok_id == self.processor.tokenizer.unk_token_id:
                    # Fallback to nearest representation if needed
                    tok_id = self.processor.tokenizer.convert_tokens_to_ids(p.grapheme.lower())
                target_tokens.append(tok_id)
                token_to_phoneme.append(p)

        targets = torch.tensor([target_tokens], dtype=torch.int32, device=self.device)
        input_lengths = torch.tensor([num_frames], dtype=torch.int32, device=self.device)
        target_lengths = torch.tensor([len(target_tokens)], dtype=torch.int32, device=self.device)

        try:
            aligned_tokens, scores = torchaudio.functional.forced_align(
                log_probs, targets, input_lengths, target_lengths, blank=0
            )
            spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0], blank=0)
        except Exception:
            # In case forced alignment fails (e.g. audio too short or extreme mismatch)
            return [], hidden_states

        # Filter out delimiter tokens and match with phonemes
        aligned_segments: List[AlignedSegment] = []
        phoneme_idx = 0

        # We first collect raw spans that correspond to phonemes
        char_spans = []
        for span in spans:
            tok_id = span.token
            if tok_id == delimiter_id:
                continue
            if phoneme_idx < len(token_to_phoneme):
                char_spans.append((span, token_to_phoneme[phoneme_idx]))
                phoneme_idx += 1

        if not char_spans:
            return [], hidden_states

        # Expand spans to bridge blank boundaries naturally
        for i, (span, phoneme_item) in enumerate(char_spans):
            # Start frame: extend halfway back to previous span's end
            if i > 0:
                prev_end = char_spans[i - 1][0].end
                effective_start = max(0, (span.start + prev_end) // 2)
            else:
                effective_start = max(0, span.start - 2)

            # End frame: extend halfway forward to next span's start
            if i < len(char_spans) - 1:
                next_start = char_spans[i + 1][0].start
                effective_end = min(num_frames, (span.end + next_start) // 2)
            else:
                effective_end = min(num_frames, span.end + 2)

            effective_end = max(effective_end, effective_start + 1)

            start_t = effective_start * time_per_frame
            end_t = effective_end * time_per_frame
            dur = max(0.01, end_t - start_t)
            
            # span.score is log-prob; convert to probability
            prob = float(np.exp(np.clip(span.score, -10.0, 0.0)))

            aligned_segments.append(AlignedSegment(
                phoneme=phoneme_item,
                start_time=round(start_t, 4),
                end_time=round(end_t, 4),
                duration=round(dur, 4),
                confidence=round(prob, 4),
                start_frame=effective_start,
                end_frame=effective_end
            ))

        return aligned_segments, hidden_states
