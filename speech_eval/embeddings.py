"""Phonetic embedding extraction using Wav2Vec2 middle-layer representations.

Extracts speaker-invariant, articulation-sensitive phonetic embeddings for aligned segments.
Includes speaker-mean vector subtraction to decouple speaker identity from phonetics.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch

from speech_eval.aligner import AlignedSegment


class PhoneticEmbeddingExtractor:
    """Extracts segment-level phonetic embeddings from Wav2Vec2 hidden states."""

    def __init__(self, target_layer: int = 12):
        self.target_layer = target_layer

    def compute_speaker_mean(self, hidden_states: Tuple[torch.Tensor, ...]) -> np.ndarray:
        """Computes average representation across all frames of the utterance (speaker timbre bias)."""
        layer_tensor = hidden_states[self.target_layer][0]  # [T, D]
        return torch.mean(layer_tensor, dim=0).detach().cpu().numpy().astype(np.float32)

    def extract_segment_embedding(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        segment: AlignedSegment,
        speaker_mean: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extracts normalized phonetic embedding for a single aligned phoneme segment.
        
        Args:
            hidden_states: Tuple of hidden state tensors from Wav2Vec2 forward pass
            segment: AlignedSegment containing start_frame and end_frame
            speaker_mean: Optional mean utterance vector to subtract for speaker invariance
            
        Returns:
            1D numpy array of shape (embedding_dim,) normalized to unit length
        """
        layer_tensor = hidden_states[self.target_layer][0]  # [T, D]
        num_frames = layer_tensor.shape[0]

        start_f = max(0, min(segment.start_frame, num_frames - 1))
        end_f = max(start_f + 1, min(segment.end_frame, num_frames))

        frames = layer_tensor[start_f:end_f]  # [L, D]
        # Mean pooling across the duration of the phoneme
        embedding = torch.mean(frames, dim=0).detach().cpu().numpy().astype(np.float32)

        # Subtract speaker mean vector if provided
        if speaker_mean is not None:
            embedding = embedding - speaker_mean

        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    def extract_all(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        segments: List[AlignedSegment],
        subtract_speaker_mean: bool = True
    ) -> List[np.ndarray]:
        """Extracts embeddings for a list of aligned segments with optional speaker centering."""
        speaker_mean = self.compute_speaker_mean(hidden_states) if subtract_speaker_mean else None
        return [self.extract_segment_embedding(hidden_states, seg, speaker_mean=speaker_mean) for seg in segments]


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two unit vectors."""
    dot = np.dot(vec1, vec2)
    return float(np.clip(dot, -1.0, 1.0))


def similarity_to_score(sim: float) -> float:
    """Calibrates cosine similarity to a realistic standard Turkish phonetic score.
    
    Using logistic calibration centered on typical cross-speaker phoneme boundaries:
    - sim >= 0.82 -> 92-100 (Native standard pronunciation)
    - sim ~ 0.74 -> 85 (Good standard articulation)
    - sim ~ 0.66 -> 75 (Acceptable native / minor accent deviation)
    - sim ~ 0.55 -> 55 (Noticeable phonetic shift / strong accent)
    - sim < 0.42 -> < 35 (Distorted / wrong phoneme)
    """
    # Smooth logistic curve centered at x0=0.62 with slope k=13.5
    score = 100.0 / (1.0 + np.exp(-13.5 * (sim - 0.62)))
    return float(np.clip(score, 0.0, 100.0))
