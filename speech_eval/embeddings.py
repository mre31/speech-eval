"""Phonetic embedding extraction using Wav2Vec2 middle-layer representations.

Extracts speaker-invariant, articulation-sensitive phonetic embeddings for aligned segments.
"""

from typing import List, Tuple
import numpy as np
import torch

from speech_eval.aligner import AlignedSegment


class PhoneticEmbeddingExtractor:
    """Extracts segment-level phonetic embeddings from Wav2Vec2 hidden states."""

    def __init__(self, target_layer: int = 12):
        self.target_layer = target_layer

    def extract_segment_embedding(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        segment: AlignedSegment
    ) -> np.ndarray:
        """Extracts normalized phonetic embedding for a single aligned phoneme segment.
        
        Args:
            hidden_states: Tuple of hidden state tensors from Wav2Vec2 forward pass
            segment: AlignedSegment containing start_frame and end_frame
            
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

        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    def extract_all(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        segments: List[AlignedSegment]
    ) -> List[np.ndarray]:
        """Extracts embeddings for a list of aligned segments."""
        return [self.extract_segment_embedding(hidden_states, seg) for seg in segments]


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two unit vectors."""
    dot = np.dot(vec1, vec2)
    return float(np.clip(dot, -1.0, 1.0))


def similarity_to_score(sim: float, min_val: float = 0.50, max_val: float = 0.95) -> float:
    """Calibrates cosine similarity to a standard 0-100 scale."""
    clamped = np.clip((sim - min_val) / (max_val - min_val), 0.0, 1.0)
    return float(clamped * 100.0)
