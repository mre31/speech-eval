"""Configuration settings for Turkish Speech Evaluation system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch


@dataclass
class EvalConfig:
    # Audio settings
    sample_rate: int = 16000
    target_lufs_db: float = -20.0
    min_audio_duration_sec: float = 0.5
    silence_top_db: float = 30.0

    # Model identifiers
    wav2vec2_model_id: str = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"
    whisper_model_size: str = "large-v3"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Embedding configuration
    embedding_layer: int = 12  # Layer with high phonetic articulation, low speaker identity
    
    # Scoring weights (sum to 1.0)
    embedding_weight: float = 0.60
    acoustic_weight: float = 0.10
    duration_weight: float = 0.15
    prosody_weight: float = 0.15

    # Speaker invariance & Observed phoneme layer
    subtract_speaker_embedding_mean: bool = True
    enable_observed_phoneme_layer: bool = True

    # Filtering & Outlier thresholds
    min_phoneme_duration_sec: float = 0.025  # 25 ms
    min_alignment_confidence: float = 0.15   # Minimum CTC alignment probability
    outlier_trim_ratio: float = 0.10        # Trimmed mean ratio for robust aggregation

    # High-grade TTS Ensemble: Azure Neural (Male & Female) + Meta MMS VITS + Piper DFKI VITS
    tts_engines: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "edge", "voice": "tr-TR-AhmetNeural", "label": "Azure Ahmet (Erkek)"},
            {"type": "edge", "voice": "tr-TR-EmelNeural", "label": "Azure Emel (Kadın)"},
            {"type": "mms", "model_id": "facebook/mms-tts-tur", "label": "Meta MMS-TTS (VITS GPU)"},
            {"type": "piper", "model_path": str(Path(__file__).parent.parent / "data" / "piper_models" / "tr_TR-dfki-medium.onnx"), "label": "Piper DFKI (VITS ONNX)"},
        ]
    )

    # Reference Database mode (False = only runtime multi-TTS comparison)
    use_offline_database: bool = False
    reference_db_path: Path = Path(__file__).parent.parent / "data" / "reference_stats.json"


default_config = EvalConfig()
