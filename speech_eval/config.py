"""Configuration settings for Turkish Speech Evaluation system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
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
    whisper_model_size: str = "base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Embedding configuration
    embedding_layer: int = 12  # Layer with high phonetic articulation, low speaker identity
    
    # Scoring weights (sum to 1.0)
    embedding_weight: float = 0.50
    acoustic_weight: float = 0.25
    duration_weight: float = 0.15
    prosody_weight: float = 0.10

    # Filtering & Outlier thresholds
    min_phoneme_duration_sec: float = 0.025  # 25 ms
    min_alignment_confidence: float = 0.15   # Minimum CTC alignment probability
    outlier_trim_ratio: float = 0.10        # Trimmed mean ratio for robust aggregation

    # TTS Reference configuration
    # Multiple distinct neutral Istanbul Turkish voices + slight rate/pitch variations
    tts_speakers: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("tr-TR-AhmetNeural", "+0%", "+0Hz"),
            ("tr-TR-EmelNeural", "+0%", "+0Hz"),
            ("tr-TR-AhmetNeural", "-5%", "+2Hz"),
            ("tr-TR-EmelNeural", "+5%", "-2Hz"),
        ]
    )

    # Reference Database mode (False = only runtime multi-TTS comparison)
    use_offline_database: bool = False
    reference_db_path: Path = Path(__file__).parent.parent / "data" / "reference_stats.json"


default_config = EvalConfig()
