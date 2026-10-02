"""Speech-Eval: Standart Türkçe Telaffuz ve Fonetik Doğruluk Ölçüm Sistemi."""

from speech_eval.config import EvalConfig, default_config
from speech_eval.pipeline import SpeechEvaluator
from speech_eval.scorer import EvaluationResult, PhonemeScoreDetail, WordScoreDetail

__version__ = "0.1.0"
__all__ = [
    "EvalConfig",
    "default_config",
    "SpeechEvaluator",
    "EvaluationResult",
    "PhonemeScoreDetail",
    "WordScoreDetail",
]
