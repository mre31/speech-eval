"""End-to-end pipeline test."""

import numpy as np
from speech_eval.config import EvalConfig
from speech_eval.pipeline import SpeechEvaluator
from speech_eval.scorer import EvaluationResult


def test_evaluator_sample():
    # Use CPU in pipeline unit test to allow tests to run concurrently with the active GPU server
    config = EvalConfig(whisper_model_size="base", device="cpu")
    evaluator = SpeechEvaluator(config=config)
    res = evaluator.evaluate("/tmp/test_tts.mp3", target_text="merhaba bugün nasılsınız")

    assert isinstance(res, EvaluationResult)
    assert 0.0 <= res.overall_score <= 100.0
    assert 0.0 <= res.pronunciation_score <= 100.0
    assert 0.0 <= res.vowel_score <= 100.0
    assert 0.0 <= res.consonant_score <= 100.0
    assert 0.0 <= res.rhythm_score <= 100.0
    assert 0.0 <= res.intonation_score <= 100.0
    assert len(res.word_scores) == 3
    assert len(res.phoneme_scores) > 10
