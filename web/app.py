"""FastAPI server for Turkish Speech Evaluation Web UI."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from speech_eval.pipeline import SpeechEvaluator
from speech_eval.config import default_config

from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
evaluator: Optional[SpeechEvaluator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluator
    evaluator = SpeechEvaluator()
    yield

app = FastAPI(title="Türkçe Telaffuz ve Şive Ölçer", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index file not found")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/presets")
def get_presets():
    """Returns sample phonetically rich practice sentences."""
    return {
        "presets": [
            {
                "title": "Günlük Konuşma (Ön ve Art Ünlüler)",
                "text": "Ben bugün eve erken gidiyorum."
            },
            {
                "title": "Yumuşak G ve Uzun Ünlü Artikülasyonu",
                "text": "Yağmurlu günlerde dışarı çıkmak oldukça zorlaşıyor."
            },
            {
                "title": "Ön Yuvarlak Ünlüler (ö, ü) ve Damaksıl Ünsüzler",
                "text": "Öğleden sonra şehir kütüphanesinde buluşacağız."
            },
            {
                "title": "I/İ Ayrımı ve Düz Ünlüler",
                "text": "Ilık süt ve taze sıcak ekmek aldım."
            },
            {
                "title": "Ç/Ş Sürtünmeli ve Patlamalı Ünsüzler",
                "text": "Çocuklar parkta neşeyle koşup oyun oynuyorlardı."
            }
        ]
    }


@app.post("/api/evaluate")
async def evaluate_audio(
    file: UploadFile = File(...),
    target_text: Optional[str] = Form(None)
):
    """Processes audio and returns standard Turkish closeness scores and phoneme diagnostics."""
    global evaluator
    if evaluator is None:
        evaluator = SpeechEvaluator()

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        clean_text = target_text.strip() if target_text and target_text.strip() else None
        res = evaluator.evaluate(tmp_path, target_text=clean_text)

        # Build clean JSON response
        data = {
            "overall_score": res.overall_score,
            "subscores": {
                "pronunciation": res.pronunciation_score,
                "vowels": res.vowel_score,
                "consonants": res.consonant_score,
                "rhythm": res.rhythm_score,
                "intonation": res.intonation_score
            },
            "recognized_text": res.recognized_text,
            "expected_text": res.expected_text,
            "problematic_phonemes": res.problematic_phonemes,
            "word_scores": [
                {
                    "word": w.word,
                    "word_index": w.word_index,
                    "score": w.score,
                    "phonemes": [
                        {
                            "grapheme": p.grapheme,
                            "ipa": p.ipa,
                            "key": p.phoneme_key,
                            "score": p.total_score,
                            "embedding_score": p.embedding_score,
                            "acoustic_score": p.acoustic_score,
                            "acoustic_available": p.acoustic_available,
                            "effective_score": p.effective_score,
                            "duration_score": p.duration_score,
                            "diagnostic": p.diagnostic_note,
                            "start": p.start_time,
                            "end": p.end_time,
                            "duration": p.duration,
                            "confidence": p.alignment_confidence,
                            "is_valid": p.is_valid
                        } for p in w.phonemes
                    ]
                } for w in res.word_scores
            ]
        }
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)
