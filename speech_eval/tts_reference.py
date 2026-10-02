"""Multi-engine, high-fidelity standard Turkish TTS reference generator.

Combines state-of-the-art neural TTS models:
1. Microsoft Azure Neural Studio Voices (AhmetNeural & EmelNeural) via edge-tts
2. Meta MMS-TTS Turkish (VITS neural architecture running on CUDA GPU)
3. Piper DFKI Turkish (high-precision local VITS ONNX neural model)

Provides pristine, natural standard Istanbul Turkish references without robotic SSML distortion.
"""

import asyncio
import concurrent.futures
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import librosa
import edge_tts

from speech_eval.preprocessing import preprocess_audio
from speech_eval.g2p import PhonemeItem
from speech_eval.aligner import TurkishAligner, AlignedSegment
from speech_eval.embeddings import PhoneticEmbeddingExtractor


def _run_coroutine_sync(coro):
    """Executes an async coroutine synchronously, handling already-running event loops safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class TTSReferenceEngine:
    """Manages high-grade multi-model Turkish TTS generation and reference extraction."""

    def __init__(
        self,
        engines: Optional[List[Dict[str, Any]]] = None,
        aligner: Optional[TurkishAligner] = None,
        extractor: Optional[PhoneticEmbeddingExtractor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.aligner = aligner
        self.extractor = extractor or PhoneticEmbeddingExtractor(target_layer=12)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

        # Default high-fidelity ensemble configuration
        piper_default_path = Path(__file__).parent.parent / "data" / "piper_models" / "tr_TR-dfki-medium.onnx"
        self.engines = engines or [
            {"type": "edge", "voice": "tr-TR-AhmetNeural", "label": "Azure Ahmet (Erkek)"},
            {"type": "edge", "voice": "tr-TR-EmelNeural", "label": "Azure Emel (Kadın)"},
            {"type": "mms", "model_id": "facebook/mms-tts-tur", "label": "Meta MMS-TTS (VITS GPU)"},
            {"type": "piper", "model_path": str(piper_default_path), "label": "Piper DFKI (VITS ONNX)"},
        ]

        # Lazy model holders
        self._mms_model = None
        self._mms_tokenizer = None
        self._piper_voice = None

    def _get_mms_model(self, model_id: str = "facebook/mms-tts-tur"):
        if self._mms_model is None:
            from transformers import VitsModel, AutoTokenizer
            try:
                self._mms_tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                self._mms_model = VitsModel.from_pretrained(model_id, local_files_only=True).to(self.device)
            except Exception:
                self._mms_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._mms_model = VitsModel.from_pretrained(model_id).to(self.device)
            self._mms_model.eval()
        return self._mms_tokenizer, self._mms_model

    def _get_piper_voice(self, model_path: str):
        if self._piper_voice is None and Path(model_path).exists():
            from piper.voice import PiperVoice
            self._piper_voice = PiperVoice.load(model_path)
        return self._piper_voice

    async def _generate_edge_tts(self, text: str, voice: str, output_path: Path) -> np.ndarray:
        """Synthesizes speech using edge-tts with native neutral rate and pitch."""
        communicate = edge_tts.Communicate(text, voice, rate="+0%", pitch="+0Hz")
        await communicate.save(str(output_path))
        return preprocess_audio(output_path, sr=16000, trim=True)

    def _generate_mms_tts(self, text: str) -> Optional[np.ndarray]:
        """Synthesizes speech using Meta MMS-TTS VITS model on GPU."""
        try:
            tokenizer, model = self._get_mms_model()
            inputs = tokenizer(text.lower(), return_tensors="pt").to(self.device)
            with torch.no_grad():
                waveform = model(**inputs).waveform[0].cpu().numpy()
            return preprocess_audio(waveform, sr=16000, trim=True)
        except Exception:
            return None

    def _generate_piper_tts(self, text: str, model_path: str) -> Optional[np.ndarray]:
        """Synthesizes speech using Piper DFKI ONNX neural model."""
        try:
            voice = self._get_piper_voice(model_path)
            if voice is None:
                return None
            chunks = list(voice.synthesize(text.lower()))
            if not chunks:
                return None
            audio_22k = np.concatenate([c.audio_float_array for c in chunks])
            audio_16k = librosa.resample(audio_22k, orig_sr=voice.config.sample_rate, target_sr=16000)
            return preprocess_audio(audio_16k, sr=16000, trim=True)
        except Exception:
            return None

    def generate_references(
        self,
        text: str,
        phonemes: List[PhonemeItem]
    ) -> List[Dict[str, Any]]:
        """Synthesizes speech using all configured high-grade models and extracts aligned segments.
        
        Args:
            text: Normalized Turkish text
            phonemes: Expected phoneme list
            
        Returns:
            List of reference speaker dictionaries with audio, segments, and phonetic embeddings.
        """
        cache_key = text.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.aligner is None:
            self.aligner = TurkishAligner()

        results: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for i, eng in enumerate(self.engines):
                eng_type = eng.get("type")
                label = eng.get("label", f"Engine {i}")
                audio = None

                # 1. Edge-TTS (Azure Neural Ahmet / Emel)
                if eng_type == "edge":
                    voice = eng.get("voice", "tr-TR-AhmetNeural")
                    out_file = tmp_path / f"edge_{i}_{voice}.mp3"
                    try:
                        audio = _run_coroutine_sync(self._generate_edge_tts(text, voice, out_file))
                    except Exception:
                        continue

                # 2. Meta MMS-TTS (VITS GPU)
                elif eng_type == "mms":
                    audio = self._generate_mms_tts(text)

                # 3. Piper DFKI (VITS ONNX)
                elif eng_type == "piper":
                    model_path = eng.get("model_path", "")
                    audio = self._generate_piper_tts(text, model_path)

                if audio is None or len(audio) < 1600:
                    continue

                segments, hidden_states = self.aligner.align(
                    audio=audio,
                    phonemes=phonemes,
                    sr=16000,
                    return_hidden_states=True
                )

                if segments and hidden_states is not None:
                    embeddings = self.extractor.extract_all(hidden_states, segments)
                    results.append({
                        "speaker": label,
                        "audio": audio,
                        "segments": segments,
                        "embeddings": embeddings
                    })

        self._cache[cache_key] = results
        return results
