"""Multi-speaker standard Turkish TTS reference generator.

Generates multiple standard Turkish TTS reference audios for the same text
using distinct neutral voices and acoustic variations to eliminate single-speaker bias.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import edge_tts
import librosa

from speech_eval.preprocessing import preprocess_audio
from speech_eval.g2p import PhonemeItem
from speech_eval.aligner import TurkishAligner, AlignedSegment
from speech_eval.embeddings import PhoneticEmbeddingExtractor


def _run_coroutine_sync(coro):
    """Executes an async coroutine synchronously, handling already-running event loops safely."""
    import concurrent.futures
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
    """Manages multi-speaker neutral Turkish TTS generation and reference extraction."""

    def __init__(
        self,
        speakers: Optional[List[Tuple[str, str, str]]] = None,
        aligner: Optional[TurkishAligner] = None,
        extractor: Optional[PhoneticEmbeddingExtractor] = None
    ):
        # Default: 4 configurations combining male & female neural voices with subtle variations
        self.speakers = speakers or [
            ("tr-TR-AhmetNeural", "+0%", "+0Hz"),
            ("tr-TR-EmelNeural", "+0%", "+0Hz"),
            ("tr-TR-AhmetNeural", "-5%", "+2Hz"),
            ("tr-TR-EmelNeural", "+5%", "-2Hz"),
        ]
        self.aligner = aligner
        self.extractor = extractor or PhoneticEmbeddingExtractor(target_layer=12)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    async def _generate_single_tts(
        self,
        text: str,
        voice: str,
        rate: str,
        pitch: str,
        output_path: Path
    ) -> np.ndarray:
        """Synthesizes speech using edge-tts with specific rate and pitch."""
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(output_path))
        audio = preprocess_audio(output_path, sr=16000, trim=True)
        return audio

    def generate_references(
        self,
        text: str,
        phonemes: List[PhonemeItem]
    ) -> List[Dict[str, Any]]:
        """Generates audios from all configured standard TTS speakers and extracts aligned segments.
        
        Args:
            text: Normalized text to synthesize
            phonemes: Expected phoneme list
            
        Returns:
            List of dictionaries, each containing:
            - 'speaker': voice identifier
            - 'audio': 16kHz audio array
            - 'segments': List[AlignedSegment]
            - 'embeddings': List[np.ndarray]
        """
        cache_key = text.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.aligner is None:
            self.aligner = TurkishAligner()

        results: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for i, (voice, rate, pitch) in enumerate(self.speakers):
                out_file = tmp_path / f"ref_{i}_{voice}.mp3"
                try:
                    audio = _run_coroutine_sync(self._generate_single_tts(text, voice, rate, pitch, out_file))
                except Exception:
                    continue

                if len(audio) < 1600:  # < 0.1s
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
                        "speaker": f"{voice} (rate={rate}, pitch={pitch})",
                        "audio": audio,
                        "segments": segments,
                        "embeddings": embeddings
                    })

        self._cache[cache_key] = results
        return results
