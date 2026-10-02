"""End-to-end evaluation pipeline for Turkish pronunciation and accent closeness."""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import numpy as np

from speech_eval.config import EvalConfig, default_config
from speech_eval.preprocessing import preprocess_audio
from speech_eval.stt import TurkishSTT
from speech_eval.g2p import TurkishG2P
from speech_eval.aligner import TurkishAligner
from speech_eval.embeddings import PhoneticEmbeddingExtractor, compute_cosine_similarity, similarity_to_score
from speech_eval.tts_reference import TTSReferenceEngine
from speech_eval.acoustics import AcousticFeatureExtractor, compare_vowel_acoustics, compare_consonant_acoustics
from speech_eval.prosody import ProsodyDurationAnalyzer
from speech_eval.distribution import PhonemeDistributionManager
from speech_eval.scorer import SpeechScorer, EvaluationResult, PhonemeScoreDetail


class SpeechEvaluator:
    """Orchestrates the complete Turkish speech evaluation pipeline."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or default_config

        # Pipeline components (lazy initialized where appropriate)
        self.stt = TurkishSTT(model_size=self.config.whisper_model_size, device=self.config.device)
        self.g2p = TurkishG2P()
        self.aligner = TurkishAligner(model_id=self.config.wav2vec2_model_id, device=self.config.device)
        self.extractor = PhoneticEmbeddingExtractor(target_layer=self.config.embedding_layer)
        self.tts_engine = TTSReferenceEngine(
            engines=self.config.tts_engines,
            aligner=self.aligner,
            extractor=self.extractor,
            device=self.config.device
        )
        self.acoustics = AcousticFeatureExtractor(sr=self.config.sample_rate)
        self.prosody = ProsodyDurationAnalyzer(sr=self.config.sample_rate)
        self.scorer = SpeechScorer(config=self.config)

        # Statistical offline reference manager (only active if use_offline_database=True)
        if self.config.use_offline_database and self.config.reference_db_path.exists():
            self.dist_mgr = PhonemeDistributionManager(db_path=self.config.reference_db_path)
        else:
            self.dist_mgr = None

    def evaluate(
        self,
        audio_input: Union[str, Path, np.ndarray],
        target_text: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluates user audio against standard Turkish pronunciation.
        
        Args:
            audio_input: File path or raw audio waveform
            target_text: Optional expected sentence text (prompt)
            
        Returns:
            EvaluationResult dataclass with comprehensive breakdown
        """
        # 1. Preprocessing
        audio = preprocess_audio(audio_input, sr=self.config.sample_rate, trim=True)
        if len(audio) < self.config.sample_rate * self.config.min_audio_duration_sec:
            raise ValueError("Ses süresi analiz için çok kısa (en az 0.5 saniye olmalı).")

        # 2. STT Transcription
        raw_text, norm_text = self.stt.transcribe(audio, target_text=target_text)
        if not norm_text:
            raise ValueError("Kayıtta Türkçe konuşma tespit edilemedi.")

        # 3. Grapheme-to-Phoneme
        phonemes = self.g2p.convert(norm_text)
        if not phonemes:
            raise ValueError("Fonem dizisi oluşturulamadı.")

        # 4. User Forced Alignment
        user_segments, user_hidden = self.aligner.align(
            audio=audio,
            phonemes=phonemes,
            sr=self.config.sample_rate,
            return_hidden_states=True
        )

        if not user_segments or user_hidden is None:
            raise ValueError("Ses ile fonem dizisi hizalanamadı.")

        # Extract user embeddings
        user_embeddings = self.extractor.extract_all(user_hidden, user_segments)

        # 5. Standard References (Multi-TTS and/or Offline Distributions)
        ref_data = self.tts_engine.generate_references(norm_text, phonemes)

        # Collect reference segments & embeddings
        ref_segments_list = [r["segments"] for r in ref_data]

        # 6. Prosody & Duration Analysis
        duration_details = self.prosody.analyze_durations(user_segments, ref_segments_list)
        intonation_score, pitch_contour = self.prosody.analyze_intonation(audio, user_segments)

        # Extract acoustic features for user segments
        user_acoustic_feats = [
            self.acoustics.extract_features(audio, seg) for seg in user_segments
        ]

        # Extract acoustic features for reference segments
        ref_acoustic_feats_by_idx = []
        for i in range(len(user_segments)):
            idx_feats = []
            for r in ref_data:
                if i < len(r["segments"]):
                    ref_audio = r["audio"]
                    ref_seg = r["segments"][i]
                    idx_feats.append(self.acoustics.extract_features(ref_audio, ref_seg))
            ref_acoustic_feats_by_idx.append(idx_feats)

        # 7. Phoneme-level scoring
        phoneme_score_details: List[PhonemeScoreDetail] = []

        for i, (u_seg, u_emb, u_ac, dur_info) in enumerate(zip(
            user_segments, user_embeddings, user_acoustic_feats, duration_details
        )):
            # Embedding similarity against all standard reference speakers
            sims = []
            for r in ref_data:
                if i < len(r["embeddings"]):
                    ref_emb = r["embeddings"][i]
                    sim = compute_cosine_similarity(u_emb, ref_emb)
                    sims.append(sim)

            if sims:
                # Use median similarity across references to avoid single speaker bias
                med_sim = float(np.median(sims))
                emb_score = similarity_to_score(med_sim)
            else:
                emb_score = 80.0

            p_item = u_seg.phoneme

            # Check Mahalanobis distance if offline distribution is enabled
            if self.dist_mgr is not None:
                prev_p = user_segments[i - 1].phoneme.grapheme if i > 0 else "^"
                next_p = user_segments[i + 1].phoneme.grapheme if i < len(user_segments) - 1 else "$"
                dist = self.dist_mgr.get_distribution(p_item.grapheme, prev_p, next_p)

                if dist is not None:
                    maha_dist = self.dist_mgr.compute_mahalanobis_distance(u_emb, dist)
                    maha_score = self.dist_mgr.mahalanobis_to_score(maha_dist)
                    # Blend cosine similarity with Mahalanobis score
                    emb_score = 0.5 * emb_score + 0.5 * maha_score

            # Acoustic feature score & diagnostic note
            r_feats = ref_acoustic_feats_by_idx[i] if i < len(ref_acoustic_feats_by_idx) else []
            if p_item.is_vowel:
                ac_score, diag_note = compare_vowel_acoustics(u_ac, r_feats)
            else:
                ac_score, diag_note = compare_consonant_acoustics(u_ac, r_feats)

            # Duration score
            dur_score = dur_info["duration_score"]

            # Compute combined phoneme score
            p_score_detail = self.scorer.score_single_phoneme(
                phoneme=p_item,
                duration=u_seg.duration,
                confidence=u_seg.confidence,
                embedding_score=emb_score,
                acoustic_score=ac_score,
                duration_score=dur_score,
                prosody_score=intonation_score,
                diagnostic_note=diag_note
            )
            p_score_detail.start_time = u_seg.start_time
            p_score_detail.end_time = u_seg.end_time

            phoneme_score_details.append(p_score_detail)

        # 8. Robust Aggregation
        result = self.scorer.aggregate_results(
            phoneme_scores=phoneme_score_details,
            intonation_score=intonation_score,
            recognized_text=raw_text,
            expected_text=norm_text
        )

        return result
