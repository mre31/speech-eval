"""Script to precompute standard Turkish reference phoneme distributions.

Synthesizes phonetically balanced Turkish sentences across multiple standard TTS voices,
extracts phonetic embeddings and acoustic targets, and saves mean/covariance distributions
for Mahalanobis distance scoring.
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Ensure speech_eval is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from speech_eval.config import default_config
from speech_eval.g2p import TurkishG2P
from speech_eval.aligner import TurkishAligner
from speech_eval.embeddings import PhoneticEmbeddingExtractor
from speech_eval.tts_reference import TTSReferenceEngine
from speech_eval.distribution import PhonemeDistributionManager


# Phonetically diverse Turkish corpus covering all vowels, consonants, clusters, and allophones
TRAINING_SENTENCES = [
    "Ben bugün eve erken gidiyorum.",
    "Yağmurlu günlerde dışarı çıkmak oldukça zorlaşıyor.",
    "Türkçenin zengin ses yapısı ve fonetik uyumu inceleniyor.",
    "Öğleden sonra şehir kütüphanesinde buluşacağız.",
    "Ilık süt ve taze sıcak ekmek aldım.",
    "Çocuklar parkta neşeyle koşup oyun oynuyorlardı.",
    "Gözlüklerini masanın üzerine bırakıp odadan ayrıldı.",
    "Sağlıklı beslenmek ve düzenli yürüyüş yapmak çok önemlidir.",
    "Sabah rüzgarı deniz kıyısında serin bir hava estiriyordu.",
    "Kâğıt ve kalemini hazırlayıp dikkatle dinlemeye başladı.",
    "Geniş bahçede rengarenk güller ve ağaçlar vardı.",
    "Tren istasyonundaki kalabalık hızla dağılmaya başladı."
]


def build_database(output_path: Path):
    print("Standart Türkçe Referans Veritabanı Oluşturuluyor...")
    g2p = TurkishG2P()
    aligner = TurkishAligner(model_id=default_config.wav2vec2_model_id, device=default_config.device)
    extractor = PhoneticEmbeddingExtractor(target_layer=default_config.embedding_layer)
    tts_engine = TTSReferenceEngine(
        engines=default_config.tts_engines,
        aligner=aligner,
        extractor=extractor,
        device=default_config.device
    )
    dist_mgr = PhonemeDistributionManager()

    # Collect embeddings per phoneme key and triphone key
    mono_pools = {}
    tri_pools = {}

    for sentence in tqdm(TRAINING_SENTENCES, desc="Cümleler işleniyor"):
        norm_text = sentence.lower().strip()
        phonemes = g2p.convert(norm_text)
        if not phonemes:
            continue

        ref_data = tts_engine.generate_references(norm_text, phonemes)

        for ref_item in ref_data:
            segments = ref_item["segments"]
            embeddings = ref_item["embeddings"]

            for i, (seg, emb) in enumerate(zip(segments, embeddings)):
                curr_p = seg.phoneme.grapheme
                prev_p = segments[i - 1].phoneme.grapheme if i > 0 else "^"
                next_p = segments[i + 1].phoneme.grapheme if i < len(segments) - 1 else "$"

                # Monophone pool
                mono_key = f"_{curr_p}_"
                mono_pools.setdefault(mono_key, []).append(emb)

                # Triphone pool
                tri_key = dist_mgr.make_context_key(curr_p, prev_p, next_p)
                tri_pools.setdefault(tri_key, []).append(emb)

    print(f"Toplam {len(mono_pools)} monophon ve {len(tri_pools)} triphon havuzu toplandı.")

    # Calculate distributions
    for key, embs in mono_pools.items():
        dist_mgr.update_distribution(key, embs)

    for key, embs in tri_pools.items():
        if len(embs) >= 2:
            dist_mgr.update_distribution(key, embs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dist_mgr.save(output_path)
    print(f"Referans veritabanı başarıyla kaydedildi: {output_path}")


if __name__ == "__main__":
    out_file = default_config.reference_db_path
    build_database(out_file)
