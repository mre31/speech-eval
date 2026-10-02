# Türkçe Şive ve Telaffuz Ölçer (Speech-Eval)

Pretrained speech encoder modelleri, zorunlu hizalama (forced alignment), akustik formant analizi, çoklu konuşmacılı TTS ve istatistiksel referans dağılımları kullanarak bir konuşmanın **standart İstanbul Türkçesi telaffuzuna yakınlığını 0–100 arasında ölçen** sistemdir.

Özel bir ML modeli veya etiketli şive veri seti gerektirmez.

---

## Temel Prensip

Sistem konuşmacının **ses rengini (timbre/speaker identity)** değil, **fonemleri telaffuz etme biçimini (artikülasyon, formant oranları, süre, tonlama)** ölçer. 

- Tek bir TTS konuşmacısına benzerlik aranmaz; erkek ve kadın çoklu nötr referanslar kullanılır.
- Cinsiyet, yaş ve ses yolu uzunluğu farklılıkları formant oranları ($F_2/F_1$) ve logaritmik perde normalizasyonu ile elenir.

---

## Genel Mimari

```text
Kullanıcı Sesi (.wav / .mp3 / mikrofon)
      ↓
[Audio Preprocessing] (16 kHz mono, high-pass filter, silence trim, RMS norm)
      ↓
[Speech-to-Text] (Faster-Whisper veya kullanıcı hedef metni)
      ↓
[G2P - Grapheme-to-Phoneme] (Türkçe IPA dönüşümü ve damaksıl/art ünlü allofonları)
      ↓
[Wav2Vec2 CTC Forced Aligner] (Türkçe fonem zaman aralıkları ve güven skorları)
      ↓
┌────────────────────────────────────────────────────────┐
│ 1. Fonetik Embedding (Wav2Vec2 Layer 12 temsilleri)    │
│ 2. Akustik Analiz (Praat F1/F2/F3 formant oranları)   │
│ 3. Süre Normalizasyonu (Yerel konuşma hızı oranı)      │
│ 4. Prozodi & Tonlama (Yarıton normalize F0 perde eğrisi)│
└────────────────────────────────────────────────────────┘
      ↓
[Çoklu TTS Referansları & Mahalanobis Referans Dağılımı]
      ↓
[Robut Trimmed Weighted Aggregation]
      ↓
0–100 Standart Türkçe Yakınlık Skoru
```

---

## Kurulum

Projeyi `uv` ile kurup çalıştırabilirsiniz:

```bash
# Sanal ortam oluştur
uv venv --python 3.12 .venv
source .venv/bin/activate

# Bağımlılıkları ve paketi yükle
uv pip install -e .
```

---

## Kullanım

### 1. Web Arayüzü

Mikrofonla canlı kayıt, dosya yükleme, ses dalgası görselleştiricisi ve etkileşimli kelime/fonem analiz kartları içeren modern web arayüzünü başlatmak için:

```bash
uv run uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```

Tarayıcınızda açın: `http://localhost:8000`

### 2. Komut Satırı (CLI)

```bash
# Ses dosyasını değerlendir (varsayılan: sadece anlık çoklu TTS referansları)
uv run speech-eval evaluate --audio ornek_ses.wav

# Beklenen metinle birlikte değerlendir
uv run speech-eval evaluate --audio ornek_ses.wav --text "Ben bugün eve erken gidiyorum."

# İsteğe bağlı: Çevrimdışı istatistik veritabanını da analize dahil et
uv run speech-eval evaluate --audio ornek_ses.wav --use-db

# JSON çıktısı al
uv run speech-eval evaluate --audio ornek_ses.wav --json
```

### 3. Python API

```python
from speech_eval import SpeechEvaluator

evaluator = SpeechEvaluator()
result = evaluator.evaluate("kayit.wav", target_text="Ben bugün eve erken gidiyorum.")

print(f"Standart Türkçe Yakınlığı: {result.overall_score} / 100")
print(f"Telaffuz: {result.pronunciation_score}")
print(f"Ünlüler: {result.vowel_score}")
print(f"Ünsüzler: {result.consonant_score}")
print(f"Ritim: {result.rhythm_score}")
print(f"Tonlama: {result.intonation_score}")

for word in result.word_scores:
    print(f"Kelime: {word.word} -> {word.score}")
```

### 4. Çevrimdışı Referans Veritabanını Güncelleme

```bash
uv run speech-eval build-db
```

Bu komut fonetik olarak dengeli Türkçe cümleleri çoklu TTS konuşmacılarıyla sentezleyip `data/reference_stats.json` dosyasında monofon ve bağlama duyarlı trifon (triphone) kovaryans/ortalama dağılımlarını hesaplar.

---

## Skorlama ve Ağırlıklar

Skorlama konfigürasyonu `speech_eval/config.py` üzerinden ayarlanabilir:

| Metrik | Varsayılan Ağırlık | Açıklama |
| :--- | :---: | :--- |
| **Phonetic Embedding** | %50 | Wav2Vec2 Layer-12 temsillerinin standart referanslarla kosinüs benzerliği ve Mahalanobis mesafesi |
| **Acoustic Features** | %25 | Ünlüler için $F_2/F_1$ formant oranları; ünsüzler için spektral ağırlık merkezi ve gürültülülük |
| **Duration** | %15 | Konuşmacının ortalama konuşma hızına göre normalize edilmiş bağıl fonem süreleri |
| **Prosody & Intonation** | %10 | Medyan perdeye göre yarıton (semitone) cinsinden cümle ezgisi ve perde varyansı |

---

## Testler

Tüm testleri çalıştırmak için:

```bash
uv run pytest tests/
```
