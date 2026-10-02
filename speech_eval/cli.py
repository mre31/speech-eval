"""Command-line interface for speech-eval."""

import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from speech_eval.pipeline import SpeechEvaluator
from speech_eval.config import default_config


def main():
    parser = argparse.ArgumentParser(
        description="Standart Türkçe Telaffuz ve Fonetik Doğruluk Ölçer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Ses kaydını değerlendir")
    eval_parser.add_argument("--audio", "-a", type=str, required=True, help="Ses dosyası yolu (.wav, .mp3, vb.)")
    eval_parser.add_argument("--text", "-t", type=str, default=None, help="Beklenen metin (isteğe bağlı)")
    eval_parser.add_argument("--json", "-j", action="store_true", help="Sonucu JSON formatında yazdır")
    eval_parser.add_argument("--output", "-o", type=str, default=None, help="JSON çıktı dosya yolu")
    eval_parser.add_argument("--use-db", action="store_true", help="Çevrimdışı istatistik veritabanını da analize dahil et (varsayılan: sadece çoklu TTS)")

    # Build DB command
    db_parser = subparsers.add_parser("build-db", help="Çevrimdışı referans istatistik veritabanını oluştur")
    db_parser.add_argument("--output", "-o", type=str, default=None, help="Çıktı veritabanı JSON yolu")

    args = parser.parse_args()

    if args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "build-db":
        from scripts.build_reference_db import build_database
        out_path = Path(args.output) if args.output else default_config.reference_db_path
        build_database(out_path)
    else:
        parser.print_help()


def run_evaluation(args):
    console = Console()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        console.print(f"[red]Hata: Ses dosyası bulunamadı: {audio_path}[/red]")
        sys.exit(1)

    if not args.json:
        console.print(Panel(f"[bold cyan]Türkçe Şive / Telaffuz Analizi Başlatılıyor[/bold cyan]\nSes Dosyası: {audio_path.name}"))

    from speech_eval.config import EvalConfig
    cfg = EvalConfig(use_offline_database=getattr(args, "use_db", False))
    evaluator = SpeechEvaluator(config=cfg)
    try:
        result = evaluator.evaluate(audio_path, target_text=args.text)
    except Exception as e:
        console.print(f"[red]Analiz sırasında hata oluştu:[/red] {e}")
        sys.exit(1)

    # JSON output mode
    if args.json or args.output:
        data = {
            "overall_score": result.overall_score,
            "subscores": {
                "pronunciation": result.pronunciation_score,
                "vowels": result.vowel_score,
                "consonants": result.consonant_score,
                "rhythm": result.rhythm_score,
                "intonation": result.intonation_score
            },
            "recognized_text": result.recognized_text,
            "expected_text": result.expected_text,
            "problematic_phonemes": result.problematic_phonemes,
            "word_scores": [
                {"word": w.word, "score": w.score} for w in result.word_scores
            ]
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Sonuçlar kaydedildi: {args.output}[/green]")
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return

    # Rich Terminal Output
    # Overall Score Panel
    score_color = "green" if result.overall_score >= 80 else ("yellow" if result.overall_score >= 60 else "red")
    console.print(f"\n[bold]Standart Türkçe Yakınlığı:[/bold] [{score_color} bold text-xl]{result.overall_score} / 100[/{score_color} bold text-xl]")

    # Subscores Table
    sub_table = Table(title="Kategori Skorları", show_header=True, header_style="bold magenta")
    sub_table.add_column("Metrik", style="dim")
    sub_table.add_column("Puan", justify="right")
    sub_table.add_column("Açıklama")

    sub_table.add_row("Telaffuz", f"{result.pronunciation_score}", "Fonetik artikülasyon ve ses benzerliği")
    sub_table.add_row("Ünlüler", f"{result.vowel_score}", "Formant frekans oranları ve ağız açıklığı")
    sub_table.add_row("Ünsüzler", f"{result.consonant_score}", "Sürtünme, patlama ve boğumlanma yeri")
    sub_table.add_row("Ritim", f"{result.rhythm_score}", "Konuşma hızına göre normalize edilmiş süreler")
    sub_table.add_row("Tonlama", f"{result.intonation_score}", "Cümle ezgisi ve F0 perde eğrisi")
    console.print(sub_table)

    # Word Scores Table
    w_table = Table(title="Kelime Bazlı Telaffuz", show_header=True, header_style="bold cyan")
    w_table.add_column("Kelime")
    w_table.add_column("Skor", justify="right")
    w_table.add_column("Durum")

    for w in result.word_scores:
        status_color = "green" if w.score >= 80 else ("yellow" if w.score >= 65 else "red")
        status_text = "Çok İyi" if w.score >= 85 else ("İyi" if w.score >= 75 else "Geliştirilebilir")
        w_table.add_row(w.word, f"[{status_color}]{w.score}[/{status_color}]", status_text)
    console.print(w_table)

    # Problematic Phonemes
    if result.problematic_phonemes:
        p_table = Table(title="Dikkat Çeken Fonem Sapmaları", show_header=True, header_style="bold red")
        p_table.add_column("Fonem")
        p_table.add_column("Skor", justify="right")
        p_table.add_column("Tekrar", justify="center")
        p_table.add_column("Teşhis")

        for p in result.problematic_phonemes:
            p_table.add_row(p["phoneme"], f"{p['score']}", str(p["count"]), p["diagnostic"])
        console.print(p_table)
    else:
        console.print("[green]✓ Belirgin bir fonetik sapma tespit edilmedi. Telaffuz standart Türkçeye oldukça yakın.[/green]")


if __name__ == "__main__":
    main()
