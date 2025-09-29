import argparse
import gc
import logging
import os
import re
import subprocess
import time
import warnings
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pywintypes
import soundfile as sf
import torch
import torchaudio
import win32file
import win32pipe
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from xcodec2.modeling_xcodec2 import XCodec2Model

# 特定の警告をピンポイントで抑制
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
# Whisperの入力パラメータ名変更に関する警告を抑制
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated", category=FutureWarning)

# PyTorchの分散処理警告を抑制（ログシステム経由）
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# transformersの警告を抑制
logging.getLogger("transformers").setLevel(logging.ERROR)


class Llasa:
    """
    Llasa voice generation tool / Llasa音声生成ツール

    Performance optimizations / パフォーマンス最適化:
    - Selectable Whisper models (anime-whisper/whisper-large-v3-turbo) / Whisperモデル選択可能（anime-whisper/whisper-large-v3-turbo）
    - SageAttention is automatically detected and enabled if available / SageAttentionは利用可能な場合自動検出・有効化
    - Use --quantization to enable 4-bit quantization for VRAM efficiency / --quantizationで4bit量子化を有効化しVRAM効率化
    """

    def __init__(self):
        self.model = None
        self.batch_count = 0
        self.quantization_enabled = False  # デフォルトで量子化無効
        self.seed = None  # テスト用ランダムシード（再現性確保のため）
        self.paths = []
        self.play_audio = False  # 音声再生フラグ
        self.volume = 1.0  # 音声再生音量
        self.speed = 1.0  # 音声再生速度
        self.ffplay_process = None  # 前回の再生プロセス

        # モデル関連の属性
        self.llasa_model = None
        self.tokenizer = None
        self.codec_model = None
        self.whisper_model = None
        self.whisper_model_name = "litagin/anime-whisper"  # デフォルト
        self.whisper_generate_kwargs = None

        # 時間計測用の属性
        self.generation_start_time = None
        self.generation_times = []
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # より安定した量子化タイプ
        )

        # テキスト正規化用のマッピング（Anime-Llasa-3B-Demoと同じ）
        self.REPLACE_MAP = {
            r"\t": "",
            r"\[n\]": "",
            r" ": "",
            r"　": "",
            r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
            r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
            r"[\uff5e\u301C]": "ー",
            r"？": "?",
            r"！": "!",
            r"[●◯〇]": "○",
            r"♥": "♡",
        }

        # 文字種変換用のマッピング
        self.FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
            {
                chr(full): chr(half)
                for full, half in zip(
                    list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
                    list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
                )
            }
        )

        self.HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
            {chr(half): chr(full) for half, full in zip(range(0xFF61, 0xFF9F), range(0x30A1, 0x30FB))}
        )

        self.FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
            {chr(full): chr(half) for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))}
        )

        # 無効文字パターン（Anime-Llasa-3B-Demoと同じ）
        self.INVALID_PATTERN = re.compile(
            r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
            r"\u0041-\u005A\u0061-\u007A"
            r"\u0030-\u0039"
            r"。、!?…♪♡○]"
        )

    def parse_arguments(self):
        """
        引数を解析する。
        -m --model で "NandemoGHS/Anime-Llasa-3B" といったモデル名を引数から指定する
        -b --batch_count バッチ生成回数を指定する。デフォルトは0でバッチ生成回数をプロンプトで確認する。-1なら無制限に生成
        残りの引数はパスのリストとして扱う
        """
        parser = argparse.ArgumentParser(description="Llasa voice generation tool / Llasa音声生成ツール")

        # === 基本設定 (Core Settings) ===
        # モデル指定
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="NandemoGHS/Anime-Llasa-3B",
            help="Model name to use (e.g., NandemoGHS/Anime-Llasa-3B) / 使用するモデル名 (例: NandemoGHS/Anime-Llasa-3B)",
        )

        # Whisperモデル選択オプション
        parser.add_argument(
            "-w",
            "--whisper",
            type=str,
            default="litagin/anime-whisper",
            help="Whisper model ID to use for speech recognition (e.g., litagin/anime-whisper, openai/whisper-large-v3-turbo) (default: litagin/anime-whisper) / 音声認識に使用するWhisperモデルID（例: litagin/anime-whisper, openai/whisper-large-v3-turbo）（デフォルト: litagin/anime-whisper）",
        )

        # パスのリスト（処理対象ファイル）
        parser.add_argument(
            "paths",
            nargs="*",
            help="List of sound file paths to process (text is loaded from specified text file) / 処理する音声ファイルパスのリスト（テキストは指定されたテキストファイルから読み込み）",
        )

        # テキストファイル指定オプション
        parser.add_argument(
            "-t",
            "--text",
            type=str,
            default="Dialogue.txt",
            help="Text file path to load dialogue lines from (default: Dialogue.txt) / 台詞テキストを読み込むファイルパス（デフォルト: Dialogue.txt）",
        )

        # 出力フォルダ指定オプション
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default="Output",
            help="Output directory for generated audio files (default: Output) / 生成された音声ファイルの出力ディレクトリ（デフォルト: Output）",
        )

        # === 動作モード (Operation Mode) ===
        # 継続モード（サーバーモード）- 名前付きパイプによるプロセス間通信
        parser.add_argument(
            "--server",
            action="store_true",
            default=False,
            help="Run in server mode using named pipe for inter-process communication / 名前付きパイプによるプロセス間通信でサーバーモードを実行",
        )

        # バッチ生成回数指定
        parser.add_argument(
            "-b",
            "--batch_count",
            type=int,
            default=0,
            help="Batch generation count (0: prompt for confirmation, -1: unlimited generation, other: specified number) / バッチ生成回数 (0: プロンプトで確認, -1: 無制限生成, その他: 指定数)",
        )

        # === パフォーマンス・最適化 (Performance Optimization) ===
        # 量子化オプション
        parser.add_argument(
            "-q",
            "--quantization",
            action="store_true",
            default=False,
            help="Enable 4bit quantization for VRAM efficiency (may slightly reduce quality) / VRAM効率化のため4bit量子化を有効化（品質がわずかに低下する可能性）",
        )

        # === 生成パラメータ (Generation Parameters) ===
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.8,
            help="Temperature for text generation (default: 0.8) / テキスト生成の温度パラメータ（デフォルト: 0.8）",
        )

        parser.add_argument(
            "--top_p",
            type=float,
            default=1.0,
            help="Top-p value for nucleus sampling (default: 1.0) / ニューリアスサンプリングのtop-p値（デフォルト: 1.0）",
        )

        parser.add_argument(
            "--repetition_penalty",
            type=float,
            default=1.1,
            help="Repetition penalty for text generation (default: 1.1) / テキスト生成の繰り返しペナルティ（デフォルト: 1.1）",
        )

        # === 出力・再生 (Output & Playback) ===
        # 音声再生オプション
        parser.add_argument(
            "-p",
            "--play",
            action="store_true",
            default=False,
            help="Play generated audio using ffplay.exe (skips if previous playback is still running) / 生成された音声をffplay.exeで再生（前回の再生が実行中の場合はスキップ）",
        )

        # 音量設定オプション
        parser.add_argument(
            "-v",
            "--volume",
            type=float,
            default=1.0,
            help="Volume level for audio playbook (0.0-2.0, default: 1.0) / 音声再生の音量レベル（0.0-2.0、デフォルト: 1.0）",
        )

        # 再生速度設定オプション
        parser.add_argument(
            "-s",
            "--speed",
            type=float,
            default=1.0,
            help="Playback speed for audio playbook (0.5-2.0, default: 1.0) / 音声再生の再生速度（0.5-2.0、デフォルト: 1.0）",
        )

        # === テスト・デバッグ (Testing & Debug) ===
        # テスト用シード設定オプション
        parser.add_argument(
            "--test-seed",
            type=int,
            default=None,
            help="Random seed for reproducible test results (for testing purposes only, default: None - random seed) / テスト用の再現可能な結果のためのランダムシード（テスト目的のみ、デフォルト: None - ランダムシード）",
        )

        args = parser.parse_args()

        # クラス属性に設定
        self.model = args.model
        self.batch_count = args.batch_count
        # 量子化設定: デフォルトはFalse、--quantizationが指定されていればTrue
        self.quantization_enabled = args.quantization
        # 音声再生設定
        self.play_audio = args.play
        # 音量設定（0.0-2.0の範囲でチェック）
        self.volume = max(0.0, min(2.0, args.volume))
        if args.volume != self.volume:
            print(
                f"Warning: Volume clamped to valid range: {args.volume} -> {self.volume} / 警告: 音量を有効範囲に調整: {args.volume} -> {self.volume}"
            )
        # 再生速度設定（0.5-2.0の範囲でチェック）
        self.speed = max(0.5, min(2.0, args.speed))
        if args.speed != self.speed:
            print(
                f"Warning: Speed clamped to valid range: {args.speed} -> {self.speed} / 警告: 再生速度を有効範囲に調整: {args.speed} -> {self.speed}"
            )
        # デーモンモード設定（名前付きパイプ使用）
        self.server_mode = args.server
        # テキストファイルパス設定
        self.text_file_path = args.text
        # 出力ディレクトリ設定
        self.output_dir = args.output
        # 生成パラメータ設定
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        # テスト用シード設定
        self.seed = args.test_seed
        # Whisperモデル設定
        self.whisper_model_name = args.whisper
        self.paths = args.paths

        # サーバーモードの場合は音声ファイル検索をスキップ（リクエスト時に処理）
        if self.server_mode:
            self.sound_files = []
            return args

        # パスのリストからsound_filesを用意する。
        # フォルダなら再帰的に探索する。
        # sound_files は import soundfile が読み込み可能な拡張子。

        self.sound_files = []

        # soundfileが対応する音声ファイル拡張子
        audio_extensions = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".au", ".caf"}

        for path in self.paths:
            # パスを正規化（相対パスを絶対パスに変換、バックスラッシュを統一）
            normalized_path = os.path.abspath(os.path.normpath(path))

            # 正規化されたパスの存在確認
            if not os.path.exists(normalized_path):
                # 元のパスでも試してみる
                if os.path.exists(path):
                    normalized_path = path
                else:
                    print(
                        f"Warning: Path does not exist: {normalized_path} / 警告: パスが存在しません: {normalized_path}"
                    )
                    continue

            # 正規化されたパスを使用
            path = normalized_path

            if os.path.isdir(path):
                # フォルダなら再帰的に探索
                for root, dirs, files in os.walk(path):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        file_ext = os.path.splitext(filename)[1].lower()

                        if file_ext in audio_extensions:
                            self.sound_files.append(file_path)
            else:
                # ファイルなら拡張子で判別
                file_ext = os.path.splitext(path)[1].lower()

                if file_ext in audio_extensions:
                    self.sound_files.append(path)

        # ファイル検索結果を表示
        print(f"Found {len(self.sound_files)} sound files. / 音声ファイル{len(self.sound_files)}個を発見しました。")

        # sound_files が空ならエラーで終了（デーモンモードの場合は除く）
        if not self.sound_files and not self.server_mode:
            raise ValueError(
                "Error: No valid sound files found. Please provide sound files in supported formats (.wav, .flac, .ogg, .mp3, .aiff, .au, .caf). / "
                "エラー: 有効な音声ファイルが見つかりません。対応形式（.wav, .flac, .ogg, .mp3, .aiff, .au, .caf）の音声ファイルを提供してください。"
            )

        return args

    def _collect_audio_data(self):
        """サウンドファイルを収集し、有効なオーディオデータを確認する（Whisper転写ベース）"""
        for sound_file in self.sound_files:
            # 音声ファイルが実際に読み込み可能かチェック
            try:
                # soundfileで音声ファイルを読み込み
                waveform, sample_rate = sf.read(sound_file)

                if len(waveform) == 0:
                    print(f"Warning: Empty audio file: {sound_file} / 警告: 空の音声ファイル: {sound_file}")
                    continue

                # 音声長をチェック（15秒制限）
                duration = len(waveform) / sample_rate
                if duration > 15.0:
                    print(
                        f"Warning: Audio too long ({duration:.2f}s), trimming to 15s: {sound_file} / 警告: 音声が長すぎます（{duration:.2f}秒）、15秒に切り詰めます: {sound_file}"
                    )
                    waveform = waveform[: int(sample_rate * 15)]
                    duration = 15.0

                # ステレオをモノラルに変換
                if len(waveform.shape) > 1 and waveform.shape[1] > 1:
                    waveform = np.mean(waveform, axis=1)
                    print(
                        f"Info: Converted stereo to mono: {sound_file} / 情報: ステレオをモノラルに変換: {sound_file}"
                    )

                # 16kHzにリサンプリング（必要な場合）
                target_rate = 16000
                if sample_rate != target_rate:
                    # 簡易リサンプリング（実際の使用時にはtorchのresampleを使用する）
                    print(
                        f"Info: Audio needs resampling from {sample_rate}Hz to {target_rate}Hz: {sound_file} / 情報: {sample_rate}Hzから{target_rate}Hzへのリサンプリングが必要: {sound_file}"
                    )

                self.audio_data.append(
                    {
                        "file": sound_file,
                        "duration": duration,
                        "samplerate": sample_rate,
                        "channels": 1 if len(waveform.shape) == 1 else waveform.shape[1],
                        "needs_resampling": sample_rate != target_rate,
                        "waveform_shape": waveform.shape,
                    }
                )

            except Exception as e:
                print(
                    f"Warning: Could not read audio file {sound_file}: {e} / 警告: 音声ファイルを読み込めません {sound_file}: {e}"
                )
                continue

        print(
            f"Collected {len(self.audio_data)} valid audio files. / {len(self.audio_data)}個の有効な音声ファイルを収集しました。"
        )

        if not self.audio_data and not self.server_mode:
            raise ValueError(
                "Error: No valid audio files found. Please ensure audio files are in supported formats and accessible. / "
                "エラー: 有効な音声ファイルが見つかりません。音声ファイルがサポートされている形式でアクセス可能であることを確認してください。"
            )

    def _collect_audio_files_from_path(self, input_path):
        """入力パスから音声ファイルを収集する"""
        audio_extensions = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".au", ".caf"}
        audio_files = []

        if os.path.isfile(input_path):
            # 単一ファイルの場合
            if any(input_path.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(input_path)
        elif os.path.isdir(input_path):
            # ディレクトリの場合、再帰的に音声ファイルを検索
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(root, file))

        return sorted(audio_files)

    def _load_dialogue_txt(self):
        """指定されたテキストファイルから有効な行を収集する（リードオンリーで再読み込み）"""
        dialogue_file = self.text_file_path

        # 既存のtext_linesをクリア
        self.text_lines = []

        print(f"Loading text lines from {dialogue_file}... / {dialogue_file}からテキスト行を読み込み中...")

        if not os.path.exists(dialogue_file):
            raise ValueError(
                f"Error: {dialogue_file} not found. Please create the text file with text lines to generate. / "
                f"エラー: {dialogue_file}が見つかりません。生成するテキスト行を含むテキストファイルを作成してください。"
            )

        try:
            with open(dialogue_file, "r", encoding="utf-8-sig") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # 空行をスキップ
                    if not line:
                        continue

                    # コメント行をスキップ（#で始まる行）
                    if line.startswith("#"):
                        continue

                    # 読み上げ不可能な文字のチェック（制御文字など）
                    if self._is_readable_line(line):
                        # テキストを正規化
                        normalized_text = self.normalize_text(line)

                        # 正規化後も有効な文字が残っているかチェック
                        if normalized_text and self._is_readable_line(normalized_text):
                            # テキスト長制限（300文字）
                            if len(normalized_text) > 300:
                                print(
                                    f"Warning: Text too long ({len(normalized_text)} chars), truncating to 300 chars: {dialogue_file}:{line_num} / 警告: テキストが長すぎます（{len(normalized_text)}文字）、300文字に切り詰めます: {dialogue_file}:{line_num}"
                                )
                                normalized_text = normalized_text[:300]

                            self.text_lines.append(
                                {
                                    "text": normalized_text,
                                    "original_text": line,
                                    "file": dialogue_file,
                                    "line_number": line_num,
                                }
                            )

        except UnicodeDecodeError:
            print(
                f"Warning: Failed to decode file {dialogue_file}, trying with different encoding / 警告: ファイル{dialogue_file}のデコードに失敗、別のエンコーディングを試行"
            )
            try:
                with open(dialogue_file, "r", encoding="shift_jis") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith("#") and self._is_readable_line(line):
                            # テキストを正規化
                            normalized_text = self.normalize_text(line)

                            # 正規化後も有効な文字が残っているかチェック
                            if normalized_text and self._is_readable_line(normalized_text):
                                # テキスト長制限（300文字）
                                if len(normalized_text) > 300:
                                    print(
                                        f"Warning: Text too long ({len(normalized_text)} chars), truncating to 300 chars: {dialogue_file}:{line_num} / 警告: テキストが長すぎます（{len(normalized_text)}文字）、300文字に切り詰めます: {dialogue_file}:{line_num}"
                                    )
                                    normalized_text = normalized_text[:300]

                                self.text_lines.append(
                                    {
                                        "text": normalized_text,
                                        "original_text": line,
                                        "file": dialogue_file,
                                        "line_number": line_num,
                                    }
                                )
            except Exception as e:
                print(
                    f"Error: Could not read file {dialogue_file}: {e} / エラー: ファイル{dialogue_file}を読み込めません: {e}"
                )
                raise

        except Exception as e:
            print(
                f"Error: Could not read file {dialogue_file}: {e} / エラー: ファイル{dialogue_file}を読み込めません: {e}"
            )
            raise

        print(
            f"Collected {len(self.text_lines)} valid text lines from {dialogue_file}. / {dialogue_file}から{len(self.text_lines)}行の有効なテキストを収集しました。"
        )

        if not self.text_lines:
            raise ValueError(
                f"Error: No valid text lines found in {dialogue_file}. / "
                f"エラー: {dialogue_file}に有効なテキスト行が見つかりません。"
            )

    def _display_generation_summary(self):
        """補正済みのサウンドファイルとテキストの読み一覧、バッチサイズと総生成数を表示"""
        print("\n" + "=" * 80)
        print("LLASA VOICE GENERATION SUMMARY / Llasa音声生成サマリー")
        print("=" * 80)

        # 音声ファイル一覧
        print(f"\nAUDIO FILES ({len(self.audio_data)} items) / 音声ファイル一覧 ({len(self.audio_data)}個):")
        print("-" * 60)
        for i, audio_data in enumerate(self.audio_data, 1):
            filename = os.path.basename(audio_data["file"])
            print(f"  {i:2d}. {filename} ({audio_data['duration']:.1f}s)")

        # テキスト読み一覧
        print(f"\nTEXT READINGS ({len(self.text_lines)} items) / テキスト読み一覧 ({len(self.text_lines)}個):")
        print("-" * 60)
        for i, text_data in enumerate(self.text_lines, 1):
            # 長いテキストは省略表示
            display_text = text_data["text"]
            if len(display_text) > 50:
                display_text = display_text[:47] + "..."
            print(f"  {i:2d}. {display_text}")

        # バッチサイズの計算
        batch_size = len(self.audio_data) * len(self.text_lines)

        print("\n" + "=" * 80)
        print("BATCH SIZE AND GENERATION COUNT / バッチサイズと生成数")
        print("=" * 80)
        print(f"Batch size: {len(self.audio_data)} audio × {len(self.text_lines)} text = {batch_size} combinations")
        print(
            f"バッチサイズ: 音声{len(self.audio_data)}個 × テキスト{len(self.text_lines)}行 = {batch_size}個の組み合わせ"
        )

        # バッチカウントに応じた総生成数の表示
        if self.batch_count == -1:
            print("\nGeneration mode: PERPETUAL / 生成モード: 永続生成")
            print("The system will generate continuously until stopped.")
            print("システムは停止されるまで継続的に生成します。")
        elif self.batch_count == 0:
            # サーバーモードでも対話的入力を可能にする
            print("\nGeneration mode: INTERACTIVE / 生成モード: 対話的")
            if self.server_mode:
                print("Server mode with interactive batch count / サーバーモードでの対話的バッチ数設定")
            try:
                user_input = input("Enter the number of batches to generate / 生成するバッチ数を入力してください: ")
                user_batches = int(user_input.strip())
                if user_batches <= 0 and user_batches != -1:
                    raise ValueError("Batch count must be greater than 0 / バッチ数は1以上である必要があります")
                # 対話で入力されたバッチ数を保存
                self.batch_count = user_batches
                if user_batches == -1:
                    print("Perpetual generation mode selected / 永続生成モードが選択されました")
                    print("The system will generate continuously until stopped (Ctrl+C)")
                    print("システムは停止されるまで継続的に生成します (Ctrl+C)")
                else:
                    total_generations = batch_size * user_batches
                    print(f"Total generations: {batch_size} × {user_batches} = {total_generations}")
                    print(f"総生成数: {batch_size} × {user_batches} = {total_generations}")
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(
                        "Invalid input. Please enter a valid number / 無効な入力です。有効な数値を入力してください"
                    )
                else:
                    raise e
        else:
            print(
                f"\nGeneration mode: SPECIFIED ({self.batch_count} batches) / 生成モード: 指定 ({self.batch_count}バッチ)"
            )
            total_generations = batch_size * self.batch_count
            print(f"Total generations: {batch_size} × {self.batch_count} = {total_generations}")
            print(f"総生成数: {batch_size} × {self.batch_count} = {total_generations}")

        print("=" * 80)

    def _is_readable_line(self, line):
        """行が読み上げ可能かどうかを判定する"""
        # 制御文字をチェック（タブは除く）
        if any(ord(char) < 32 and char not in ["\t"] for char in line):
            return False

        # 最小文字数チェック（1文字以上）
        if len(line.strip()) < 1:
            return False

        # Anime-Llasa-3B-Demoと同じ読み上げ可能文字パターンを使用
        # ひらがな、カタカナ、漢字、英字、数字、基本的な句読点を許可
        valid_pattern = re.compile(
            r"[\u3040-\u309F"  # ひらがな
            r"\u30A0-\u30FF"  # カタカナ
            r"\u4E00-\u9FFF"  # CJK統合漢字
            r"\u3400-\u4DBF"  # CJK拡張A
            r"\u3005"  # 々
            r"\u0041-\u005A"  # 大文字英字
            r"\u0061-\u007A"  # 小文字英字
            r"\u0030-\u0039"  # 数字
            r"。、!?…♪♡○"  # 基本的な句読点・記号
            r"\s]"  # 空白文字
        )

        # 有効な文字が含まれているかチェック
        valid_chars = len(valid_pattern.findall(line))
        total_chars = len(line.strip())

        # 少なくとも50%以上が有効な文字である必要がある
        return valid_chars > 0 and (valid_chars / total_chars) >= 0.5

    def normalize_text(self, text: str) -> str:
        """テキストを音声合成用に正規化する（Anime-Llasa-3B-Demoと同じ処理）"""
        # REPLACE_MAPによる置き換え
        for pattern, replacement in self.REPLACE_MAP.items():
            text = re.sub(pattern, replacement, text)

        # 文字種変換
        text = text.translate(self.FULLWIDTH_ALPHA_TO_HALFWIDTH)
        text = text.translate(self.FULLWIDTH_DIGITS_TO_HALFWIDTH)
        text = text.translate(self.HALFWIDTH_KATAKANA_TO_FULLWIDTH)

        # 連続する三点リーダーを二点リーダーに正規化
        text = re.sub(r"…{3,}", "……", text)

        # 無効文字を除去
        text = self.INVALID_PATTERN.sub("", text)
        text = text.strip()

        # 末尾の「。」を削除
        if text.endswith("。"):
            text = text[:-1]

        return text.strip()

    def _play_audio(self, audio_path, is_last=False):
        """
        ffplay.exeを使用して音声を再生する
        前回の再生がまだ実行中の場合は再生をスキップする（最後の音声以外）
        is_last=Trueの場合は再生完了まで待機する
        """
        if not self.play_audio:
            return

        # 前回の再生プロセスがまだ実行中かチェック（最後の音声以外）
        if not is_last and self.ffplay_process is not None:
            if self.ffplay_process.poll() is None:  # プロセスがまだ実行中
                print(
                    "⏭️  Skipping playback (previous playback is still running) / 再生をスキップ（前回の再生がまだ実行中）"
                )
                return

        try:
            # venv\Scripts内のffplay.exeの完全パスを構築
            script_dir = os.path.dirname(os.path.abspath(__file__))  # EasyLlasaフォルダ
            project_root = os.path.dirname(script_dir)  # プロジェクトルート
            ffplay_path = os.path.join(project_root, "venv", "Scripts", "ffplay.exe")

            # ffplay.exeが存在するかチェック
            if not os.path.exists(ffplay_path):
                # フォールバックでPATHからffplay.exeを探す
                ffplay_path = "ffplay.exe"

            # ffplay.exeで音声を再生（一回再生後自動終了、音量設定・再生速度付き）
            command = [ffplay_path, "-nodisp", "-autoexit", "-volume", str(int(self.volume * 100))]
            # 再生速度が1.0でない場合、atempoフィルターを追加
            if self.speed != 1.0:
                command.extend(["-af", f"atempo={self.speed}"])
            command.append(audio_path)
            self.ffplay_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if is_last:
                print(
                    f"🔊 Playing final audio (vol: {self.volume:.1f}): {os.path.basename(audio_path)} / 最終音声を再生中 (音量: {self.volume:.1f}): {os.path.basename(audio_path)}"
                )
                print("⏳ Waiting for final audio playback to complete... / 最終音声の再生完了を待機中...")
                # 最後の音声の場合は再生完了まで待機
                self.ffplay_process.wait()
                print("✅ Final audio playback completed / 最終音声の再生が完了しました")
            else:
                print(
                    f"🔊 Playing audio (vol: {self.volume:.1f}): {os.path.basename(audio_path)} / 音声を再生中 (音量: {self.volume:.1f}): {os.path.basename(audio_path)}"
                )
        except FileNotFoundError:
            print("⚠️  ffplay.exe not found in PATH / ffplay.exeがPATHに見つかりません")
        except Exception as e:
            print(f"⚠️  Failed to play audio: {e} / 音声再生に失敗: {e}")

    def _format_time(self, seconds):
        """秒を時:分:秒の形式でフォーマットする"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def _calculate_eta(self, completed_count, total_count, elapsed_time):
        """予想残り時間を計算する"""
        if completed_count == 0:
            return "不明"

        avg_time_per_generation = elapsed_time / completed_count
        remaining_generations = total_count - completed_count
        eta_seconds = avg_time_per_generation * remaining_generations

        return self._format_time(eta_seconds)

    def _process_single_generation(self, audio_data, text_data, generated_count, total_generations):
        """単一の音声生成処理を行う"""
        try:
            # 個別生成の開始時間
            single_generation_start = time.time()

            result = self._generate_speech(
                sample_audio_path=audio_data["file"],
                target_text=text_data["text"],
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

            if result is not None:
                sample_rate, audio_array = result

                # 現在時刻を取得してファイル名に使用
                now = datetime.now()
                timestamp = f"{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}-{now.second:02d}{now.microsecond//10000:02d}"

                # テキストの長さを調整（OSの制限255文字を考慮、32文字の余裕を持たせる）
                # timestamp(13文字) + "-" + ".wav"(4文字) = 18文字 + 余裕32文字 = 50文字を除いた205文字が使用可能
                available_length = 205
                text_part = text_data["text"].replace(" ", "_")

                # テキストの最大長を設定
                text_max = min(len(text_part), available_length)
                text_truncated = text_part[:text_max]

                # 出力ファイル名を生成（テキストのみ）
                output_filename = f"{timestamp}-{text_truncated}.wav"
                # ファイル名の無効文字を除去
                output_filename = re.sub(r'[<>:"/\\|?*]', "_", output_filename)
                output_path = os.path.join(self.output_dir, output_filename)

                # 出力ディレクトリを作成
                os.makedirs(self.output_dir, exist_ok=True)

                # 音声ファイルを保存
                sf.write(output_path, audio_array, sample_rate)
                print(f"Saved: {output_path} / 保存しました: {output_path}")

                # 時間情報を計算・表示（再生前に計測）
                single_generation_time = time.time() - single_generation_start
                total_elapsed_time = time.time() - self.generation_start_time
                self.generation_times.append(single_generation_time)

                # 最後の音声かどうかを判定
                is_last_audio = False
                if self.batch_count != -1 and total_generations != float("inf"):
                    # 有限回数モードの場合：次の生成で終了かどうか
                    is_last_audio = (generated_count + 1) >= total_generations

                # 音声再生（--playオプションが指定されている場合）
                self._play_audio(output_path, is_last=is_last_audio)

                print(f"⏱️  Generation time: {single_generation_time:.2f}s / 生成時間: {single_generation_time:.2f}秒")
                print(
                    f"📊 Total elapsed: {self._format_time(total_elapsed_time)} / 総経過時間: {self._format_time(total_elapsed_time)}"
                )

                if self.batch_count != -1 and total_generations != float("inf"):
                    eta = self._calculate_eta(generated_count + 1, total_generations, total_elapsed_time)
                    print(f"⏳ ETA: {eta} / 予想残り時間: {eta}")

                return True  # 成功
            else:
                print("Failed to generate audio / 音声生成失敗")
                return False  # 失敗

        except Exception as e:
            print(f"Error generating audio: {e} / 音声生成エラー: {e}")
            return False  # 失敗

    def _execute_server_generation(self, audio_data, text_data, generated_count):
        """サーバーモードでの単一音声生成処理"""
        generation_start = time.time()

        result = self._generate_speech(
            sample_audio_path=audio_data["file"],
            target_text=text_data["text"],
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )

        if result is not None:
            sample_rate, audio_array = result

            # 出力ファイル名を生成
            now = datetime.now()
            timestamp = f"{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}-{now.second:02d}{now.microsecond//10000:02d}"
            text_part = text_data["text"].replace(" ", "_")
            text_max = min(len(text_part), 205)
            text_truncated = text_part[:text_max]
            output_filename = f"{timestamp}-{text_truncated}.wav"
            output_filename = re.sub(r'[<>:"/\\|?*]', "_", output_filename)
            output_path = os.path.join(self.output_dir, output_filename)

            # 出力ディレクトリを作成
            os.makedirs(self.output_dir, exist_ok=True)

            # 音声ファイルを保存
            sf.write(output_path, audio_array, sample_rate)

            generation_time = time.time() - generation_start

            print(f"      ✅ Saved: {output_path}")
            print(f"      ⏱️  Generation time: {generation_time:.2f}s / 生成時間: {generation_time:.2f}秒")

            # 音声再生（--playオプションが指定されている場合）
            self._play_audio(output_path, is_last=False)

            return True  # 成功
        else:
            print("      ❌ Failed to generate audio / 音声生成失敗")
            return False  # 失敗

    def _setup_sage_attention(self):
        """SageAttentionの自動検出とセットアップ"""
        # 常に自動検出を試みる
        try:
            import sageattention

            print(
                "SageAttention auto-detected and enabled for attention optimization / SageAttentionを自動検出し、注意機構の最適化を有効化"
            )

            # SageAttentionの最適化を適用
            torch.backends.cuda.enable_math_sdp(False)  # デフォルトのSDPを無効化してSageAttentionを優先
            print("SageAttention optimization applied / SageAttention最適化を適用しました")

        except ImportError:
            print("SageAttention not available. Consider installing with: pip install sageattention")
            print("SageAttentionが利用できません。インストールを検討してください: pip install sageattention")
        except Exception as e:
            print(f"Warning: Failed to setup SageAttention: {e}")
            print(f"警告: SageAttentionのセットアップに失敗: {e}")

    def _load_models(self):
        """モデルを読み込む（量子化オプション対応）"""
        # テスト用シードの設定（再現性確保のため）
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            # numpyのシードも設定
            import numpy as np

            np.random.seed(self.seed)
            # transformersのset_seed関数も使用
            from transformers import set_seed

            set_seed(self.seed)
            print(f"Test random seed set to: {self.seed} / テスト用ランダムシードを設定: {self.seed}")

        if self.quantization_enabled:
            print("Loading models with 4bit quantization for VRAM efficiency...")
            print("4bit量子化でVRAM効率化のためモデルを読み込み中...")
        else:
            print("Loading models without quantization (higher quality, more VRAM usage)...")
            print("量子化なしでモデルを読み込み中（高品質、VRAM使用量は多め）...")

        # SageAttentionの自動検出と設定
        self._setup_sage_attention()

        try:
            # Llasaモデルとトークナイザーの読み込み
            print(f"Loading Llasa model: {self.model} / Llasaモデルを読み込み中: {self.model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

            # pad_tokenが設定されていない場合、eos_tokenを使用
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "cuda",
                "low_cpu_mem_usage": True,
            }

            # 量子化設定に応じてデータ型を調整（logits警告の根本解決）
            if self.quantization_enabled:
                model_kwargs["quantization_config"] = self.quantization_config
                # 量子化時はfloat16を明示的に指定してlogits型を統一
                model_kwargs["torch_dtype"] = torch.float16
            else:
                # 量子化なしの場合もfloat16で統一
                model_kwargs["torch_dtype"] = torch.float16

            self.llasa_model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
            self.llasa_model.eval()

            print("Llasa model loaded successfully / Llasaモデル読み込み完了")

            # XCodec2モデルの読み込み（デフォルトのfloat32を維持）
            xcodec2_model_id = "NandemoGHS/Anime-XCodec2"
            print(f"Loading XCodec2 model: {xcodec2_model_id} / XCodec2モデルを読み込み中: {xcodec2_model_id}")
            self.codec_model = XCodec2Model.from_pretrained(xcodec2_model_id)
            self.codec_model.eval().cuda()

            print("XCodec2 model loaded successfully / XCodec2モデル読み込み完了")

            # Whisperモデルの読み込み（transformers pipelineを使用、CUDA前提）
            print(
                f"Loading Whisper model ({self.whisper_model_name}) for speech recognition... / 音声認識用Whisperモデル ({self.whisper_model_name}) を読み込み中..."
            )

            # モデルIDに基づいて適切なパラメータを設定
            if self.whisper_model_name == "litagin/anime-whisper":
                # anime-whisperの場合（HuggingFaceページの仕様に完全準拠）
                self.whisper_model = pipeline(
                    "automatic-speech-recognition",
                    model=self.whisper_model_name,
                    device="cuda",
                    torch_dtype=torch.float16,
                    chunk_length_s=30.0,
                    batch_size=64,
                )
                # anime-whisper用の生成パラメータ（初期プロンプト使用禁止）
                self.whisper_generate_kwargs = {
                    "language": "Japanese",
                    "no_repeat_ngram_size": 0,
                    "repetition_penalty": 1.0,
                }
            elif self.whisper_model_name == "openai/whisper-large-v3-turbo":
                # whisper-large-v3-turboの場合（Anime-Llasa-3B-Demoの仕様に完全準拠）
                self.whisper_model = pipeline(
                    "automatic-speech-recognition",
                    model=self.whisper_model_name,
                    torch_dtype=torch.float16,
                    device="cuda",
                )
                self.whisper_generate_kwargs = None  # generate_kwargsは使用しない（Demo.pyと一致）
            else:
                # その他のWhisperモデルの場合（一般的なwhisperモデル用パラメータ）
                self.whisper_model = pipeline(
                    "automatic-speech-recognition",
                    model=self.whisper_model_name,
                    device="cuda",
                    torch_dtype=torch.float16,
                    chunk_length_s=30.0,
                    batch_size=16,  # 一般的なモデルでは控えめなバッチサイズ
                )
                # 一般的なwhisperモデル用の生成パラメータ
                self.whisper_generate_kwargs = {
                    "language": "ja",  # 一般的なwhisperモデルでは"ja"を使用
                    "task": "transcribe",
                }

            print(f"{self.whisper_model_name} model loaded successfully / {self.whisper_model_name}モデル読み込み完了")

            print("All models loaded successfully! / 全モデル読み込み完了！")

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e} / モデル読み込み失敗: {e}")

    def _ids_to_speech_tokens(self, speech_ids):
        """音声IDを音声トークン文字列に変換"""
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    def _extract_speech_ids(self, speech_tokens_str):
        """音声トークン文字列から音声IDを抽出"""
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith("<|s_") and token_str.endswith("|>"):
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids

    def _generate_speech(
        self,
        sample_audio_path: Optional[str],
        target_text: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
    ) -> Optional[Tuple[int, np.ndarray]]:
        """音声を生成する（Anime-Llasa-3B-Demoと同じ処理）"""
        if not target_text or not target_text.strip():
            print("Warning: Empty target text / 警告: 空のターゲットテキスト")
            return None

        if len(target_text) > 300:
            print(
                f"Warning: Text too long ({len(target_text)} chars), truncating to 300 / 警告: テキストが長すぎます（{len(target_text)}文字）、300文字に切り詰めます"
            )
            target_text = target_text[:300]

        target_text = self.normalize_text(target_text)

        with torch.no_grad():
            if sample_audio_path:
                print(f"Loading reference audio: {sample_audio_path} / 参照音声を読み込み中: {sample_audio_path}")
                waveform, sample_rate = torchaudio.load(sample_audio_path)

                if len(waveform[0]) / sample_rate > 15:
                    print("Warning: Trimming audio to first 15secs / 警告: 音声を最初の15秒に切り詰めます")
                    waveform = waveform[:, : sample_rate * 15]

                # ステレオをモノラルに変換
                if waveform.size(0) > 1:
                    waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
                else:
                    waveform_mono = waveform

                prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)
                prompt_wav_len = prompt_wav.shape[1]

                # 音声を転写（transformers pipelineを使用）
                audio_numpy = prompt_wav[0].numpy()

                if self.whisper_generate_kwargs:
                    # generate_kwargsがある場合（anime-whisper等）
                    result = self.whisper_model(audio_numpy, generate_kwargs=self.whisper_generate_kwargs)
                else:
                    # generate_kwargsがない場合（whisper-large-v3-turbo等、Anime-Llasa-3B-Demoと一致）
                    result = self.whisper_model(audio_numpy)
                prompt_text = result["text"].strip()

                print(f"Transcribed text: {prompt_text} / 転写されたテキスト: {prompt_text}")

                # 転写されたテキストを正規化
                prompt_text = self.normalize_text(prompt_text)
                print(f"Normalized transcribed text: {prompt_text} / 正規化された転写テキスト: {prompt_text}")

                # プロンプト音声をエンコード（XCodec2にはfloat32で渡す）
                prompt_wav_float32 = prompt_wav.to(torch.float32)
                vq_code_prompt = self.codec_model.encode_code(input_waveform=prompt_wav_float32)[0, 0, :]
                speech_ids_prefix = self._ids_to_speech_tokens(vq_code_prompt)
                input_text = prompt_text + " " + target_text
                assistant_content = "<|SPEECH_GENERATION_START|>" + "".join(speech_ids_prefix)
            else:
                input_text = target_text
                assistant_content = "<|SPEECH_GENERATION_START|>"
                speech_ids_prefix = []
                prompt_wav_len = 0

            print("Generating speech tokens... / 音声トークンを生成中...")
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": assistant_content},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors="pt", continue_final_message=True
            ).to("cuda")

            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(input_ids).to("cuda")

            speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

            outputs = self.llasa_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=2048,  # 固定値
                eos_token_id=[speech_end_id, self.tokenizer.eos_token_id],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                # 追加: 最低限いくらかの新規トークンを生成させ、極端に短い生成で<|SPEECH_GENERATION_END|>に到達してしまうのを回避
                # （短すぎるとプロンプト分を差し引いた後に空になり 1KB 前後のwavになる）
                min_new_tokens=16,
            )

            # Extract the speech tokens exactly like demo code
            if sample_audio_path:
                generated_ids = outputs[0][input_ids.shape[1] - len(speech_ids_prefix) : -1]
            else:
                generated_ids = outputs[0][input_ids.shape[1] : -1]

            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            speech_tokens = self._extract_speech_ids(speech_tokens)

            if not speech_tokens:
                print(
                    "Error: Audio generation failed - no speech tokens generated / エラー: 音声生成失敗 - 音声トークンが生成されませんでした"
                )
                return None

            # speech_tokensが数値のリストであることを確認
            try:
                speech_tokens = torch.tensor(speech_tokens, dtype=torch.long).cuda().unsqueeze(0).unsqueeze(0)
            except Exception as e:
                print(f"Error creating tensor from speech_tokens: {e}")
                print(f"speech_tokens content: {speech_tokens}")
                return None

            # 音声トークンを音声波形にデコード
            gen_wav = self.codec_model.decode_code(speech_tokens)

            # 生成部分のみが必要な場合
            if sample_audio_path and prompt_wav_len > 0:
                full_len = gen_wav.shape[-1]
                if full_len <= prompt_wav_len:
                    # ここで全て切り落とすと空になるのでガード
                    print(
                        "Warning: Generated waveform length <= prompt length. Skipping trimming to avoid empty audio. / 警告: 生成波形長がプロンプト長以下のためトリミングをスキップし空音声を回避"
                    )
                else:
                    gen_wav = gen_wav[:, :, prompt_wav_len:]

            print("Speech generation completed / 音声生成完了")
            return (16000, gen_wav[0, 0, :].cpu().numpy())

    def _generate_voices(self):
        """音声とテキストの組み合わせで音声を生成する"""
        # バッチ生成開始時にDialogue.txtを再読み込み
        try:
            self._load_dialogue_txt()
            print(
                f"Reloaded {self.text_file_path} for batch generation - {len(self.text_lines)} text lines available / バッチ生成用に{self.text_file_path}再読み込み - {len(self.text_lines)}行のテキストが利用可能"
            )
        except Exception as e:
            print(
                f"Warning: Failed to reload {self.text_file_path}: {e} / 警告: {self.text_file_path}の再読み込みに失敗: {e}"
            )
            print("Using previously loaded text lines... / 以前に読み込まれたテキスト行を使用します...")

        # バッチサイズの計算
        batch_size = len(self.audio_data) * len(self.text_lines)

        # 総生成数の計算
        if self.batch_count == -1:
            print("Perpetual generation mode - generating until interrupted / 永続生成モード - 中断されるまで生成")
            total_generations = float("inf")
        else:
            total_generations = batch_size * self.batch_count

        generated_count = 0
        current_batch = 0

        # 全体の生成開始時間を記録
        self.generation_start_time = time.time()

        try:
            # batch_count = -1の場合は永続生成モード（全組み合わせを繰り返し）
            if self.batch_count == -1:
                while True:  # 無限ループで全組み合わせを繰り返し
                    for j, text_data in enumerate(self.text_lines):
                        for i, audio_data in enumerate(self.audio_data):
                            print(
                                f"\nText: {text_data['text'][:50]}{'...' if len(text_data['text']) > 50 else ''} / テキスト: {text_data['text'][:50]}{'...' if len(text_data['text']) > 50 else ''}"
                            )
                            print(
                                f"Audio: {os.path.basename(audio_data['file'])} ({audio_data['duration']:.1f}s) / 音声: {os.path.basename(audio_data['file'])} ({audio_data['duration']:.1f}秒)"
                            )

                            print(
                                f"\nGenerating perpetual mode - {generated_count + 1}/∞ / 永続モード生成中 - {generated_count + 1}/∞"
                            )

                            # 音声を生成
                            if self._process_single_generation(
                                audio_data, text_data, generated_count, total_generations
                            ):
                                generated_count += 1
            else:
                # 通常のバッチモード
                for j, text_data in enumerate(self.text_lines):
                    for i, audio_data in enumerate(self.audio_data):
                        print(
                            f"\nText: {text_data['text'][:50]}{'...' if len(text_data['text']) > 50 else ''} / テキスト: {text_data['text'][:50]}{'...' if len(text_data['text']) > 50 else ''}"
                        )
                        print(
                            f"Audio: {os.path.basename(audio_data['file'])} ({audio_data['duration']:.1f}s) / 音声: {os.path.basename(audio_data['file'])} ({audio_data['duration']:.1f}秒)"
                        )

                        current_batch = 0
                        while current_batch < self.batch_count:
                            if generated_count >= total_generations:
                                break

                            print(
                                f"\nGenerating batch {current_batch + 1} - {generated_count + 1}/{total_generations} / バッチ{current_batch + 1}生成中 - {generated_count + 1}/{total_generations}"
                            )

                            # 音声を生成
                            if self._process_single_generation(
                                audio_data, text_data, generated_count, total_generations
                            ):
                                generated_count += 1

                            current_batch += 1

                        if generated_count >= total_generations:
                            break

                    if generated_count >= total_generations:
                        break

        except KeyboardInterrupt:
            if self.generation_start_time:
                interrupted_time = time.time() - self.generation_start_time
                print(
                    f"\n⏹️  Generation interrupted after {self._format_time(interrupted_time)} / {self._format_time(interrupted_time)}後に中断されました"
                )
            print(
                f"Generated {generated_count} files before interruption / 中断前に{generated_count}ファイルを生成しました"
            )

        # 総生成時間の表示
        if self.generation_start_time:
            total_generation_time = time.time() - self.generation_start_time
            print("\n" + "=" * 60)
            print("🎉 GENERATION SUMMARY / 生成サマリー")
            print("=" * 60)
            print(f"📁 Generated files: {generated_count} / 生成ファイル数: {generated_count}個")
            print(
                f"⏱️  Total generation time: {self._format_time(total_generation_time)} / 総生成時間: {self._format_time(total_generation_time)}"
            )

            if self.generation_times:
                avg_time = sum(self.generation_times) / len(self.generation_times)
                min_time = min(self.generation_times)
                max_time = max(self.generation_times)
                print(f"📊 Average time per file: {avg_time:.2f}s / 1ファイル平均時間: {avg_time:.2f}秒")
                print(f"⚡ Fastest generation: {min_time:.2f}s / 最速生成: {min_time:.2f}秒")
                print(f"🐌 Slowest generation: {max_time:.2f}s / 最遅生成: {max_time:.2f}秒")
            print("=" * 60)

        print(
            f"\nVoice generation completed! Generated {generated_count} audio files / 音声生成完了！{generated_count}個の音声ファイルを生成しました"
        )

    def _run_server_mode(self):
        """サーバーモード実行（名前付きパイプによるプロセス間通信）"""
        pipe_name = r"\\.\pipe\llasa_pipe"

        print(f"Starting named pipe server: {pipe_name}")
        print("Server mode ready. Waiting for connections... / サーバーモード準備完了。接続待機中...")
        print("Note: Dialogue.txt is reloaded on each request / 注意: 各リクエストでDialogue.txtが再読み込みされます")
        print("=" * 60)

        generated_count = 0
        pipe = None

        try:
            while True:
                try:
                    # 既存のパイプがあれば閉じる
                    if pipe is not None:
                        try:
                            win32file.CloseHandle(pipe)
                        except:
                            pass

                    # 名前付きパイプを作成
                    pipe = win32pipe.CreateNamedPipe(
                        pipe_name,
                        win32pipe.PIPE_ACCESS_DUPLEX,
                        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                        win32pipe.PIPE_UNLIMITED_INSTANCES,  # 複数インスタンスを許可
                        65536,
                        65536,
                        0,
                        None,
                    )

                    if pipe == win32file.INVALID_HANDLE_VALUE:
                        print("Failed to create named pipe / 名前付きパイプの作成に失敗")
                        break

                    print("Waiting for client connection... / クライアント接続待機中...")

                    # クライアントの接続を待つ
                    win32pipe.ConnectNamedPipe(pipe, None)
                    print("Client connected / クライアント接続完了")

                    # データを読み取る
                    try:
                        result, data = win32file.ReadFile(pipe, 4096)
                        if result == 0:  # ERROR_SUCCESS
                            message = data.decode("utf-8").strip()
                            print(f"Received: {message}")

                            if message.lower() in ["quit", "exit", "q"]:
                                print("Server mode terminated. / サーバーモード終了。")
                                win32file.CloseHandle(pipe)
                                break

                            # 受信したパスの処理（改行区切り）
                            paths = [path.strip() for path in message.split("\n") if path.strip()]

                            for input_path in paths:
                                # パスの存在確認
                                if not os.path.exists(input_path):
                                    print(
                                        f"Error: Path not found: {input_path} / エラー: パスが見つかりません: {input_path}"
                                    )
                                    continue

                                # デーモンモードでは処理前にDialogue.txtを再読み込み
                                try:
                                    self._load_dialogue_txt()
                                    print(
                                        f"Reloaded {self.text_file_path} - {len(self.text_lines)} text lines available / {self.text_file_path}再読み込み - {len(self.text_lines)}行のテキストが利用可能"
                                    )
                                except Exception as e:
                                    print(
                                        f"Warning: Failed to reload {self.text_file_path}: {e} / 警告: {self.text_file_path}の再読み込みに失敗: {e}"
                                    )
                                    if not hasattr(self, "text_lines") or not self.text_lines:
                                        print(
                                            "❌ No text lines available. Skipping... / テキスト行がありません。スキップします..."
                                        )
                                        continue

                                # 入力パスから音声ファイルを収集
                                collected_files = self._collect_audio_files_from_path(input_path)

                                if not collected_files:
                                    print(
                                        f"❌ No audio files found in: {input_path} / 音声ファイルが見つかりません: {input_path}"
                                    )
                                    continue

                                # BatchGenerateと同様に音声データを前処理
                                self.sound_files = collected_files
                                temp_audio_data = []

                                print(
                                    f"\n📁 Processing {len(collected_files)} audio file(s) / {len(collected_files)}個の音声ファイルを処理中"
                                )

                                for audio_file in collected_files:
                                    try:
                                        audio_array, sample_rate = sf.read(audio_file)
                                        if len(audio_array.shape) > 1:
                                            audio_array = np.mean(audio_array, axis=1)

                                        duration = len(audio_array) / sample_rate
                                        temp_audio_data.append(
                                            {
                                                "file": audio_file,
                                                "array": audio_array,
                                                "sample_rate": sample_rate,
                                                "duration": duration,
                                            }
                                        )
                                        print(f"  ✅ {os.path.basename(audio_file)} ({duration:.1f}s)")
                                    except Exception as e:
                                        print(f"  ❌ Error loading {os.path.basename(audio_file)}: {e}")

                                if not temp_audio_data:
                                    print("❌ No valid audio data processed / 有効な音声データが処理されませんでした")
                                    continue

                                # BatchGenerateと同じ順序：テキスト→音声→バッチの順
                                for j, text_data in enumerate(self.text_lines, 1):
                                    print(
                                        f"\n📝 Text {j}/{len(self.text_lines)}: {text_data['text'][:30]}{'...' if len(text_data['text']) > 30 else ''}"
                                    )

                                    for i, audio_data in enumerate(temp_audio_data, 1):
                                        print(
                                            f"  🎵 Audio {i}/{len(temp_audio_data)}: {os.path.basename(audio_data['file'])} ({audio_data['duration']:.1f}s)"
                                        )

                                        # バッチ数分生成（サーバーモードではself.batch_countを使用）
                                        if hasattr(self, "batch_count") and self.batch_count == -1:
                                            # 永続生成モード：このリクエストから収集された音声で永続生成
                                            print(
                                                f"    🔁 Perpetual mode started for this request / このリクエストで永続生成開始"
                                            )
                                            print(
                                                f"    ⚠️  Use Ctrl+C to stop perpetual generation / 永続生成を停止するにはCtrl+Cを使用"
                                            )

                                            perpetual_count = 0
                                            try:
                                                while True:  # 永続ループ
                                                    perpetual_count += 1
                                                    print(
                                                        f"    🔄 Perpetual batch {perpetual_count} - {generated_count + 1}/∞"
                                                    )

                                                    # 音声生成を実行
                                                    self._execute_server_generation(
                                                        audio_data, text_data, generated_count
                                                    )
                                                    generated_count += 1

                                            except KeyboardInterrupt:
                                                print(
                                                    f"\n⏹️  Perpetual generation interrupted after {perpetual_count} iterations"
                                                )
                                                print(f"永続生成が{perpetual_count}回の反復後に中断されました")
                                                break
                                        else:
                                            # 通常のバッチモード
                                            batch_count_for_server = (
                                                self.batch_count
                                                if hasattr(self, "batch_count") and self.batch_count > 0
                                                else 1
                                            )

                                            for batch_num in range(batch_count_for_server):
                                                if batch_count_for_server > 1:
                                                    print(f"    🔄 Batch {batch_num + 1}/{batch_count_for_server}")

                                                # 音声生成を実行
                                                self._execute_server_generation(audio_data, text_data, generated_count)
                                                generated_count += 1

                            # サーバーモード処理完了
                            print(
                                f"\n✅ Server request completed: {generated_count} files generated / サーバーリクエスト完了: {generated_count}個のファイルを生成"
                            )

                            # 完了メッセージをクライアントに送信
                            try:
                                response = f"Processing completed. Generated {generated_count} files."
                                win32file.WriteFile(pipe, response.encode("utf-8"))
                                win32file.FlushFileBuffers(pipe)
                            except:
                                pass

                    except pywintypes.error as e:
                        print(f"Pipe read error: {e} / パイプ読み取りエラー: {e}")

                    # クライアント接続を切断
                    try:
                        win32pipe.DisconnectNamedPipe(pipe)
                    except:
                        pass

                except pywintypes.error as e:
                    print(f"Pipe connection error: {e} / パイプ接続エラー: {e}")
                    if "すべてのパイプ インスタンスがビジー" in str(e):
                        print("Pipe busy, waiting and retrying... / パイプビジー、待機後再試行...")
                        time.sleep(1)
                        continue

        except KeyboardInterrupt:
            print("\nServer mode interrupted. / サーバーモード中断。")
        except Exception as e:
            print(f"Unexpected error: {e} / 予期しないエラー: {e}")
        finally:
            # 最終的なクリーンアップ
            if pipe is not None:
                try:
                    win32pipe.DisconnectNamedPipe(pipe)
                    win32file.CloseHandle(pipe)
                except:
                    pass

        print(
            f"Server mode finished. Generated {generated_count} files. / サーバーモード終了。{generated_count}ファイル生成。"
        )

    def run(self):
        """メイン実行関数"""
        try:
            # 引数を解析
            self.parse_arguments()

            # 初期化開始メッセージと時間計測開始
            print("\n" + "=" * 60)
            print("INITIALIZING SYSTEM / システム初期化中")
            print("=" * 60)
            print("Starting initialization... / 初期化を開始しています...")

            init_start_time = time.time()

            # サーバーモードの場合は、ファイル処理をスキップ（リクエスト時に処理するため）
            if self.server_mode:
                self.text_lines = []
                self.audio_data = []
                print("Server mode: Skipping file processing at startup (processed per request)")
                print("サーバーモード: 起動時のファイル処理をスキップ（リクエストごとに処理）")

                # サーバーモードでもバッチ数の入力処理は必要
                print("\n" + "=" * 80)
                print("BATCH SIZE AND GENERATION COUNT / バッチサイズと生成数")
                print("=" * 80)
                print("Server mode configuration / サーバーモード設定")
                print("サーバーモード設定")

                # バッチ数に応じた総生成数の表示
                if self.batch_count == -1:
                    print("\nGeneration mode: PERPETUAL / 生成モード: 永続生成")
                    print("The system will generate continuously until stopped.")
                    print("システムは停止されるまで継続的に生成します。")
                elif self.batch_count == 0:
                    # サーバーモードでも対話的入力を可能にする
                    print("\nGeneration mode: INTERACTIVE / 生成モード: 対話的")
                    print("Server mode with interactive batch count / サーバーモードでの対話的バッチ数設定")
                    try:
                        user_input = input(
                            "Enter the number of batches to generate / 生成するバッチ数を入力してください: "
                        )
                        user_batches = int(user_input.strip())
                        if user_batches <= 0 and user_batches != -1:
                            raise ValueError("Batch count must be greater than 0 / バッチ数は1以上である必要があります")
                        # 対話で入力されたバッチ数を保存
                        self.batch_count = user_batches
                        if user_batches == -1:
                            print("Perpetual generation mode selected / 永続生成モードが選択されました")
                            print("The system will generate continuously until stopped (Ctrl+C)")
                            print("システムは停止されるまで継続的に生成します (Ctrl+C)")
                        else:
                            print(f"Batch count set to: {user_batches} / バッチ数を設定: {user_batches}")
                    except ValueError as e:
                        if "invalid literal" in str(e):
                            raise ValueError(
                                "Invalid input. Please enter a valid number / 無効な入力です。有効な数値を入力してください"
                            )
                        else:
                            raise e
                else:
                    print(
                        f"\nGeneration mode: SPECIFIED ({self.batch_count} batches) / 生成モード: 指定 ({self.batch_count}バッチ)"
                    )
                print("=" * 80)
            else:
                # 通常モードでのファイル収集とテキスト処理
                # Dialogue.txtからlinesを集める。空行や読み上げられない行はスキップする。
                self.text_lines = []
                self._load_dialogue_txt()

                # サウンドファイルを収集し、Whisperで転写して使用する。
                # ファイル名に依存しない音声認識ベースの処理。
                self.audio_data = []
                self._collect_audio_data()

                # サマリー表示と総生成数の計算
                self._display_generation_summary()

            print(f"Model: {self.model}")
            print(f"Quantization: {'Enabled' if self.quantization_enabled else 'Disabled'}")
            print(f"Server mode: {'Enabled' if self.server_mode else 'Disabled'}")
            print(f"Paths: {self.paths}")
            print(f"Batch count: {self.batch_count}")
            print(f"Text lines ready for processing: {len(self.text_lines)}")
            print(f"Valid audio data: {len(self.audio_data)}")

            # すべての情報表示完了後、モデルを読み込み
            print("\n" + "=" * 60)
            print("LOADING MODELS / モデル読み込み")
            print("=" * 60)
            self._load_models()

            # 初期化完了メッセージと所要時間表示
            init_end_time = time.time()
            init_duration = init_end_time - init_start_time
            print("\n" + "=" * 60)
            print("INITIALIZATION COMPLETE / 初期化完了")
            print("=" * 60)
            print(
                f"Initialization completed in {init_duration:.2f} seconds / 初期化が{init_duration:.2f}秒で完了しました"
            )
            print("Ready to proceed! / 処理開始準備完了！")

            if self.server_mode:
                print("\n" + "=" * 60)
                print("STARTING SERVER MODE / サーバーモード開始")
                print("=" * 60)
                self._run_server_mode()
            else:
                print("\n" + "=" * 60)
                print("STARTING VOICE GENERATION / 音声生成開始")
                print("=" * 60)
                self._generate_voices()

        except ValueError as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to exit / Enterキーを押して終了...")
            return 1  # エラーコードを返す
        except Exception as e:
            print(f"\n❌ Unexpected error: {e} / 予期しないエラー: {e}")
            input("Press Enter to exit / Enterキーを押して終了...")
            return 1

        return 0  # 正常終了


def main():
    """メイン関数"""
    llasa = Llasa()
    exit_code = llasa.run()
    exit(exit_code)


if __name__ == "__main__":
    main()
