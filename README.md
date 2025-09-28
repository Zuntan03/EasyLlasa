# <ruby>EasyLlasa<rt>ｲｰｼﾞｰ ﾗｻ</rt></ruby>

実験的な環境です。

EasyLlasa は 5～15秒の日本語音声と日本語テキストから日本語音声を生成する TSTS (TextSpeechToSpeech) です。

[Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B) をローカル PC で試せる環境です。  
Geforce RTX 3060 12GB 以上を搭載した Windows PC が必要です（1音声 15秒程度）。

[Anime-Llasa-3B-Demo](https://huggingface.co/spaces/OmniAICreator/Anime-Llasa-3B-Demo) でオンラインデモを試せますので、インストール前にどうぞ。

## インストール

1. [EasyLlasaInstaller.bat](https://github.com/Zuntan03/EasyLlasa/raw/main/EasyLlasa/EasyLlasaInstaller.bat?ver=0) を右クリックから保存します。
	- リンクを開いてから右クリックから保存すると、`*.bat` ファイルでなく `*.txt` ファイルになり実行できなくなります。
2. インストール先の **空フォルダ** を `C:/EasyLlasa/` や `D:/EasyLlasa/` などの浅いパスに用意して、`EasyLlasaInstaller.bat` を置いて実行します。
	- **`発行元を確認できませんでした。このソフトウェアを実行しますか？` と表示されたら `実行` します。**
	- **`WindowsによってPCが保護されました` と表示されたら、`詳細表示` から `実行` します。**
	- **`Microsoft Visual C++ 2015-2022 Redistributable` のインストールで `このアプリがデバイスに変更を加えることを許可しますか？` と表示されたら `はい` とします。**

## チュートリアル

1. `StartServer.bat` を実行してモデルのロードを待ちます。
2. お好みのテキストエディタで `Dialogue.txt` を開いて、1行 1音声でテキストを入力します。
3. `Generate.bat` に 5~15秒程度の音声ファイルをドラッグ＆ドロップします。
4. 生成が終わると `Output/` に保存しつつプレビュー再生します。

- ランダム幅の広い生成ですので、ガチャってみてください。
	- `BatchGenerate.bat` で大量の音声をバッチで一括生成したり、バッチ数 `-1` で永続生成したりもできます。
- 音声ファイルの続きとしてテキストの音声が生成されますので、テキストに合わせて馴染みそうな音声ファイルをご利用ください。

## 最近の更新

### 2025/09/28

- `StartServer.bat` と `BatchGenerate.bat` に `@REM set HF_HOME=huggingface_cache` を追加しました。
	- `set HF_HOME=huggingface_cache` とアンコメントすることで、Windows のユーザーフォルダのストレージ消費を回避できます。
	- ただし、Huggingace ライブラリの思想である「複数プロセスでのモデル共有」ができなくなります。
- pip と Huggingface の共有キャッシュを削除する `PurgeSharedCache.bat` を追加しました。
	- 共有キャッシュを削除すると、次回の pip パッケージインストール時や Huggingface モデルの利用時に再ダウンロードが発生します。

### 2025/09/25

- EasyLlasa を公開しました。
- かなりのバイブコーディング製です。
	- のでバグったら AI 頼りですので、直せなかったらごめんなさい。

## `StartServer.bat` と `BatchGenerate.bat`

以下のプレビュー再生とデフォルトのバッチ数は `StartServer.bat` や `BatchGenerate.bat` をコピーするなどして書き換えてください。

```
set PLAY=--play --volume 0.7 --speed 1.0
set BATCH_COUNT=0
```

### 引数仕様（AI 生成）

`StartServer.bat` と `BatchGenerate.bat` は同じ引数を指定できます。

#### 基本設定 (Core Settings)
- `-m, --model`: 使用するモデル名（デフォルト: `NandemoGHS/Anime-Llasa-3B`）
- `-t, --text`: テキストファイルパス（デフォルト: `Dialogue.txt`）
- `-o, --output`: 出力ディレクトリ（デフォルト: `Output`）

#### 動作モード (Operation Mode)
- `--server`: サーバーモードで実行（名前付きパイプによるプロセス間通信）
- `-b, --batch_count`: バッチ生成回数
  - `0`: プロンプトで確認（対話的入力）
  - `-1`: 永続生成（Ctrl+Cで停止まで継続）
  - `その他の数値`: 指定回数生成

#### パフォーマンス・最適化 (Performance Optimization)
- `-q, --quantization`: 4bit量子化を有効化

#### 生成パラメータ (Generation Parameters)
- `--temperature`: テキスト生成の温度パラメータ（デフォルト: 0.8）
- `--top_p`: ニューリアスサンプリングのtop-p値（デフォルト: 1.0）
- `--repetition_penalty`: 繰り返しペナルティ（デフォルト: 1.1）

#### 出力・再生 (Output & Playback)
- `-p, --play`: 生成した音声をffplay.exeで自動再生
- `-v, --volume`: 再生音量レベル（0.0-2.0、デフォルト: 1.0）
- `-s, --speed`: 再生速度（0.5-2.0、デフォルト: 1.0）

#### テスト・デバッグ (Testing & Debug)
- `--test-seed`: 再現可能なテスト結果のためのランダムシード（テスト目的のみ）

## ライセンス

このリポジトリの内容は [MIT License](./LICENSE.txt) です。  
別途ライセンスファイルがあるフォルダ以下は、そのライセンスです。
