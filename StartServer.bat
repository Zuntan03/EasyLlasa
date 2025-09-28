@echo off
chcp 65001 > nul

@REM ヘルプ表示チェック（--help または /? の場合）
if "%~1"=="--help" goto :show_help
if "%~1"=="/?" goto :show_help
if "%~1"=="-h" goto :show_help
goto :main

:show_help
echo ===============================================
echo Llasa Server Mode / Llasa サーバーモード
echo ===============================================
echo.
echo Usage / 使用方法:
echo   StartServer.bat
echo.
echo Description / 説明:
echo   Starts Llasa in server mode for continuous audio generation.
echo   継続的な音声生成のためにLlasaをサーバーモードで開始します。
echo.
echo How it works / 動作方式:
echo   1. Keeps the AI model loaded in memory for fast processing
echo      高速処理のためAIモデルをメモリに常駐
echo   2. Waits for audio file requests via named pipe
echo      名前付きパイプ経由で音声ファイルリクエストを待機
echo   3. Generates audio for each text line in Dialogue.txt
echo      Dialogue.txtの各テキスト行に対して音声生成
echo      (Fixed to use Dialogue.txt only)
echo      （Dialogue.txt固定使用）
echo   4. Saves generated files to Output folder
echo      生成ファイルをOutputフォルダに保存
echo.
echo Usage with client / クライアントでの使用:
echo   - Drag audio files to Generate.bat to send requests
echo     音声ファイルをGenerate.batにドラッグしてリクエスト送信
echo   - Or use command line: Generate.bat "path\to\audio.wav"
echo     またはコマンドライン: Generate.bat "path\to\audio.wav"
echo.
echo Examples / 例:
echo   StartServer.bat
echo.
echo Supported audio formats / 対応音声形式:
echo   WAV, FLAC, OGG, MP3, AIFF, AU, CAF
echo.
echo Server control / サーバー制御:
echo   - Type 'quit' or 'q' in server console to stop
echo     サーバーコンソールで 'quit' または 'q' を入力して停止
echo   - Ctrl+C for emergency stop
echo     緊急停止はCtrl+C
echo.
echo Customization / カスタマイズ:
echo   Edit this batch file to modify settings:
echo   このバッチファイルを編集して設定を変更:
echo.
echo   PLAY=--play --volume 0.7 --speed 1.0
echo     Controls audio playback, volume (0.0-2.0), and speed (0.5-2.0)
echo     音声再生、音量(0.0-2.0)、再生速度(0.5-2.0)を制御
echo     Remove "--play" to disable auto-playback
echo     "--play"を削除すると自動再生を無効化
echo     Adjust "--speed" for faster/slower playback (1.0=normal)
echo     "--speed"で再生速度を調整（1.0=通常速度）
echo.
echo   BATCH_COUNT=1
echo     Number of audio files to generate per text line
echo     各テキスト行に対して生成する音声ファイル数
echo.
echo   @REM set HF_HOME=huggingface_cache
echo     Uncomment by removing '@REM ' to temporarily use local cache
echo     '@REM 'を削除してアンコメントすることでローカルキャッシュを使用
echo     This avoids using user folder for Huggingface model cache
echo     Huggingfaceモデルキャッシュでユーザーフォルダを使用しなくなります
echo     WARNING: Disables model sharing between multiple processes
echo     注意: 複数プロセス間でのモデル共有ができなくなります
echo.
echo Advanced customization / 高度なカスタマイズ:
echo   To use custom text file, modify the command line in this batch file:
echo   カスタムテキストファイルを使用するには、このバッチファイル内のコマンドラインを変更:
echo   Add: --text "YourCustomFile.txt"
echo   追加: --text "YourCustomFile.txt"
echo.
pause
exit /b 0

:main
@REM 環境変数設定
set PLAY=--play --volume 0.7 --speed 1.0
set BATCH_COUNT=1
@REM set HF_HOME=huggingface_cache

echo ===============================================
echo Llasa Pipe Server Mode / Llasa パイプサーバーモード
echo ===============================================
echo.
echo Starting pipe server mode... / パイプサーバーモード開始中...
echo Use Generate.bat to send audio processing requests
echo Generate.batを使用して音声処理リクエストを送信
echo.

@REM パイプサーバーモードで Llasa を起動
"%~dp0EasyLlasa\Llasa.bat" ^
--server ^
--quantization ^
-b %BATCH_COUNT% ^
%PLAY% %*

if %ERRORLEVEL% neq 0 (
    echo Error: Llasa server mode terminated with an error.
    echo エラー: Llasa サーバーモードがエラーで終了しました。
    pause & exit /b 1
)
