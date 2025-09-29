@echo off
chcp 65001 >nul

rem Request.bat - Send audio file paths to Llasa server via named pipe

if "%~1" == "" (
    echo Usage: Request.bat ^<audio_file_or_folder_path^> [additional_paths...]
    echo Example: Request.bat "C:\path\to\voice.wav"
    echo Example: Request.bat "C:\path\to\audio_folder"
    echo Example: Request.bat "C:\path\to\voice1.wav" "C:\path\to\voice2.wav"
    echo Note: Multiple files/folders can be dragged and dropped
    echo 注意: 複数のファイル・フォルダをドラッグ＆ドロップ可能
    pause
    exit /b 1
)

setlocal enabledelayedexpansion

echo Checking paths... / パス確認中...

set "VALID_COUNT=0"
set "PATHS_FILE=%~dp0Request-Paths.txt"
echo. > "%PATHS_FILE%"

rem Check all arguments and collect valid paths
:loop
if "%~1"=="" goto :check_valid
if exist "%~1" (
    set /a VALID_COUNT+=1
    echo Valid: "%~1"
    echo "%~1" >> "%PATHS_FILE%"
) else (
    echo Warning: Path not found - "%~1"
)
shift
goto :loop

:check_valid

if %VALID_COUNT% EQU 0 (
    echo Error: No valid paths found
    echo エラー: 有効なパスが見つかりません
    pause
    exit /b 1
)

echo.
echo Found %VALID_COUNT% valid path(s) / %VALID_COUNT%個の有効なパス
echo Text will be loaded from Dialogue.txt / テキストはDialogue.txtから読み込まれます
echo.

echo Sending request to Llasa server via named pipe...

rem PowerShellスクリプトファイルを呼び出し（パスファイルを渡す）
rem PowerShell script is in the same directory as this batch file
set "PS_SCRIPT=.\Request.ps1"

powershell -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -pathsFile "%PATHS_FILE%"
set "PS_EXIT_CODE=%ERRORLEVEL%"

rem パスファイルを削除
if exist "%PATHS_FILE%" del "%PATHS_FILE%" >nul 2>&1

if %PS_EXIT_CODE% neq 0 (
    echo Warning: Could not send to server pipe. Server may not be running.
    echo 警告: サーバーパイプに送信できませんでした。サーバーが実行されていない可能性があります。
    echo Make sure to start StartServer.bat first
    echo 先にStartServer.batを実行してください
    pause
    exit /b %PS_EXIT_CODE%
)

echo.
echo ===============================================
echo Request completed successfully / リクエスト送信完了
echo ===============================================
echo Check server console for processing
echo サーバーコンソールで処理状況を確認してください
echo.
echo Returning to Generate.bat... / Generate.batに戻ります...
