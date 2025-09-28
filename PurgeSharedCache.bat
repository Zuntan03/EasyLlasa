@echo off
chcp 65001 >nul

echo This script purges the shared cache.
echo It calls Pip_PurgeCache.bat to purge Python pip cache.
echo It calls Huggingface_DeleteCache.bat to delete Huggingface cache.
echo Caution: Deleting shared cache may cause re-downloading during next pip package installation or Huggingface model usage.
echo.
echo "このスクリプトは共有キャッシュをパージします。"
echo "Pip_PurgeCache.bat を呼び出して Python pip キャッシュをパージします。"
echo "Huggingface_DeleteCache.bat を呼び出して Huggingface キャッシュを削除します。"
echo "注意：共有キャッシュを削除すると、次回の pip パッケージインストール時や Huggingface モデルの利用時に再ダウンロードが発生します。"
echo.

echo Do you want to continue? / 続行しますか？ (Y/N):
set /p confirm=
if /i not "%confirm%"=="Y" (
    echo Operation cancelled. / 操作がキャンセルされました。
    pause & exit /b 1
)

pushd "%~dp0"
@REM 先にセットアップが必要
if not exist venv\Scripts\activate.bat (
    echo Python virtual environment not found. Please run Update.bat first.
    echo "Python の仮想環境が見つかりません。先に Update.bat を実行してください。"
    pause & popd & exit /b 1
)

call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 ( popd & exit /b 1 )

echo pip cache purge
pip cache purge
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

if not exist "%USERPROFILE%\.cache\huggingface" ( popd & exit /b 0 )

echo rmdir /S /Q "%USERPROFILE%\.cache\huggingface"
rmdir /S /Q "%USERPROFILE%\.cache\huggingface"
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

popd rem %~dp0
