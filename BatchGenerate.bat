@echo off
chcp 65001 >nul

REM Arguments check - delegate help to separate PowerShell file
if "%~1" == "" (
    powershell -ExecutionPolicy Bypass -File "%~dp0EasyLlasa\BatchGenerateHelp.ps1"
    exit /b 1
)

REM Environment settings
set PLAY=--play --volume 0.7 --speed 1.0
set BATCH_COUNT=0

REM Run Llasa in batch generation mode
"%~dp0EasyLlasa\Llasa.bat" ^
--quantization ^
-b %BATCH_COUNT% ^
%PLAY% %*

if %ERRORLEVEL% neq 0 (
    exit /b %ERRORLEVEL%
)

echo.
echo BatchGenerate.bat completed successfully
echo Check the Output folder for generated files
