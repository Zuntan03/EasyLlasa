@echo off
chcp 65001 >nul

REM Arguments check - delegate help to Request.ps1
if "%~1"=="" (
    powershell -ExecutionPolicy Bypass -File "%~dp0EasyLlasa\Request.ps1" --help
    exit /b 1
)

REM Use Request.bat to send the request with all arguments
call "%~dp0EasyLlasa\Request.bat" %*

if %ERRORLEVEL% NEQ 0 (
    exit /b %ERRORLEVEL%
)

echo.
echo Generate.bat completed successfully
timeout /t 3 /nobreak >nul
