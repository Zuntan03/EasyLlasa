@echo off
chcp 65001 > NUL

if exist "%~dp0..\venv\" ( goto :VENV_EXISTS )
call "%~dp0..\Update.bat"
if %ERRORLEVEL% neq 0 ( exit /b 1 )
:VENV_EXISTS
pushd "%~dp0.."

call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 ( popd & exit /b 1 )

python EasyLlasa\llasa.py %*
if %ERRORLEVEL% neq 0 ( popd & exit /b %ERRORLEVEL% )

popd rem "~dp0.."
