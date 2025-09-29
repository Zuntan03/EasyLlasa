@echo off
chcp 65001 > NUL
set CURL_CMD=C:\Windows\System32\curl.exe -kL
set EASY_TOOLS=%~dp0..\EasyTools
set PYTHON_ACTIVATE=%EASY_TOOLS%\Python\Python_Activate.bat

if exist %~dp0vc_redist.x64.exe ( goto :EXIST_VC_REDIST_X64 )
echo.
echo %CURL_CMD% -o %~dp0vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
%CURL_CMD% -o %~dp0vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
if %ERRORLEVEL% neq 0 ( pause & exit /b 1 )
start %~dp0vc_redist.x64.exe /install /passive /norestart
:EXIST_VC_REDIST_X64

pushd "%~dp0.."

if not exist Dialogue.txt (
	echo copy /Y EasyLlasa\Dialogue.txt .
	copy /Y EasyLlasa\Dialogue.txt .
)

call %PYTHON_ACTIVATE%
if %ERRORLEVEL% neq 0 ( popd & exit /b 1 )

call %EASY_TOOLS%\Ffmpeg\Ffmpeg_Setup.bat venv\Scripts
if %ERRORLEVEL% neq 0 ( popd & exit /b 1 )

echo python -m pip install -qq -U pip "setuptools<81" wheel
python -m pip install -qq -U pip "setuptools<81" wheel
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

echo pip install -qq transformers==4.45.2 soundfile==0.13.1 bitsandbytes==0.47.0 accelerate==1.10.1 torchao==0.13.0 pywin32==311
pip install -qq transformers==4.45.2 soundfile==0.13.1 bitsandbytes==0.47.0 accelerate==1.10.1 torchao==0.13.0 pywin32==311
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

echo pip install -qq xcodec2==0.1.5
pip install -qq xcodec2==0.1.5
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

echo pip install -qq huggingface_hub[hf_xet]
pip install -qq huggingface_hub[hf_xet]
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

echo pip install -qq triton-windows==3.3.1.post19
pip install -qq triton-windows==3.3.1.post19
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

set "TRITON_CACHE=C:\Users\%USERNAME%\.triton\cache"
if not exist "%TRITON_CACHE%" ( goto :EASY_TRITON_CACHE_NOT_FOUND )
echo rmdir /S /Q "%TRITON_CACHE%"
rmdir /S /Q "%TRITON_CACHE%"
:EASY_TRITON_CACHE_NOT_FOUND

set "TORCH_INDUCTOR_TEMP=C:\Users\%USERNAME%\AppData\Local\Temp\torchinductor_%USERNAME%"
if not exist "%TORCH_INDUCTOR_TEMP%" ( goto :EASY_TORCH_INDUCTOR_TEMP_NOT_FOUND )
echo rmdir /S /Q "%TORCH_INDUCTOR_TEMP%"
rmdir /S /Q "%TORCH_INDUCTOR_TEMP%"
:EASY_TORCH_INDUCTOR_TEMP_NOT_FOUND

if exist %EASY_PORTABLE_PYTHON_DIR%\ (
	@REM tcc.exe & VS Build Tools cl.exe
	if not exist venv\Scripts\Include\Python.h (
		echo xcopy /SQY %EASY_PORTABLE_PYTHON_DIR%\include\*.* venv\Scripts\Include\
		xcopy /SQY %EASY_PORTABLE_PYTHON_DIR%\include\*.* venv\Scripts\Include\
		echo xcopy /SQY %EASY_PORTABLE_PYTHON_DIR%\libs\*.* venv\Scripts\libs\
		xcopy /SQY %EASY_PORTABLE_PYTHON_DIR%\libs\*.* venv\Scripts\libs\
	)
)

echo pip install -qq https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post2/sageattention-2.2.0+cu128torch2.7.1.post2-cp39-abi3-win_amd64.whl
pip install -qq https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post2/sageattention-2.2.0+cu128torch2.7.1.post2-cp39-abi3-win_amd64.whl
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

echo pip install -qq torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -qq torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 2>NUL
if %ERRORLEVEL% neq 0 ( pause & popd & exit /b 1 )

popd rem "~dp0.."
