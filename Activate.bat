@echo off
chcp 65001 > NUL
cd /d %~dp0
call venv\Scripts\activate.bat
cmd /k
