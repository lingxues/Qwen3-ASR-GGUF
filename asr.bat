@echo off
chcp 65001 >nul 2>&1
setlocal

if "%~1"=="" (
    echo Please drag audio file onto this bat
    pause
    exit /b
)

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "SCRIPT_DIR=%~dp0"
set "AUDIO_FILE=%~1"

echo Processing: %~n1
echo.

python "%SCRIPT_DIR%transcribe.py" "%AUDIO_FILE%" --prec fp16 -y

echo.
pause