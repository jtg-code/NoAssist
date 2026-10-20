@echo off
rem Startet NoAssist im Dev-Modus (Live-Reload). Abhaengigkeiten werden nur
rem installiert, wenn .venv fehlt oder requirements.txt sich geaendert hat.
setlocal
cd /d "%~dp0"

set "NEED_SETUP="
if not exist ".venv\Scripts\python.exe" set "NEED_SETUP=1"
if not exist ".venv\requirements.installed" set "NEED_SETUP=1"
if not defined NEED_SETUP (
    fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul
    if errorlevel 1 set "NEED_SETUP=1"
)
if defined NEED_SETUP (
    echo Abhaengigkeiten werden installiert ...
    call "%~dp0setup.bat"
    if errorlevel 1 (
        echo Fehler: Einrichtung fehlgeschlagen.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
python src\main.py --dev %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo NoAssist wurde mit Exit-Code %EXITCODE% beendet.
    pause
)
exit /b %EXITCODE%
