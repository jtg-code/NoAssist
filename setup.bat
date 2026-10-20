@echo off
rem Richtet die virtuelle Umgebung ein und installiert die Abhaengigkeiten.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Lege virtuelle Umgebung .venv an ...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo Fehler: .venv konnte nicht angelegt werden. Ist Python installiert?
        exit /b 1
    )
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

rem Merkt sich den installierten Stand; start.cmd installiert nur bei Aenderungen neu.
copy /y requirements.txt ".venv\requirements.installed" >nul
exit /b 0
