@echo off
REM Arranca la interfaz en http://localhost:6023
cd /d "%~dp0frontend"

if not exist "node_modules" (
    echo Instalando dependencias de npm por primera vez...
    call npm install
    if errorlevel 1 (
        echo [ERROR] Fallo npm install
        pause
        exit /b 1
    )
)

echo Iniciando audio_track frontend en http://localhost:6023
call npm run dev
