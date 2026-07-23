@echo off
REM Arranca la API en http://127.0.0.1:6024
cd /d "%~dp0backend"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] No existe el entorno virtual en backend\venv
    echo Crealo con:  python -m venv venv
    echo Luego:       venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist ".env" (
    echo [AVISO] No existe backend\.env — copia .env.example y agrega tu HF_TOKEN.
    echo.
)

echo Iniciando audio_track backend en http://127.0.0.1:6024
uvicorn app.main:app --host 127.0.0.1 --port 6024 --reload
