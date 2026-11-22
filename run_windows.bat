@echo off
REM RajaswaMitra - one-click start on Windows
cd /d %~dp0
title RajaswaMitra
if "%RM_MODEL%"=="" set RM_MODEL=qwen2.5:1.5b
set RM_PROVIDER=ollama

where python >nul 2>nul || (echo Python not found. Install Python 3.11+ from python.org and tick "Add Python to PATH". & pause & exit /b 1)

where ollama >nul 2>nul && (
  ollama list >nul 2>nul || (echo Starting Ollama... & start "" /min ollama serve & timeout /t 6 >nul)
  ollama list | findstr /i /c:"%RM_MODEL%" >nul || (echo Downloading model %RM_MODEL% - first time only... & ollama pull %RM_MODEL%)
) || (echo Ollama not found - the app will open; choose Demo mode or install Ollama from ollama.com)

if not exist .venv (
  echo Creating virtual environment and installing packages - first time only...
  python -m venv .venv
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
echo.
echo  RajaswaMitra is starting. Open http://localhost:8501
echo  Colleagues on the office network:  http://%COMPUTERNAME%:8501
echo  Keep this window open while using the app.
echo.
start "" http://localhost:8501
streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false
pause
