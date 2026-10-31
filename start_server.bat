@echo off
title Evidence-Backed Code RAG - Studio Server
echo =====================================================================
echo  Starting Evidence-Backed Local Code RAG 2.1.0 (Zero-Trust)
echo =====================================================================
echo.

set RAG_AUTH_TOKEN=test-secret-token
set PYTHONUNBUFFERED=1
cd /d "%~dp0"

echo [1/3] Environment configured:
echo       - Auth Token: %RAG_AUTH_TOKEN%
echo       - Project Root: %CD%
echo.

echo [2/3] Checking Ollama Local LLM status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] Ollama is running on http://localhost:11434
) else (
    echo       [NOTE] Ollama is not running.
    echo              Start Ollama in another terminal if you want LLM responses:
    echo              ollama run qwen2.5-coder:1.5b
)
echo.

echo [3/3] Launching FastAPI Web Server...
echo       - Manager Dashboard: http://localhost:8000/manager
echo       - Developer Chat:    http://localhost:8000/chat
echo       - Swagger OpenAPI:   http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server at any time.
echo =====================================================================
echo.

"C:\Users\RECHIVARTHINI SK\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
