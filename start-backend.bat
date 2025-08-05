@echo off
echo 🚀 Starting FinTech AI Platform Backend...
echo ========================================

cd /d "%~dp0backend"

echo 📊 Starting FastAPI server...
echo Backend will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo.

python main.py
pause
