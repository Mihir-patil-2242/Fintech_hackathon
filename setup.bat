@echo off
echo 🏦 FinTech AI Platform - Hackathon Setup
echo ======================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.10+ first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.

REM Backend Setup
echo 🔧 Setting up Backend...
cd backend

echo 📦 Installing Python dependencies...
pip install -r requirements.txt

echo ✅ Backend setup complete!
echo.

REM Frontend Setup
echo 🎨 Setting up Frontend...
cd ..\frontend

echo 📦 Installing Node.js dependencies...
npm install

echo ✅ Frontend setup complete!
echo.

echo 🚀 Setup Complete! Ready to run the platform.
echo.
echo To start the platform:
echo 1. Terminal 1: cd backend ^&^& python main.py
echo 2. Terminal 2: cd frontend ^&^& npm run dev
echo.
echo Then visit:
echo - Frontend: http://localhost:3000
echo - Backend API: http://localhost:8000
echo - API Docs: http://localhost:8000/docs
echo.
pause
