@echo off
echo 🎨 Starting FinTech AI Platform Frontend...
echo =========================================

cd /d "%~dp0frontend"

echo 📱 Starting React development server...
echo Frontend will be available at: http://localhost:3000
echo.

npm run dev
pause
