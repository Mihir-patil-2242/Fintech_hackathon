@echo off
title FinTech Platform Launcher
color 0A

echo.
echo  _____ _       _____         _       ______ _       _    __                      
echo ^|  ___^| ^|     ^|_   _^|       ^| ^|      ^| ___ \ ^|     ^| ^|  / _^|                     
echo ^| ^|__ ^| ^|       ^| ^|    ___  ^| ^|__    ^| ^|_/ / ^|  __ ^| ^| ^| ^|_   ___   _ __  _ __ ___  
echo ^|  __^|^| ^|       ^| ^|   / _ \ ^| '_ \   ^|  __/^| ^| / _` ^| ^| ^|  _^| / _ \ ^| '__^|^| '_ ` _ \ 
echo ^| ^|   ^| ^|      ^| ^|  ^|  __/ ^| ^| ^| ^|  ^| ^|   ^| ^|^| (_^| ^| ^| ^| ^|  ^|  __/ ^| ^|   ^| ^| ^| ^| ^| ^|
echo \_^|   ^|_^|      \_^/   \___^| ^|_^| ^|_^|  \_^|   ^|_^| \__,_^|_^| \_^|   \___^| ^|_^|   ^|_^| ^|_^| ^|_^|
echo.
echo                               v2.0 - DeFi Platform
echo.

echo [1] Start Backend Only
echo [2] Start Frontend Only  
echo [3] Start Both (Recommended)
echo [4] Quick Setup (Install deps + Start both)
echo [5] Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto backend
if "%choice%"=="2" goto frontend
if "%choice%"=="3" goto both
if "%choice%"=="4" goto setup
if "%choice%"=="5" goto exit

:backend
echo.
echo Starting Backend...
cd /d "C:\Desktop\final_opus2\backend"
echo Installing minimal requirements...
pip install fastapi uvicorn pydantic --quiet
echo.
echo Backend starting at http://localhost:8000
echo API Docs available at http://localhost:8000/docs
echo.
python server.py
goto end

:frontend
echo.
echo Starting Frontend...
cd /d "C:\Desktop\final_opus2\frontend"
echo.
echo Frontend will start at http://localhost:5173
echo.
npm run dev
goto end

:both
echo.
echo Starting Backend in background...
cd /d "C:\Desktop\final_opus2\backend"
pip install fastapi uvicorn pydantic --quiet >nul 2>&1
start /min cmd /c "python server.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo Starting Frontend...
cd /d "C:\Desktop\final_opus2\frontend"
echo.
echo ================================================================
echo  Backend:  http://localhost:8000     (API Docs: /docs)
echo  Frontend: http://localhost:5173     (Main App)
echo ================================================================
echo.
npm run dev
goto end

:setup
echo.
echo Quick Setup - Installing all dependencies...
echo.

echo [Backend] Installing Python packages...
cd /d "C:\Desktop\final_opus2\backend"
pip install fastapi uvicorn pydantic --quiet

echo [Frontend] Installing Node packages...
cd /d "C:\Desktop\final_opus2\frontend"
npm install --silent

echo.
echo Setup complete! Starting both services...
echo.

cd /d "C:\Desktop\final_opus2\backend"
start /min cmd /c "python server.py"

timeout /t 3 /nobreak >nul

cd /d "C:\Desktop\final_opus2\frontend"
echo.
echo ================================================================
echo  Backend:  http://localhost:8000     (API Docs: /docs)
echo  Frontend: http://localhost:5173     (Main App)
echo ================================================================
echo.
npm run dev
goto end

:exit
exit

:end
echo.
echo Press any key to close...
pause >nul
