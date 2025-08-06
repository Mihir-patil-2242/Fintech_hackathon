@echo off
echo Starting FinTech Platform Backend...
echo.

cd /d "C:\Desktop\final_opus2\backend"

echo Installing minimal requirements...
pip install fastapi uvicorn pydantic 2>nul

echo Starting server...
python server.py
if errorlevel 1 (
    echo Python failed, trying python3...
    python3 server.py
    if errorlevel 1 (
        echo Python not found, trying py...
        py server.py
        if errorlevel 1 (
            echo No Python interpreter found!
            echo Please install Python from https://python.org
            pause
        )
    )
)
