@echo off
echo 🔧 Fixing Frontend Issues...
echo ==============================

cd /d "%~dp0frontend"

echo 📦 Cleaning node_modules...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del package-lock.json

echo 📦 Installing fresh dependencies...
npm install

echo 🎨 Installing additional dependencies...
npm install daisyui@latest tailwindcss@latest postcss@latest autoprefixer@latest

echo ✅ Frontend fixed! Ready to run.
echo.
echo Starting frontend server...
npm run dev

pause
