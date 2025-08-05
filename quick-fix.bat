@echo off
echo 🔧 Quick Frontend Fix - Removing TailwindCSS Issues
echo ==================================================

cd /d "%~dp0frontend"

echo 📦 Stopping dev server if running...
taskkill /f /im node.exe 2>nul

echo 🧹 Removing problematic packages...
npm uninstall tailwindcss postcss autoprefixer daisyui

echo 📦 Installing minimal dependencies...
npm install

echo ✅ Fixed! Starting simple version without TailwindCSS...
echo.
echo The frontend will now work with basic styling.
echo Once confirmed working, we can add TailwindCSS back properly.
echo.

npm run dev

pause
