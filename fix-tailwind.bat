@echo off
echo 🔧 Fixing TailwindCSS PostCSS Issue...
echo =====================================

cd /d "%~dp0frontend"

echo 📦 Installing correct TailwindCSS version...
npm uninstall tailwindcss postcss autoprefixer
npm install -D tailwindcss@latest postcss@latest autoprefixer@latest
npm install daisyui@latest

echo 🎨 Initializing TailwindCSS...
npx tailwindcss init -p

echo ✅ TailwindCSS fixed! 
echo.
echo Restarting development server...
npm run dev

pause
