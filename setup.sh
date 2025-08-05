#!/bin/bash

echo "🏦 FinTech AI Platform - Hackathon Setup"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Backend Setup
echo "🔧 Setting up Backend..."
cd backend

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Backend setup complete!"
echo ""

# Frontend Setup
echo "🎨 Setting up Frontend..."
cd ../frontend

echo "📦 Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"
echo ""

echo "🚀 Setup Complete! Ready to run the platform."
echo ""
echo "To start the platform:"
echo "1. Terminal 1: cd backend && python main.py"
echo "2. Terminal 2: cd frontend && npm run dev"
echo ""
echo "Then visit:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
