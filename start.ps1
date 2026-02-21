# Multi-Model Cat Breed Classification System - Startup Script
# This script starts both backend and frontend servers

Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host "🐱 Kedi Cinsi Tahmin Sistemi - Multi-Model"
Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host ""

# Check if Python is available
Write-Host "🔍 Checking requirements..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Check if Node.js is available
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found! Please install Node.js 16+" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Ask which service to start
Write-Host "🚀 What would you like to start?" -ForegroundColor Yellow
Write-Host "1. Backend only (FastAPI)"
Write-Host "2. Frontend only (React + Vite)"
Write-Host "3. Both (Backend + Frontend)"
Write-Host "4. Run tests"
Write-Host ""
$choice = Read-Host "Enter your choice (1-4)"

function Start-Backend {
    Write-Host ""
    Write-Host "🔧 Starting Backend..." -ForegroundColor Cyan
    Write-Host "   API will be available at: http://localhost:8000" -ForegroundColor Gray
    Write-Host "   Swagger docs at: http://localhost:8000/docs" -ForegroundColor Gray
    Write-Host ""
    
    # Check if virtual environment exists
    if (Test-Path "venv\Scripts\activate.ps1") {
        Write-Host "📦 Activating virtual environment..." -ForegroundColor Gray
        & "venv\Scripts\activate.ps1"
    } else {
        Write-Host "⚠️  Virtual environment not found. Using global Python." -ForegroundColor Yellow
    }
    
    # Start uvicorn
    python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
}

function Start-Frontend {
    Write-Host ""
    Write-Host "🎨 Starting Frontend..." -ForegroundColor Cyan
    Write-Host "   UI will be available at: http://localhost:5173" -ForegroundColor Gray
    Write-Host ""
    
    # Check if node_modules exists
    if (-not (Test-Path "frontend\node_modules")) {
        Write-Host "📦 Installing npm dependencies..." -ForegroundColor Yellow
        Set-Location frontend
        npm install
        Set-Location ..
    }
    
    # Start Vite dev server
    Set-Location frontend
    npm run dev
}

function Run-Tests {
    Write-Host ""
    Write-Host "🧪 Running tests..." -ForegroundColor Cyan
    Write-Host "   Make sure backend is running first!" -ForegroundColor Yellow
    Write-Host ""
    
    Start-Sleep -Seconds 2
    python test_multi_model.py
}

switch ($choice) {
    "1" {
        Start-Backend
    }
    "2" {
        Start-Frontend
    }
    "3" {
        Write-Host ""
        Write-Host "🚀 Starting both services..." -ForegroundColor Cyan
        Write-Host "   Backend will start first, then frontend in a new window" -ForegroundColor Gray
        Write-Host ""
        
        # Start backend in current window
        Write-Host "Starting backend in 3 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        
        # Start frontend in new window
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🎨 Starting Frontend...' -ForegroundColor Cyan; cd frontend; npm run dev"
        
        # Start backend in current window
        Start-Backend
    }
    "4" {
        Run-Tests
    }
    default {
        Write-Host "❌ Invalid choice. Please run the script again and select 1-4." -ForegroundColor Red
        exit 1
    }
}
