#!/usr/bin/env powershell
# Backend startup script for Windows PowerShell

$ErrorActionPreference = "Stop"
$BACKEND_DIR = $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Sprint Analysis Backend Starter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Host "[X] Python not found! Please install Python 3.9+" -ForegroundColor Red
    exit 1
}
$PYTHON = $pythonCmd.Source
Write-Host "[1/4] Python: $PYTHON" -ForegroundColor Green

# Check dependencies
Write-Host "[2/4] Checking dependencies..." -ForegroundColor Gray
Set-Location $BACKEND_DIR

& $PYTHON -c "import fastapi, uvicorn, cv2, mediapipe" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    Installing dependencies..." -ForegroundColor Yellow
    & $PYTHON -m pip install -r requirements.txt
}
Write-Host "    Dependencies OK" -ForegroundColor Green

# Check model
Write-Host "[3/4] Checking MediaPipe model..." -ForegroundColor Gray
$MODEL_FILE = "$BACKEND_DIR\models\pose_landmarker_full.task"
if (-not (Test-Path $MODEL_FILE)) {
    Write-Host "    Downloading model..." -ForegroundColor Yellow
    & $PYTHON scripts\download_model.py
}
Write-Host "    Model OK" -ForegroundColor Green

# Check .env
Write-Host "[4/4] Checking configuration..." -ForegroundColor Gray
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
    }
}
Write-Host "    Configuration OK" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Starting Uvicorn Server..." -ForegroundColor Green
Write-Host "  URL: http://localhost:8000" -ForegroundColor Green
Write-Host "  Health: http://localhost:8000/health" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Start server
& $PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
