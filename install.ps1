# Customer Churn ML Project - Installation Script for Windows (PowerShell)

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Customer Churn Prediction - Setup & Installation            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Navigate to project directory
Write-Host "[STEP 1/5] Navigating to project directory..." -ForegroundColor Yellow
$projectDir = "C:\Users\gfx10\AppData\Local\Temp\opencode\customer_churn_ml"

if (Test-Path $projectDir) {
    Set-Location $projectDir
    Write-Host "OK: Project directory found" -ForegroundColor Green
} else {
    Write-Host "ERROR: Project directory not found at $projectDir" -ForegroundColor Red
    pause
    exit 1
}
Write-Host ""

# Step 2: Check Python version
Write-Host "[STEP 2/5] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($?) {
    Write-Host "OK: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    pause
    exit 1
}
Write-Host ""

# Step 3: Create virtual environment
Write-Host "[STEP 3/5] Creating/verifying virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "OK: Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "OK: Virtual environment created" -ForegroundColor Green
}
Write-Host ""

# Step 4: Activate virtual environment and upgrade pip
Write-Host "[STEP 4/5] Activating virtual environment and upgrading pip..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel 2>&1 | Out-Null
Write-Host "OK: Virtual environment activated and pip upgraded" -ForegroundColor Green
Write-Host ""

# Step 5: Install dependencies
Write-Host "[STEP 5/5] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take 2-5 minutes..." -ForegroundColor Cyan

pip install -r requirements.txt
if ($?) {
    Write-Host "OK: All dependencies installed" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "Try running manually: pip install -r requirements.txt" -ForegroundColor Yellow
    pause
    exit 1
}
Write-Host ""

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   INSTALLATION COMPLETE!                                     ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Verify installation: python setup_check.py" -ForegroundColor White
Write-Host "  2. Train model: python train_pipeline.py" -ForegroundColor White
Write-Host "  3. Run web app: streamlit run app/streamlit_app.py" -ForegroundColor White
Write-Host ""

pause
