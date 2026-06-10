@echo off
REM Customer Churn ML Project - Installation Script for Windows

echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║   Customer Churn Prediction - Setup & Installation            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Step 1: Navigate to project directory
echo [STEP 1/5] Navigating to project directory...
cd /d "C:\Users\gfx10\AppData\Local\Temp\opencode\customer_churn_ml"
if errorlevel 1 (
    echo ERROR: Could not navigate to project directory
    pause
    exit /b 1
)
echo OK: Project directory found
echo.

REM Step 2: Check Python version
echo [STEP 2/5] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo OK: Python is installed
echo.

REM Step 3: Create virtual environment
echo [STEP 3/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo OK: Virtual environment created
) else (
    echo OK: Virtual environment already exists
)
echo.

REM Step 4: Activate virtual environment and upgrade pip
echo [STEP 4/5] Activating virtual environment and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo WARNING: Could not upgrade pip, continuing anyway
)
echo OK: Virtual environment activated
echo.

REM Step 5: Install dependencies
echo [STEP 5/5] Installing dependencies...
echo This may take 2-5 minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
)
echo OK: All dependencies installed
echo.

echo ╔════════════════════════════════════════════════════════════════╗
echo ║   INSTALLATION COMPLETE!                                     ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Next steps:
echo   1. Verify installation: python setup_check.py
echo   2. Train model: python train_pipeline.py
echo   3. Run web app: streamlit run app/streamlit_app.py
echo.
pause
