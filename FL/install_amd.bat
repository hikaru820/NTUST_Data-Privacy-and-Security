@echo off
REM ============================================================
REM  AMD GPU (DirectML) installer
REM  For: Windows + AMD Radeon / Intel Arc / any DirectX 12 GPU
REM ============================================================

cd /d "%~dp0"
echo.
echo === FL Environment Setup: AMD / DirectML ===
echo Working directory: %CD%
echo.

REM --- 1. Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.10 or 3.11 from python.org
    pause
    exit /b 1
)

REM --- 2. Create venv if not exists ---
if not exist ".venv\" (
    echo [STEP 1/4] Creating virtual environment .venv ...
    python -m venv .venv --without-pip
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [STEP 2.5/4] Installing internal pip ...
    python -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo [STEP 1/4] .venv already exists, skipping creation.
)

REM --- 3. Activate venv ---
echo [STEP 2/4] Activating virtual environment ...
call .venv\Scripts\activate.bat

REM --- 4. Upgrade pip ---
echo [STEP 3/4] Upgrading pip ...
python -m pip install --upgrade pip

REM --- 5. Install PyTorch (DirectML) + common packages ---
echo [STEP 4/4] Installing torch-directml and dependencies ...
echo This will download about 2GB, please wait.
echo.

pip install torch-directml torchvision
if errorlevel 1 (
    echo [ERROR] torch-directml installation failed.
    pause
    exit /b 1
)

pip install "flwr[simulation]==1.13.1" numpy matplotlib tqdm Pillow
if errorlevel 1 (
    echo [ERROR] Common package installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Installation complete!
echo ------------------------------------------------------------
echo  Next steps:
echo    1. Open a new terminal in this folder
echo    2. Run: .venv\Scripts\activate
echo    3. Run: python check_gpu.py
echo ============================================================
echo.
pause
