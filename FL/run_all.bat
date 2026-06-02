@echo off
REM ============================================================
REM  run_all.bat
REM  Launch Flower server + N clients in separate windows.
REM
REM  Usage examples:
REM    run_all.bat                  -> 5 clients, 10 rounds, iid (defaults)
REM    run_all.bat 3                -> 3 clients
REM    run_all.bat 3 5              -> 3 clients, 5 rounds
REM    run_all.bat 3 5 noniid       -> 3 clients, 5 rounds, non-iid (alpha=0.5)
REM    run_all.bat 3 5 noniid 0.1   -> alpha=0.1 (strong heterogeneity)
REM
REM  All windows stay open after exit so you can read logs.
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- Defaults ---
set NUM_CLIENTS=5
set ROUNDS=10
set MODE=iid
set ALPHA=0.5
set BATCH_SIZE=32

REM --- Override from CLI args ---
if not "%~1"=="" set NUM_CLIENTS=%~1
if not "%~2"=="" set ROUNDS=%~2
if not "%~3"=="" set MODE=%~3
if not "%~4"=="" set ALPHA=%~4

echo ============================================================
echo  Launching Federated Learning
echo    num_clients = %NUM_CLIENTS%
echo    rounds      = %ROUNDS%
echo    mode        = %MODE%
echo    alpha       = %ALPHA%   (only used when mode=noniid)
echo    batch_size  = %BATCH_SIZE%
echo ============================================================
echo.

REM --- Verify venv exists ---
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found at %CD%\.venv
    echo Run install_amd.bat / install_nvidia.bat / install_cpu.bat first.
    pause
    exit /b 1
)

REM --- Pre-download CIFAR-100 once to avoid race conditions ---
echo [PREP] Ensuring CIFAR-100 is downloaded (skipped if cached)...
call .venv\Scripts\activate.bat
python -c "from torchvision.datasets import CIFAR100; CIFAR100(root='Code/data', train=True, download=True); CIFAR100(root='Code/data', train=False, download=True)" 1>nul
if errorlevel 1 (
    echo [ERROR] Failed to prepare CIFAR-100. Check that torchvision is installed.
    pause
    exit /b 1
)
echo [PREP] CIFAR-100 ready.
echo.

REM --- Build save-path with experiment config encoded in filename ---
REM   IID mode example:    model_iid_5c_20r.pth
REM   Non-IID mode example: model_noniid_5c_20r_a0.1.pth
if /i "%MODE%"=="iid" (
    set SAVE_PATH=../checkpoints/model_%MODE%_%NUM_CLIENTS%c_%ROUNDS%r.pth
) else (
    set SAVE_PATH=../checkpoints/model_%MODE%_%NUM_CLIENTS%c_%ROUNDS%r_a%ALPHA%.pth
)

REM --- Launch server in a new window ---
echo [LAUNCH] Server (rounds=%ROUNDS%, min_clients=%NUM_CLIENTS%) ...
echo          save_path = !SAVE_PATH!
start "FL Server" cmd /k "call .venv\Scripts\activate.bat && cd Code && python server.py --rounds %ROUNDS% --min-clients %NUM_CLIENTS% --server-eval --save-path !SAVE_PATH!"

REM --- Wait for server to bind port ---
echo Waiting 4s for server to start ...
timeout /t 4 /nobreak >nul

REM --- Launch clients ---
set /a LAST=%NUM_CLIENTS%-1
for /l %%i in (0,1,!LAST!) do (
    echo [LAUNCH] Client %%i ...
    start "FL Client %%i" cmd /k "call .venv\Scripts\activate.bat && cd Code && python client.py --client-id %%i --num-clients %NUM_CLIENTS% --mode %MODE% --alpha %ALPHA% --batch-size %BATCH_SIZE%"
    timeout /t 1 /nobreak >nul
)

echo.
echo ============================================================
echo  All %NUM_CLIENTS% clients + 1 server launched.
echo  Watch the "FL Server" window for training progress.
echo  Close windows manually when training finishes.
echo ============================================================
echo.
pause
