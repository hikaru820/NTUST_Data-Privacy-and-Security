#!/bin/bash
# ============================================================
#  macOS 安裝腳本
#  Apple Silicon (M1/M2/M3/M4) → 自動支援 MPS (Metal) GPU 加速
#  Intel Mac → CPU only (macOS 不再支援 CUDA)
# ============================================================

cd "$(dirname "$0")"
echo ""
echo "=== FL Environment Setup: macOS ==="
echo "Working directory: $(pwd)"
echo ""

# --- 偵測架構 ---
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "Detected: Apple Silicon ($ARCH) — MPS GPU will be available"
else
    echo "Detected: Intel Mac ($ARCH) — CPU only"
fi
echo ""

# --- 1. 檢查 Python ---
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found."
    echo "Install via: brew install python@3.11"
    echo "Or download from: https://www.python.org/"
    read -p "Press Enter to exit..."
    exit 1
fi

PY_VERSION=$(python3 --version)
echo "Python: $PY_VERSION"

# --- 2. 建立 venv ---
if [ ! -d ".venv" ]; then
    echo "[STEP 1/4] Creating virtual environment .venv ..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create venv."
        read -p "Press Enter to exit..."
        exit 1
    fi
else
    echo "[STEP 1/4] .venv already exists, skipping creation."
fi

# --- 3. 啟動 venv ---
echo "[STEP 2/4] Activating virtual environment ..."
source .venv/bin/activate

# --- 4. 升級 pip ---
echo "[STEP 3/4] Upgrading pip ..."
python -m pip install --upgrade pip

# --- 5. 安裝 PyTorch + 共用套件 ---
# macOS 用預設 pip index 即可：
#   - Apple Silicon 會抓到支援 MPS 的 arm64 build
#   - Intel Mac 會抓到 x86_64 CPU build
echo "[STEP 4/4] Installing torch + dependencies ..."
echo ""

pip install torch torchvision
if [ $? -ne 0 ]; then
    echo "[ERROR] torch installation failed."
    read -p "Press Enter to exit..."
    exit 1
fi

pip install "flwr[simulation]==1.13.1" numpy matplotlib tqdm Pillow
if [ $? -ne 0 ]; then
    echo "[ERROR] Common package installation failed."
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "------------------------------------------------------------"
echo "  Next steps:"
echo "    1. Open a new terminal in this folder"
echo "    2. Run: source .venv/bin/activate"
echo "    3. Run: python check_gpu.py"
echo "============================================================"
echo ""
read -p "Press Enter to close..."
