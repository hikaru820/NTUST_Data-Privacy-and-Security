# Federated Learning (Step 1)

CIFAR-100 上的 Flower + PyTorch Federated Learning 系統，作為期末專題 Step 1。
Step 2 (Gradient Leakage Attack) 與 Step 3 (Differential Privacy Defense) 會直接重用本資料夾的 **model.py / data.py** 與訓練後的 **global_model.pth**。

---

## 📁 目錄結構

```
FL/
├── README.md                  ← 你正在看
├── .venv/                     ← Python 虛擬環境（執行 install 腳本後產生，不進 git）
├── install_amd.bat            ← AMD GPU (DirectML) 安裝
├── install_nvidia.bat         ← NVIDIA GPU (CUDA 12.1) 安裝
├── install_cpu.bat            ← CPU 安裝
├── install_mac.command        ← macOS 安裝 (Apple Silicon MPS / Intel CPU)
├── run_all.bat                ← 一鍵啟動 server + N 個 client
├── checkpoints/               ← 訓練完成的 global model 存這（自動建立）
│   └── global_model.pth
└── Code/
    ├── device_utils.py        ← 跨平台 device 抽象（CUDA / DirectML / MPS / CPU）
    ├── check_gpu.py           ← 環境驗證腳本
    ├── data.py                ← CIFAR-100 載入 + IID/Non-IID 切分
    ├── model.py               ← CNN 架構定義
    ├── client.py              ← Flower client（本地訓練 + 評估）
    ├── server.py              ← Flower server + FedAvg 策略
    └── data/                  ← CIFAR-100 dataset cache（自動下載，不進 git）
```

---

## ⚠️ 重要：執行指令時的 cwd（current working directory）

本專案有 **三個關鍵 cwd**，搞錯會找不到檔案：

| 從哪裡執行 | 對應指令 |
|---|---|
| **`FL/`** | install 腳本、`run_all.bat`、整個流程一鍵啟動 |
| **`FL/`**（且 venv 已 activate） | 啟用 venv：`.venv\Scripts\activate` |
| **`FL/Code/`**（且 venv 已 activate） | 跑單一 Python 檔：`python server.py`、`python client.py`、`python check_gpu.py` |

**判斷自己在哪：** PowerShell prompt 會顯示完整路徑，例如：
```
(.venv) PS C:\Users\...\FL\Code>
```
`(.venv)` = venv 啟用；`FL\Code` = cwd 在 Code/。

---

## 🚀 Quick Start

### 1. 安裝（一次性）

從 `FL/` 執行對應平台的安裝腳本：

| 平台 | 指令（cwd = `FL/`） |
|---|---|
| Windows + AMD / Intel Arc | `.\install_amd.bat` |
| Windows + NVIDIA | `.\install_nvidia.bat` |
| Windows / Linux 無 GPU | `.\install_cpu.bat` |
| macOS (Apple Silicon 或 Intel) | `chmod +x install_mac.command && ./install_mac.command` |

腳本會：
1. 建 `.venv/` 虛擬環境
2. 裝 PyTorch（對應平台版本）
3. 裝 Flower + 共用套件
4. 約需 3~10 分鐘（看網速）

### 2. 啟用環境（每次開新終端機都要做）

```powershell
cd C:\path\to\FL
.venv\Scripts\activate
```
prompt 出現 `(.venv)` 才算成功。

### 3. 驗證 GPU

```powershell
cd Code
python check_gpu.py
```
應該看到你的 GPU 被偵測到，並印出 matmul 速度。

### 4. 一鍵跑訓練

```powershell
cd ..              # 回到 FL/
.\run_all.bat      # 預設 5 clients、10 rounds、IID
```

---

## 🎛️ `run_all.bat` 用法

從 `FL/` 執行：
```powershell
.\run_all.bat [num_clients] [rounds] [mode] [alpha]
```

| 範例 | 效果 |
|---|---|
| `.\run_all.bat` | 5 clients、10 rounds、IID（預設） |
| `.\run_all.bat 3` | 3 clients、10 rounds、IID |
| `.\run_all.bat 3 5` | 3 clients、5 rounds、IID |
| `.\run_all.bat 5 20 noniid` | 5 clients、20 rounds、Non-IID（α=0.5） |
| `.\run_all.bat 5 20 noniid 0.1` | Non-IID 強異質（α=0.1） |

腳本會自動：
- 開 1 個 server 視窗
- 開 N 個 client 視窗（每個自動 activate venv + cd Code）
- 預先下載 CIFAR-100 避免重複下載

---

## 如果無法使用.bat
## 🛠️ 手動執行（單獨開 server / client）

每個視窗都要先：
```powershell
cd C:\path\to\FL
.venv\Scripts\activate
cd Code
```

**Server：**
```powershell
python server.py --rounds 10 --min-clients 5 --server-eval
```

**Client（每個 client 開一個視窗，client-id 從 0 開始）：**
```powershell
python client.py --client-id 0 --num-clients 5 --mode iid
python client.py --client-id 1 --num-clients 5 --mode iid
# ...以此類推
```

---

## 🔀 切換運算裝置

`device_utils.py` 自動依優先順序選擇：**CUDA → DirectML → MPS → CPU**

### 強制指定（兩種方式）

**方法 1：環境變數**（推薦給組員使用）

```powershell
# Windows PowerShell
$env:FL_DEVICE = "cpu"
python client.py --client-id 0 --num-clients 2

# Mac / Linux
export FL_DEVICE=mps
python client.py --client-id 0 --num-clients 2
```

支援值：`auto`（預設）、`cuda`、`dml`、`mps`、`cpu`

**方法 2：程式內呼叫**
```python
from device_utils import get_device
device = get_device(prefer="cpu")
```

---

## 在實作後面的step 2, 3時記得在github開新的branch!!!非常重要!!!!

## 📦 Step 1 訓練結果：怎麼給 Step 2 / Step 3 用

訓練完成後，最終 global model 會存在 `FL/checkpoints/global_model.pth`。

### 給 Step 2 (Gradient Leakage Attack) 組員

#### 你需要的素材

| 素材 | 來源 | 必要性 |
|---|---|---|
| 同一份 model 架構 | `FL/Code/model.py` import | 必要 |
| 同一份 dataset / transforms | `FL/Code/data.py` import | 必要 |
| 訓練好的 global model (任選一個) | `FL/checkpoints/model_*.pth` | 可選（建議用） |
| **要攻擊的梯度** | **自己產生**（見下方範例） | 必要 |

**重要觀念：** Step 1 不會直接給你「梯度」檔案，因為 FL 訓練時梯度只存在於 client 進程記憶體中。攻擊情境是「**假設**你攔截到某個 client 在某個訓練狀態下對某張圖計算的梯度」——這個情境**由你自己組裝**。

#### 完整範例：載入 model + 產生攻擊用梯度

```python
import sys
import torch
import torch.nn as nn

# 1. import Step 1 的程式碼
sys.path.insert(0, "../FL/Code")  # 路徑依你的資料夾位置調整
from model import create_model
from data import load_cifar100

# 2. 建立同一個 model 架構（必須跟 Step 1 一致，否則 state_dict 對不上）
model = create_model(name="cifar100_cnn", num_classes=100)

# 3. 載入訓練好的權重（可選；不載入就是用未訓練 model 攻擊）
state_dict = torch.load("../FL/checkpoints/model_iid_5c_20r.pth",
                        map_location="cpu")
model.load_state_dict(state_dict)
model.eval()  # 注意：eval 模式關閉 BN 訓練統計

# 4. 取一張受害者資料（從 trainset 抽，模擬 client 持有的資料）
trainset, _ = load_cifar100(data_dir="../FL/Code/data")
victim_x, victim_y = trainset[0]   # 第 0 張圖 + label
victim_x = victim_x.unsqueeze(0)   # 加 batch 維度 -> (1, 3, 32, 32)
victim_y = torch.tensor([victim_y])

# 5. 計算該樣本的梯度（這就是「攔截到」的東西）
criterion = nn.CrossEntropyLoss()
pred = model(victim_x)
loss = criterion(pred, victim_y)
intercepted_grad = torch.autograd.grad(loss, model.parameters())
# intercepted_grad 是 tuple of tensors，跟 model.parameters() 一一對應

# 6. 開始攻擊
#    目標：給定 (model, intercepted_grad)，重建出 victim_x 和 victim_y
#    參考 DLG / iDLG / Inverting Gradients 論文
```

#### 建議的對比實驗（給 Step 2 報告用）

可以製作一個對照表，看哪種情境下攻擊比較容易/困難：

| 情境 | model 來源 | 預期攻擊難度 |
|---|---|---|
| A | 未訓練（剛初始化） | 容易，DLG 原始論文 baseline |
| B | `model_iid_5c_20r.pth` (訓練到一半) | 中等，貼近真實 FL 場景 |
| C | 同 B 但用 `eval()` 改成 `train()`（BN 用 batch 統計） | 困難 |
| D | Non-IID `model_noniid_5c_20r_a0.1.pth` | 可能不同（client 資料分佈很偏） |

#### ⚠️ 注意事項

- **不要自己重寫 model 架構**，一定要 `from model import create_model`，否則 state_dict 對不上
- **`model.eval()` vs `model.train()`** 會影響 BatchNorm 行為，攻擊結果會差很多，固定一種方便對比
- 攻擊用的 batch size 越小越容易成功（原 DLG 論文用 single sample）
- CIFAR-100 圖太小（32×32）反而比 ImageNet 更容易攻擊，這是好事

### 給 Step 3 (Differential Privacy) 組員

⚠️ **先看「DP 環境注意事項」（下一節）**，有兩個一定會踩的雷。

**最乾淨的接法：** 在 `Code/` 旁邊建一個 `client_dp.py`，**只覆寫 `make_optimizer()`**（Step 1 已經把這個 factory 抽出來了），其他 import 全部沿用：

```python
# Code/client_dp.py
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch.optim as optim

# 沿用 Step 1 的所有東西
from client import (
    FlowerClient, train, evaluate,
    get_parameters, set_parameters,
    make_criterion,
)
from model import create_model

# 全域 PrivacyEngine（讓 make_optimizer 能拿到）
_privacy_engine = PrivacyEngine()
_dml_state = {}   # 暫存 model/trainloader 被 PrivacyEngine 改造後的引用

def make_optimizer(model, lr, momentum=0.9):
    """Override Step 1 的 make_optimizer。"""
    base_opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # 注意：PrivacyEngine.make_private 也會包 model 跟 dataloader，
    # 所以這裡的整合方式要看 opacus 文件選一個（簡化範例略）
    return base_opt

# 之後直接 monkey-patch：
import client
client.make_optimizer = make_optimizer
```

**另一種接法（server-side DP）：** 改 `server.py` 的 strategy，在 `aggregate_fit` 後對聚合結果加雜訊。不用 opacus，純 NumPy 操作。這個方式繞過 opacus 的限制，AMD GPU 也能跑。詳見 Step 3 paper 的 distributed_dp。

### ⚠️ DP 環境注意事項（重要！）

**雷 1：BatchNorm 不支援**

我們的 `model.py` 用了 BatchNorm，但 opacus DP-SGD 不支援。建 model 後**第一件事**：
```python
from opacus.validators import ModuleValidator
model = create_model()
model = ModuleValidator.fix(model)   # 自動把 BN -> GroupNorm
```
這會讓 baseline accuracy 掉 2~5%，屬正常現象。

**雷 2：opacus 不支援 DirectML**

opacus 內部用 `functorch.vmap` 計算 per-sample gradients，DirectML backend 不支援。AMD GPU 用戶有三條路：
1. **改用 CPU 跑**：慢但能跑（`$env:FL_DEVICE = "cpu"`）
2. **改用 server-side DP**：不需要 opacus
3. **換 CUDA 機器或 Google Colab**

NVIDIA GPU 不受影響。

**推薦超參數起點（CIFAR-100）：**
- `noise_multiplier = 1.0`（雜訊強度）
- `max_grad_norm = 1.0`（梯度裁切上限）
- `target_epsilon = 8.0`（最終隱私預算）
- 從 epsilon 大（弱隱私）→ epsilon 小（強隱私）做 trade-off 曲線

### 共用 testset 評估

不論是 Step 2 還是 Step 3，都可以直接 import 同一份 testset：
```python
from data import load_cifar100
_, testset = load_cifar100(data_dir="../FL/Code/data")
```

---

## 🆘 Troubleshooting

| 症狀 | 原因 | 解法 |
|---|---|---|
| `'python' is not recognized` | 沒裝 Python 或沒加 PATH | 從 [python.org](https://www.python.org/) 重裝，勾「Add to PATH」 |
| `Python was not found` + Microsoft Store 提示 | 系統 python 是 MS Store 空殼 | 同上，裝真的 Python |
| import 報 DLL 錯誤 | 沒 activate venv，用到系統 python | 跑 `.venv\Scripts\activate`，prompt 要有 `(.venv)` |
| `ModuleNotFoundError: device_utils` | cwd 不在 `FL/Code` | `cd Code` 再跑 |
| `Address already in use` (port 8080) | 之前的 server 沒關乾淨 | `taskkill /F /IM python.exe`（會砍所有 python 進程） |
| Client 顯示 OOM | GPU 記憶體不夠 | 啟動時加 `--batch-size 16` 或更小 |
| Accuracy 一直 ~1% 不動 | random level，可能 lr 或資料切分有問題 | 檢查 `fit_config` 的 lr，或先用 IID 跑通 |
| Server 卡住沒進度 | client 數不到 `min_clients` | 確認所有 client 視窗都跑起來 |
| DirectML 跳特定 op 不支援 | DML 對某些 op 支援不全 | 該 client 改用 CPU：`$env:FL_DEVICE = "cpu"` |

---

## 📊 預期結果

CIFAR-100、5 clients、20 rounds、IID：
- 預期 server-side accuracy 約 **40~55%**
- 每 round 約 30 秒 ~ 2 分鐘（看 GPU）
- 總時長約 10~40 分鐘

Non-IID（α=0.1）會明顯偏低，可能只到 25~35%，這正是 FL 的核心挑戰之一。

--