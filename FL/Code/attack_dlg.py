"""
Step 2 — Gradient Leakage Attack (DLG / iDLG)

攻擊情境：
  聯邦學習中，server（或惡意第三方）可以攔截某一 client 上傳的梯度。
  本腳本模擬該攻擊：先從 victim client 計算出「洩漏梯度」，再透過
  梯度匹配最佳化，從隨機雜訊中還原出原始訓練影像與標籤。

支援兩種方法：
  - DLG  (Zhu et al., NeurIPS 2019)：同時最佳化 dummy 影像與 soft dummy 標籤
  - iDLG (Zhao et al., 2020)：先從梯度符號分析出真實標籤，再只最佳化影像
    （收斂更穩定，特別適用於 batch size = 1）

使用範例：
  python attack_dlg.py --method idlg --iters 2000 --all-clients
  python attack_dlg.py --method dlg   --iters 2000 --batch-size 4
"""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from data import load_dataset
from model import create_model


# CIFAR-100 資料集的正規化參數（訓練時使用相同值）
# 反正規化時需要用到，以便將 tensor 轉回可視化的 [0,1] 影像
CIFAR100_MEAN = torch.tensor((0.5071, 0.4867, 0.4408)).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor((0.2675, 0.2565, 0.2761)).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DLG/iDLG gradient leakage attack for the Step 1 FL model."
    )
    parser.add_argument(
        "--checkpoint",
        default="../checkpoints/global_model.pth",
        help="Step 1 訓練後的全域模型權重路徑；找不到時使用隨機初始化的模型。",
    )
    parser.add_argument("--client-id", type=int, default=0,
                        help="要攻擊的 client 編號（0 ~ num-clients-1）。")
    parser.add_argument(
        "--all-clients",
        action="store_true",
        help="依序攻擊所有 client（0 到 num-clients-1），用於批次實驗。",
    )
    parser.add_argument("--num-clients", type=int, default=5,
                        help="聯邦學習系統的 client 總數，需與 Step 1 訓練設定一致。")
    parser.add_argument("--mode", default="iid", choices=["iid", "noniid"],
                        help="資料分配方式：iid（均勻）或 noniid（Dirichlet 分配）。")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Non-IID 時 Dirichlet 分配的集中度，值越小資料越不均勻。")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="被攻擊的 batch 大小。batch 越大，重建難度越高。")
    parser.add_argument("--method", default="idlg", choices=["dlg", "idlg"],
                        help="攻擊方法：dlg（同時最佳化影像與標籤）或 idlg（先推斷標籤再最佳化影像）。")
    parser.add_argument("--iters", type=int, default=2000,
                        help="梯度匹配最佳化的迭代次數，越多越有機會收斂但耗時越長。")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Adam optimizer 的學習率。")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="從 client dataloader 取第幾個 batch 作為攻擊目標。")
    parser.add_argument("--seed", type=int, default=1,
                        help="隨機種子，確保實驗可重現。")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="運算裝置。梯度匹配需要二階梯度（create_graph=True），CPU 最安全。",
    )
    parser.add_argument("--out-dir", default="../results",
                        help="輸出目錄，存放重建影像、損失曲線與指標 CSV。")
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="每隔幾個 iteration 印一次進度。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定所有隨機源，確保每次執行的 dummy 初始化相同，結果可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """解析裝置名稱，自動退回 CPU（當要求的裝置不可用時）。"""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[WARN] MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def load_model(checkpoint: str, device: torch.device) -> nn.Module:
    """
    載入 Step 1 訓練的全域模型。

    攻擊者需要知道模型架構（白箱攻擊假設），但不需要知道訓練資料。
    requires_grad_(True) 是必要的，因為梯度匹配需要對模型參數計算高階梯度。
    """
    model = create_model().to(device)
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Using random weights.")

    model.eval()
    # 攻擊需要透過模型參數計算 dummy 梯度，因此必須開啟參數梯度
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """將正規化後的 tensor 還原為 [0, 1] 範圍的影像，用於儲存與計算 PSNR。"""
    mean = CIFAR100_MEAN.to(device=x.device, dtype=x.dtype)
    std = CIFAR100_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def normalized_bounds(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    計算正規化空間中對應 pixel [0, 1] 的上下界。

    每次 optimizer step 後用 clamp_ 強制 dummy_x 落在合法像素範圍內，
    防止最佳化飄移到無意義的數值空間。
    """
    mean = CIFAR100_MEAN.to(device)
    std = CIFAR100_STD.to(device)
    lower = (torch.zeros_like(mean) - mean) / std
    upper = (torch.ones_like(mean) - mean) / std
    return lower, upper


def soft_cross_entropy(logits: torch.Tensor, label_logits: torch.Tensor) -> torch.Tensor:
    """
    DLG 使用的 soft cross-entropy loss。

    DLG 中的標籤是可微分的 logits，不能直接用 F.cross_entropy（只接受 hard label）。
    這裡先把 label_logits 轉成概率分佈，再計算與 model 輸出的 KL 散度。
    """
    probs = F.softmax(label_logits, dim=-1)
    return -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def gradient_distance(
    dummy_grads: Sequence[torch.Tensor],
    target_grads: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    計算 dummy 梯度與目標（洩漏）梯度之間的 L2 距離（所有層加總）。

    這是 DLG/iDLG 的核心目標函數：
      L = Σ_layers || ∇_dummy - ∇_target ||²
    當 L 趨近 0，表示 dummy_x 產生的梯度與真實資料的梯度已非常接近。
    """
    distance = torch.zeros((), device=dummy_grads[0].device)
    for dummy_grad, target_grad in zip(dummy_grads, target_grads):
        distance = distance + F.mse_loss(dummy_grad, target_grad, reduction="sum")
    return distance


def infer_idlg_labels(target_grads: Sequence[torch.Tensor], batch_size: int) -> torch.Tensor:
    """
    iDLG 的標籤推斷：從最後一層 bias 的梯度符號直接推算真實標籤。

    原理（來自 iDLG 論文）：
      對於 cross-entropy loss，最後一層 bias 的梯度為：
        ∂L/∂b_c = softmax(z)_c - 1{y == c}
      只有真實類別 c = y 時，梯度會有額外的 -1，因此梯度最小（最負）的那個位置就是標籤。

    batch_size > 1 時 iDLG 不完全精確，改用「梯度最小的前 k 個位置」當近似。
    """
    last_bias_grad = target_grads[-1].detach()
    if batch_size == 1:
        # 單張影像：梯度最小值的 index 就是真實標籤
        return torch.argmin(last_bias_grad).view(1)

    # 多張影像：取最負的前 k 個 index 作為 batch 內標籤的近似估計
    k = min(batch_size, last_bias_grad.numel())
    return torch.topk(-last_bias_grad, k=k).indices


def take_victim_batch(
    client_id: int,
    num_clients: int,
    batch_size: int,
    mode: str,
    alpha: float,
    sample_index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    從指定 client 的 dataloader 取出攻擊目標 batch。

    在真實攻擊中，攻擊者無法取得這份資料，只能觀察梯度；
    這裡模擬「洩漏梯度」的來源，並在實驗結束後用於計算 MSE/PSNR 量化重建品質。
    """
    trainloader, _ = load_dataset(
        num_clients=num_clients,
        client_id=client_id,
        batch_size=batch_size,
        mode=mode,
        alpha=alpha,
        seed=1,
    )

    for batch_idx, (x, y) in enumerate(trainloader):
        if batch_idx == sample_index:
            return x.to(device), y.to(device)

    raise ValueError(
        f"sample-index {sample_index} is out of range for client {client_id}."
    )


def compute_target_gradients(
    model: nn.Module,
    victim_x: torch.Tensor,
    victim_y: torch.Tensor,
) -> List[torch.Tensor]:
    """
    模擬「洩漏梯度」：用真實 victim batch 對模型做一次 forward + backward，
    取出所有參數的梯度。

    在真實的聯邦學習攻擊中，這些梯度是 server 從 client 上傳的 model update 中截取的；
    這裡直接計算以便控制實驗條件。detach() 後作為固定目標，不參與後續最佳化。
    """
    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    logits = model(victim_x)
    loss = criterion(logits, victim_y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return [grad.detach() for grad in grads]


def optimize_reconstruction(
    model: nn.Module,
    target_grads: Sequence[torch.Tensor],
    victim_shape: torch.Size,
    method: str,
    iters: int,
    lr: float,
    device: torch.device,
    log_every: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[float]]:
    """
    核心攻擊迴圈：透過梯度匹配最佳化還原原始影像。

    演算法流程：
      1. 初始化隨機 dummy_x（與原始影像同形狀）
      2. 每個 iteration：
         a. 用 dummy_x 對模型做 forward，計算 dummy 梯度
         b. 計算 dummy 梯度與 target 梯度的 L2 距離（gradient_distance）
         c. 對 dummy_x 做反向傳播，更新 dummy_x
         d. clamp dummy_x 至合法像素範圍
      3. 迭代結束後 dummy_x 即為重建的影像

    DLG vs iDLG 的差異只在標籤的處理方式（見各分支的說明）。
    """
    params = list(model.parameters())
    # 從隨機雜訊開始，讓最佳化從無到有還原影像
    dummy_x = torch.randn(victim_shape, device=device, requires_grad=True)
    lower, upper = normalized_bounds(device)

    dummy_label_logits: Optional[torch.Tensor] = None
    fixed_labels: Optional[torch.Tensor] = None
    optim_params: List[torch.Tensor] = [dummy_x]

    if method == "dlg":
        # DLG：標籤也是可學習的參數，與影像一起最佳化
        # 用 soft label logits 取代 hard label，使整個計算圖可微分
        dummy_label_logits = torch.randn(
            victim_shape[0], 100, device=device, requires_grad=True
        )
        optim_params.append(dummy_label_logits)
    else:
        # iDLG：先用梯度符號分析推算出真實標籤（固定不變），
        # 最佳化時只需要調整影像，收斂更快且更穩定
        fixed_labels = infer_idlg_labels(target_grads, victim_shape[0]).to(device)

    optimizer = torch.optim.Adam(optim_params, lr=lr)
    history: List[float] = []  # 記錄每個 iteration 的梯度匹配損失，用於繪製收斂曲線

    for step in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        logits = model(dummy_x)
        if method == "dlg":
            assert dummy_label_logits is not None
            dummy_loss = soft_cross_entropy(logits, dummy_label_logits)
        else:
            assert fixed_labels is not None
            dummy_loss = F.cross_entropy(logits, fixed_labels)

        # create_graph=True：讓 dummy 梯度本身也在計算圖中，
        # 以便 gradient_distance 的 backward 能更新 dummy_x
        dummy_grads = torch.autograd.grad(dummy_loss, params, create_graph=True)
        match_loss = gradient_distance(dummy_grads, target_grads)
        match_loss.backward()
        optimizer.step()

        # 將 dummy_x 夾回正規化空間中合法的像素範圍 [lower, upper]
        with torch.no_grad():
            dummy_x.clamp_(lower, upper)

        loss_value = float(match_loss.detach().cpu())
        history.append(loss_value)
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == iters):
            print(f"    iter {step:5d}/{iters}: grad_match={loss_value:.6f}")

    # 取出最終推斷標籤（DLG 從 soft logits 取 argmax；iDLG 直接使用推斷值）
    inferred_labels = None
    if method == "dlg" and dummy_label_logits is not None:
        inferred_labels = dummy_label_logits.detach().argmax(dim=1)
    elif fixed_labels is not None:
        inferred_labels = fixed_labels.detach()

    return dummy_x.detach(), inferred_labels, history


def image_metrics(victim_x: torch.Tensor, recovered_x: torch.Tensor) -> Tuple[float, float]:
    """
    計算重建品質指標：
      - MSE：均方誤差，越低越好（完美重建 = 0）
      - PSNR：峰值信噪比（dB），越高越好（完美重建 = ∞）
    兩者均在反正規化後的 [0,1] 影像空間計算。
    """
    victim_img = denormalize(victim_x).detach()
    recovered_img = denormalize(recovered_x).detach()
    mse = F.mse_loss(recovered_img, victim_img).item()
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return mse, psnr


def ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
    """建立輸出目錄結構：original_images / recovered_images / comparisons。"""
    original_dir = base_dir / "original_images"
    recovered_dir = base_dir / "recovered_images"
    comparison_dir = base_dir / "comparisons"
    original_dir.mkdir(parents=True, exist_ok=True)
    recovered_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    return original_dir, recovered_dir, comparison_dir


def save_attack_images(
    victim_x: torch.Tensor,
    recovered_x: torch.Tensor,
    original_path: Path,
    recovered_path: Path,
    comparison_path: Path,
) -> None:
    """
    儲存三種影像：
      - original：被攻擊的真實訓練影像（ground truth）
      - recovered：攻擊還原出的 dummy 影像
      - comparison：左原圖右重建的並排對比圖，方便在報告中展示攻擊效果
    """
    victim_img = denormalize(victim_x.detach().cpu())
    recovered_img = denormalize(recovered_x.detach().cpu())
    save_image(victim_img, original_path)
    save_image(recovered_img, recovered_path)
    # batch size = 1 時排成 1×2 的格子；batch size > 1 時上排原圖、下排重建圖
    nrow = 2 if victim_img.size(0) == 1 else victim_img.size(0)
    grid = make_grid(torch.cat([victim_img, recovered_img], dim=0), nrow=nrow)
    save_image(grid, comparison_path)


def save_loss_curve(history: Sequence[float], path: Path) -> None:
    """將每個 iteration 的梯度匹配損失寫入 CSV，可用於繪製收斂曲線。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iter", "grad_match"])
        for idx, value in enumerate(history, start=1):
            writer.writerow([idx, f"{value:.8f}"])


def append_metrics(csv_path: Path, rows: Iterable[dict]) -> None:
    """
    將本次攻擊的量化指標附加寫入彙整 CSV。

    若 CSV 已存在但欄位不符（版本異動），自動覆寫以避免格式混亂。
    彙整 CSV 方便跨 client、跨方法、跨 batch size 做比較分析。
    """
    fieldnames = [
        "client_id",
        "method",
        "mode",
        "alpha",
        "batch_size",
        "sample_index",
        "iters",
        "mse",
        "psnr",
        "true_labels",
        "inferred_labels",
        "final_grad_match",
        "checkpoint",
        "original_image",
        "recovered_image",
        "comparison_image",
        "loss_curve",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    mode = "a"
    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_header = next(reader, [])
        if existing_header != fieldnames:
            mode = "w"
            file_exists = False

    with csv_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_single_attack(
    args: argparse.Namespace,
    model: nn.Module,
    client_id: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """
    對單一 client 執行完整的梯度洩漏攻擊流程：
      1. 取得 victim batch（模擬洩漏梯度的來源）
      2. 計算洩漏梯度（target gradients）
      3. 執行梯度匹配最佳化還原影像
      4. 計算 MSE / PSNR 量化重建品質
      5. 儲存影像、損失曲線、指標
    """
    print(
        f"[ATTACK] client={client_id} method={args.method} "
        f"batch_size={args.batch_size} mode={args.mode}"
    )
    victim_x, victim_y = take_victim_batch(
        client_id=client_id,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        mode=args.mode,
        alpha=args.alpha,
        sample_index=args.sample_index,
        device=device,
    )
    # 步驟 1：計算洩漏梯度（在真實攻擊中此步驟由 server 攔截 client 上傳的 update）
    target_grads = compute_target_gradients(model, victim_x, victim_y)

    # 步驟 2：從隨機雜訊出發，透過梯度匹配最佳化還原影像
    recovered_x, inferred_labels, history = optimize_reconstruction(
        model=model,
        target_grads=target_grads,
        victim_shape=victim_x.shape,
        method=args.method,
        iters=args.iters,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
    )
    mse, psnr = image_metrics(victim_x, recovered_x)

    # 步驟 3：儲存結果
    original_dir, recovered_dir, comparison_dir = ensure_output_dirs(out_dir)
    loss_curve_dir = out_dir / "loss_curves"
    loss_curve_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.method}_{args.mode}_client{client_id}_"
        f"bs{args.batch_size}_sample{args.sample_index}_{args.iters}iters"
    )
    original_path = original_dir / f"{stem}_original.png"
    recovered_path = recovered_dir / f"{stem}_recovered.png"
    comparison_path = comparison_dir / f"{stem}_comparison.png"
    loss_curve_path = loss_curve_dir / f"{stem}_loss.csv"
    save_attack_images(victim_x, recovered_x, original_path, recovered_path, comparison_path)
    save_loss_curve(history, loss_curve_path)

    true_labels = victim_y.detach().cpu().tolist()
    inferred = [] if inferred_labels is None else inferred_labels.detach().cpu().tolist()
    row = {
        "client_id": client_id,
        "method": args.method,
        "mode": args.mode,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "sample_index": args.sample_index,
        "iters": args.iters,
        "mse": f"{mse:.8f}",
        "psnr": f"{psnr:.4f}",
        "true_labels": true_labels,
        "inferred_labels": inferred,
        "final_grad_match": f"{history[-1]:.8f}" if history else "",
        "checkpoint": args.checkpoint,
        "original_image": str(original_path),
        "recovered_image": str(recovered_path),
        "comparison_image": str(comparison_path),
        "loss_curve": str(loss_curve_path),
    }
    print(
        f"[DONE] client={client_id} mse={mse:.6f} psnr={psnr:.2f} "
        f"true={true_labels} inferred={inferred}"
    )
    return row


def main() -> None:
    """
    進入點：解析參數、載入模型，依序對每個目標 client 執行攻擊，
    最後將所有結果寫入彙整 CSV。
    """
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] device={device}")

    model = load_model(args.checkpoint, device)
    out_dir = Path(args.out_dir)
    # --all-clients 時攻擊全部 client；否則只攻擊 --client-id 指定的那一個
    client_ids = range(args.num_clients) if args.all_clients else [args.client_id]

    rows = []
    for client_id in client_ids:
        rows.append(run_single_attack(args, model, client_id, device, out_dir))

    metrics_path = out_dir / "attack_metrics.csv"
    append_metrics(metrics_path, rows)
    print(f"[INFO] Metrics appended to {metrics_path}")


if __name__ == "__main__":
    main()
