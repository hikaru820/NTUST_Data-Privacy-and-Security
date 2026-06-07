"""
Step 3-A — 用 Differential Privacy 防禦 Step 2 的梯度洩漏攻擊

實驗目的（對應作業評分點「證明防禦有效」與「How do you perturb the data?」）：
  Step 2 的攻擊靠「攔截到的真實梯度」做 gradient matching 還原原圖。
  本腳本在那組梯度離開 client 前，套上 DP 的 Gaussian mechanism
  （per-vector clip 到 C + 加 N(0, (σ·C)²) 雜訊），再交給同一支 DLG/iDLG
  攻擊去還原，掃描不同的 noise multiplier σ，觀察：
    σ 越大 -> 還原影像越像雜訊 -> MSE 上升、PSNR 下降 -> 防禦越強。

設計上完全重用 Step 2 的 attack_dlg.py，只在「計算梯度」與「攻擊」之間
插入 perturb_gradients() 這一道防禦，確保攻防條件一致、可公平比較。

使用範例：
  python attack_dp_defense.py --method idlg --iters 1000 \
      --sigmas 0 0.01 0.05 0.1 0.5 1.0 --max-grad-norm 1.0
"""

import argparse
import csv
from pathlib import Path
from typing import List

import torch
from torchvision.utils import make_grid, save_image

# 直接重用 Step 2 的攻擊元件，確保攻防使用完全相同的程式碼路徑
from attack_dlg import (
    set_seed,
    resolve_device,
    load_model,
    take_victim_batch,
    compute_target_gradients,
    optimize_reconstruction,
    image_metrics,
    denormalize,
)
from dp_mechanism import perturb_gradients, compute_epsilon, create_dp_model


def load_attacked_model(checkpoint: str, device: torch.device):
    """
    載入被攻擊的模型。Step 3 的 DP checkpoint 是 GroupNorm 模型存的，
    與 Step 2 的 BatchNorm load_model 不相容，因此這裡用 create_dp_model
    （GroupNorm）建模型再 strict-load。找不到 checkpoint 時退回 Step 2 的
    BatchNorm 隨機模型，方便在沒有 DP 訓練結果時也能跑攻擊。
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        print(f"[WARN] checkpoint 不存在：{ckpt_path}，改用 Step 2 隨機 BatchNorm 模型。")
        return load_model(checkpoint, device)

    model = create_dp_model().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"[INFO] Loaded DP (GroupNorm) checkpoint: {ckpt_path}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 3-A: defend the Step 2 gradient leakage attack with DP."
    )
    p.add_argument("--checkpoint", default="../checkpoints/global_model.pth",
                   help="被攻擊的全域模型權重；找不到時用隨機初始化模型。")
    p.add_argument("--client-id", type=int, default=0)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--mode", default="iid", choices=["iid", "noniid"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1,
                   help="被攻擊的 batch 大小（攻擊在 bs=1 時最容易成功）。")
    p.add_argument("--method", default="idlg", choices=["dlg", "idlg"])
    p.add_argument("--iters", type=int, default=1000,
                   help="每個 σ 的梯度匹配迭代數。")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    # DP 防禦超參數
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="梯度裁切上限 C（Gaussian mechanism 的 sensitivity）。")
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
                   help="要掃描的 noise multiplier σ 清單（0 = 只裁切不加噪）。")
    p.add_argument("--out-dir", default="../results/dp_defense")
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] device={device}")

    # 載入要攻擊的模型（白箱假設：攻擊者知道架構與權重）
    model = load_attacked_model(args.checkpoint, device)

    # 取受害者 batch，並算出「原始的洩漏梯度」（尚未加防禦）
    victim_x, victim_y = take_victim_batch(
        client_id=args.client_id, num_clients=args.num_clients,
        batch_size=args.batch_size, mode=args.mode, alpha=args.alpha,
        sample_index=args.sample_index, device=device,
    )
    clean_grads = compute_target_gradients(model, victim_x, victim_y)
    print(f"[INFO] victim labels = {victim_y.detach().cpu().tolist()}")

    out_dir = Path(args.out_dir)
    recovered_dir = out_dir / "recovered_images"
    recovered_dir.mkdir(parents=True, exist_ok=True)
    # 另存一份原圖（ground truth），供報告做 original vs 各 σ 的對照表
    save_image(denormalize(victim_x.detach().cpu()), out_dir / "original.png")

    # 抽樣率假設：被攔截的更新用整個本地 batch 算一步 -> 用於估「單次釋出」的 ε。
    # 真正可比較的 ε 來自 Step 3-B 的訓練累積；這裡的 ε 僅供「單次梯度釋出」參考。
    sample_rate = 1.0

    rows: List[dict] = []
    recovered_for_grid: List[torch.Tensor] = []
    for sigma in args.sigmas:
        # 用固定 generator 讓每個 σ 的雜訊可重現
        gen = torch.Generator(device=device).manual_seed(args.seed)
        # 防禦：對洩漏梯度套 Gaussian mechanism（複製一份，不污染 clean_grads）
        defended_grads = perturb_gradients(
            clean_grads, max_grad_norm=args.max_grad_norm,
            noise_multiplier=sigma, generator=gen,
        )

        # 用被防禦過的梯度重跑 Step 2 攻擊
        set_seed(args.seed)  # 固定 dummy 初始化，攻擊起點對所有 σ 一致
        recovered_x, _, history = optimize_reconstruction(
            model=model, target_grads=defended_grads,
            victim_shape=victim_x.shape, method=args.method,
            iters=args.iters, lr=args.lr, device=device,
            log_every=0,
        )
        mse, psnr = image_metrics(victim_x, recovered_x)
        eps = compute_epsilon(sigma, sample_rate=sample_rate, steps=1)

        # 存單張還原圖
        rec_img = denormalize(recovered_x.detach().cpu())
        save_image(rec_img, recovered_dir / f"sigma_{sigma:g}.png")
        recovered_for_grid.append(rec_img)

        eps_str = "inf" if eps == float("inf") else f"{eps:.3f}"
        print(f"[σ={sigma:<5g}] mse={mse:.5f}  psnr={psnr:6.2f} dB  "
              f"grad_match={history[-1]:.4f}  eps(1-step)={eps_str}")
        rows.append({
            "sigma": sigma,
            "max_grad_norm": args.max_grad_norm,
            "method": args.method,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "iters": args.iters,
            "mse": f"{mse:.8f}",
            "psnr": f"{psnr:.4f}",
            "final_grad_match": f"{history[-1]:.8f}",
            "eps_single_release": eps_str,
        })

    # 對照圖：最左原圖，往右是各 σ 的還原結果（橫向排列），一眼看出防禦效果
    original_img = denormalize(victim_x.detach().cpu())
    grid_imgs = [original_img] + recovered_for_grid
    grid = make_grid(torch.cat(grid_imgs, dim=0),
                     nrow=len(grid_imgs), padding=2)
    grid_path = out_dir / f"defense_grid_{args.method}_{args.mode}.png"
    save_image(grid, grid_path)
    print(f"[INFO] comparison grid (original + each σ) saved to {grid_path}")

    # 彙整 CSV：給報告畫 PSNR-vs-σ 曲線用
    csv_path = out_dir / "defense_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
