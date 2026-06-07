"""
Step 3-B — DP-FL 訓練：privacy(ε) vs accuracy trade-off

實驗目的（對應作業評分點「Observe the trade-off of privacy and accuracy」）：
  把 DP-SGD（per-sample clip + Gaussian noise）放進聯邦學習的每個 client 本地訓練，
  掃描不同的 noise multiplier σ，量測：
    σ 越大 -> 隱私越強（ε 越小）-> 但梯度雜訊越多 -> test accuracy 越低。

為什麼用單一進程模擬 FedAvg，而不是開多個 Flower terminal？
  本實驗要掃描多個 σ、每個 σ 跑多個 round，用單進程模擬可重現、好掃參數、
  也方便存每個 σ 的最終 global model（供 Step 3-A 攻擊用）。
  忠於 Step 1 Flower 系統的對接版本另見 client_dp.py（用同一套 DP 機制）。

注意：
  - 模型的 BatchNorm 會先換成 GroupNorm（per-sample DP-SGD 必需，見 dp_mechanism）。
  - ε 用 opacus 的 RDP accountant 累積，sample_rate = batch_size / 單一 client 樣本數。
  - 為了在筆電 CPU 上跑得動，預設用每個 client 的資料子集；報告要好看可調大。

使用範例（先跑 baseline 與幾個 σ）：
  python dp_experiment.py --num-clients 5 --rounds 10 --samples-per-client 1000 \
      --sigmas 0 0.5 1.0 2.0 --max-grad-norm 1.0
"""

import argparse
import copy
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import load_cifar100, partition_iid, partition_dirichlet
from model import create_model
from dp_mechanism import (
    replace_bn_with_groupnorm,
    dp_sgd_gradient,
    EpsilonAccountant,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 3-B: DP federated training, privacy-accuracy trade-off."
    )
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16,
                   help="本地訓練 batch（per-sample 梯度會吃記憶體，別開太大）。")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--mode", default="iid", choices=["iid", "noniid"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--samples-per-client", type=int, default=1000,
                   help="每個 client 取多少訓練樣本（子集，加速；-1 = 全部）。")
    p.add_argument("--test-samples", type=int, default=2000,
                   help="評估用的 test 子集大小（-1 = 全部 10000 張）。")
    # DP 超參數掃描
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0],
                   help="要掃描的 noise multiplier σ（0 = 不加噪 baseline）。")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="per-sample 梯度裁切上限 C。")
    p.add_argument("--delta", type=float, default=1e-5,
                   help="(ε, δ)-DP 的 δ，慣例取 1/dataset 量級。")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                   help="per-sample vmap 在 MPS 上可能不穩，預設 CPU 最安全。")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="../results/dp_experiment")
    p.add_argument("--ckpt-dir", default="../checkpoints")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_client_subsets(args, trainset) -> List[List[int]]:
    """依 mode 切分資料給各 client，並可選擇只取前 N 筆加速。"""
    if args.mode == "iid":
        parts = partition_iid(trainset, args.num_clients, seed=args.seed)
    else:
        parts = partition_dirichlet(trainset, args.num_clients, args.alpha, seed=args.seed)

    if args.samples_per_client > 0:
        rng = np.random.default_rng(args.seed)
        parts = [
            rng.permutation(idx)[: args.samples_per_client].tolist() for idx in parts
        ]
    return parts


@torch.no_grad()
def evaluate(model: nn.Module, testloader, device) -> float:
    """回傳 test accuracy。"""
    model.eval()
    correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / total


def local_dp_train(
    model: nn.Module, loader: DataLoader, device, args, sigma: float,
    accountant: EpsilonAccountant, sample_rate: float, generator: torch.Generator,
) -> None:
    """
    單一 client 的本地 DP-SGD 訓練（in-place 更新 model 權重）。

    每個 batch：算 per-sample 梯度 -> 各自 clip 到 C -> 加總加噪 -> 平均 ->
    塞進 param.grad -> SGD step。每個 batch 對隱私帳本記一步。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    for _ in range(args.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            dp_grads = dp_sgd_gradient(
                model, x, y, criterion,
                max_grad_norm=args.max_grad_norm,
                noise_multiplier=sigma,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            for name, param in model.named_parameters():
                param.grad = dp_grads[name]
            optimizer.step()
            accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)


def fedavg(global_model: nn.Module, client_states: List[Dict], weights: List[int]) -> None:
    """以樣本數加權平均各 client 的 state_dict，寫回 global_model（in-place）。"""
    total = float(sum(weights))
    avg = copy.deepcopy(client_states[0])
    for key in avg.keys():
        stacked = torch.stack(
            [state[key].float() * (w / total) for state, w in zip(client_states, weights)],
            dim=0,
        )
        avg[key] = stacked.sum(dim=0)
    global_model.load_state_dict(avg)


def run_one_sigma(args, sigma: float, trainset, client_parts, testloader, device) -> List[dict]:
    """對單一 σ 跑完整 DP-FL，回傳每個 round 的 (sigma, round, acc, eps)。"""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 建 global model 並把 BN 換成 GroupNorm（DP 必需）
    global_model = create_model().to(device)
    replace_bn_with_groupnorm(global_model)
    global_model.to(device)

    # 每個 client 一份資料 loader、隱私帳本、雜訊 generator
    loaders, accountants, generators, client_sizes = [], [], [], []
    for cid in range(args.num_clients):
        subset = Subset(trainset, client_parts[cid])
        loaders.append(DataLoader(subset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True))
        accountants.append(EpsilonAccountant())
        generators.append(torch.Generator(device=device).manual_seed(args.seed + cid))
        client_sizes.append(len(subset))

    rows: List[dict] = []
    for rnd in range(1, args.rounds + 1):
        client_states, weights = [], []
        for cid in range(args.num_clients):
            # 每個 client 從目前 global 權重開始本地訓練
            local_model = copy.deepcopy(global_model)
            sample_rate = args.batch_size / max(1, client_sizes[cid])
            local_dp_train(local_model, loaders[cid], device, args, sigma,
                           accountants[cid], sample_rate, generators[cid])
            client_states.append(local_model.state_dict())
            weights.append(client_sizes[cid])

        fedavg(global_model, client_states, weights)
        acc = evaluate(global_model, testloader, device)
        # σ=0 代表完全不加噪 -> 沒有 DP 保證 -> ε=inf（baseline）。
        # 否則取所有 client 中最壞（最大）的 ε 作為系統隱私保證。
        eps = float("inf") if sigma == 0 else max(
            a.epsilon(delta=args.delta) for a in accountants
        )
        eps_str = "inf" if eps == float("inf") else f"{eps:.4f}"
        print(f"  [σ={sigma:<4g} round {rnd:2d}/{args.rounds}] "
              f"test_acc={acc:.4f}  eps={eps_str}")
        rows.append({"sigma": sigma, "round": rnd,
                     "test_acc": f"{acc:.4f}", "epsilon": eps_str})

    # 存這個 σ 的最終 global model（供 Step 3-A 攻擊；σ=0 即 DP 關閉的 baseline）
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = "baseline" if sigma == 0 else f"sigma{sigma:g}"
    torch.save(global_model.state_dict(), ckpt_dir / f"dp_global_{tag}.pth")
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[INFO] device={device}  sigmas={args.sigmas}")
    if device.type == "mps":
        print("[WARN] MPS 上 per-sample vmap 可能報錯；若失敗請改 --device cpu。")

    trainset, testset = load_cifar100(data_dir="./data")
    client_parts = build_client_subsets(args, trainset)
    print(f"[INFO] client sizes = {[len(p) for p in client_parts]}")

    if args.test_samples > 0:
        idx = np.random.default_rng(args.seed).permutation(len(testset))[: args.test_samples]
        testset = Subset(testset, idx.tolist())
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    all_rows: List[dict] = []
    summary: List[dict] = []
    for sigma in args.sigmas:
        print(f"[RUN] noise_multiplier σ = {sigma}")
        rows = run_one_sigma(args, sigma, trainset, client_parts, testloader, device)
        all_rows.extend(rows)
        summary.append({
            "sigma": sigma,
            "max_grad_norm": args.max_grad_norm,
            "final_acc": rows[-1]["test_acc"],
            "final_epsilon": rows[-1]["epsilon"],
            "rounds": args.rounds,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 逐 round 明細（畫收斂曲線用）
    with (out_dir / "per_round.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sigma", "round", "test_acc", "epsilon"])
        w.writeheader()
        w.writerows(all_rows)
    # 每個 σ 的最終結果（畫 accuracy-vs-ε trade-off 用）
    with (out_dir / "tradeoff.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[INFO] results saved to {out_dir}/per_round.csv 與 tradeoff.csv")
    print("[INFO] 最終 trade-off：")
    for s in summary:
        print(f"    σ={s['sigma']:<4} acc={s['final_acc']}  ε={s['final_epsilon']}")


if __name__ == "__main__":
    main()
