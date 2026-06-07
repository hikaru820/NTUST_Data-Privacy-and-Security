"""
Step 3 — Differential Privacy 共用機制

本模組是 Step 3 兩個實驗（A: 防禦攻擊 / B: privacy-utility trade-off）共用的核心，
集中放「怎麼擾動梯度」與「怎麼算 ε」的程式碼，讓兩個實驗共用同一份實作。

DP-SGD 的本質只有兩個動作（Abadi et al., 2016）：
  1. Per-sample gradient clipping：把每個樣本的梯度 L2 norm 裁切到上限 C
     -> 限制單一樣本對更新的最大影響力（bounded sensitivity）。
  2. Gaussian noise：對裁切後的梯度加上 N(0, (σ·C)²) 雜訊
     -> 讓「有沒有某個樣本」在輸出上難以區分，提供 (ε, δ)-DP 保證。

對 Step 2 的梯度洩漏攻擊來說，這兩步剛好打在攻擊命脈：
  攻擊靠「攔截到的真實梯度」做 gradient matching；梯度被裁切+加噪後，
  還原出的影像就會崩成雜訊（見 attack_dp_defense.py）。

⚠️ BatchNorm 問題：model.py 用了 BatchNorm，但 BN 在 batch 內耦合各樣本，
   使「per-sample 梯度」失去意義，違反 sample-level DP。所以做 DP 訓練前要先把
   BN 換成 GroupNorm（replace_bn_with_groupnorm），這會讓 baseline accuracy
   微幅下降，屬正常現象。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from model import create_model


# ------------------------------------------------------------------
# 1. BatchNorm -> GroupNorm（DP 訓練前的前處理）
# ------------------------------------------------------------------
def replace_bn_with_groupnorm(model: nn.Module, num_groups: int = 32) -> nn.Module:
    """
    遞迴把 model 內所有 BatchNorm2d 換成 GroupNorm（in-place）。

    GroupNorm 不依賴 batch 內統計量，每個樣本獨立正規化，因此和 per-sample
    DP-SGD 相容。num_groups 會自動取 min(num_groups, channel 數) 並確保整除。
    等同 opacus.validators.ModuleValidator.fix() 對 BN 做的事，但不需要 opacus，
    也避開 opacus 在 MPS/DirectML 上的 vmap 限制。
    """
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = num_groups
            while num_channels % groups != 0:   # GroupNorm 要求 channel 能被 group 整除
                groups //= 2
            setattr(model, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_bn_with_groupnorm(child, num_groups)
    return model


def create_dp_model(num_classes: int = 100, num_groups: int = 32) -> nn.Module:
    """
    建立 DP 相容的模型：先用 Step 1 的 create_model，再把 BN 換成 GroupNorm。

    Flower 的 server 與 DP client 都要用這個函式建模，state_dict 的 key 才會
    一致（BN 和 GroupNorm 的參數名不同，混用會在 set_parameters 時對不上）。
    """
    model = create_model(num_classes=num_classes)
    replace_bn_with_groupnorm(model, num_groups)
    return model


# ------------------------------------------------------------------
# 2. 梯度向量的 Gaussian mechanism（給 Step 3-A 防禦用）
#    對「整組已聚合的梯度」做 flat clipping + 加噪。
#    這對應攻擊情境：攻擊者攔截到的是一組梯度，我們在它離開 client 前擾動它。
# ------------------------------------------------------------------
def _global_l2_norm(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """把多個 tensor 串起來看成一個大向量，計算其 L2 norm。"""
    return torch.sqrt(sum((t.detach() ** 2).sum() for t in tensors))


def perturb_gradients(
    grads: Sequence[torch.Tensor],
    max_grad_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """
    對一整組梯度套用 Gaussian mechanism（防禦 Step 2 攻擊）：
      1. flat clip：所有層串成一個向量，整體 L2 norm 超過 C 就等比例縮小。
      2. add noise：每個元素加 N(0, (σ·C)²) 高斯雜訊。

    回傳擾動後的梯度（detached），可直接餵給 Step 2 的 gradient matching。
    noise_multiplier = 0 時只裁切不加噪，方便做 ablation。
    """
    grads = [g.detach().clone() for g in grads]
    total_norm = _global_l2_norm(grads)
    clip_coef = min(1.0, max_grad_norm / (total_norm.item() + 1e-12))
    std = noise_multiplier * max_grad_norm

    perturbed = []
    for g in grads:
        g = g * clip_coef
        if std > 0:
            noise = torch.normal(
                mean=0.0, std=std, size=g.shape,
                generator=generator, device=g.device, dtype=g.dtype,
            )
            g = g + noise
        perturbed.append(g)
    return perturbed


# ------------------------------------------------------------------
# 3. Per-sample DP-SGD 梯度（給 Step 3-B 訓練用）
#    用 torch.func 算每個樣本的梯度 -> 各自 clip -> 加總 -> 加噪 -> 平均。
# ------------------------------------------------------------------
def _per_sample_grads(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    用 torch.func.vmap + grad 一次算出 batch 內每個樣本的梯度。

    回傳 dict：param_name -> tensor，shape = (batch_size, *param_shape)。
    計算期間強制 model 進 eval 模式：關閉 dropout 的隨機性（否則 vmap 會因
    隨機運算報錯），且 GroupNorm 在 train/eval 行為一致，確保 per-sample 梯度
    有良好定義。算完還原原本的 train/eval 狀態。
    """
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_on_one(p, single_x, single_y):
        out = functional_call(model, (p, buffers), (single_x.unsqueeze(0),))
        return criterion(out, single_y.unsqueeze(0))

    was_training = model.training
    model.eval()
    try:
        # in_dims=(None, 0, 0)：params 不沿 batch 展開，x/y 各自沿第 0 維展開
        per_sample = vmap(grad(loss_on_one), in_dims=(None, 0, 0))(params, x, y)
    finally:
        model.train(was_training)
    return per_sample


def dp_sgd_gradient(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    max_grad_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
    max_physical_batch: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    計算一個 mini-batch 的 DP-SGD 梯度，回傳 dict：param_name -> 平均後的 DP 梯度。

    流程（Abadi DP-SGD 的一個 step）：
      1. per-sample 梯度（torch.func）
      2. 每個樣本各自 clip 到 L2 norm <= C
      3. clip 後的梯度加總
      4. 加 N(0, (σ·C)²) 雜訊
      5. 除以 batch size 取平均
    呼叫端只要把回傳值塞進 param.grad 再 optimizer.step() 即可。

    max_physical_batch：把大 batch 切成小塊跑 vmap，避免 per-sample 梯度
    （shape = batch × 全參數）撐爆記憶體。
    """
    param_names = [k for k, _ in model.named_parameters()]
    accum: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v) for k, v in model.named_parameters()
    }
    batch_size = x.size(0)

    for start in range(0, batch_size, max_physical_batch):
        xb = x[start:start + max_physical_batch]
        yb = y[start:start + max_physical_batch]
        ps = _per_sample_grads(model, xb, yb, criterion)

        # 計算每個樣本跨所有層的 global L2 norm -> (chunk,)
        flat = torch.stack([
            ps[k].reshape(xb.size(0), -1).pow(2).sum(dim=1) for k in param_names
        ], dim=0).sum(dim=0)
        per_sample_norm = torch.sqrt(flat + 1e-12)
        clip_factor = (max_grad_norm / per_sample_norm).clamp(max=1.0)  # (chunk,)

        for k in param_names:
            g = ps[k]                                   # (chunk, *shape)
            factor = clip_factor.view(-1, *([1] * (g.dim() - 1)))
            accum[k] += (g * factor).sum(dim=0)         # clip 後加總

    # 加噪 + 平均
    std = noise_multiplier * max_grad_norm
    dp_grads: Dict[str, torch.Tensor] = {}
    for k in param_names:
        g = accum[k]
        if std > 0:
            g = g + torch.normal(
                mean=0.0, std=std, size=g.shape,
                generator=generator, device=g.device, dtype=g.dtype,
            )
        dp_grads[k] = g / batch_size
    return dp_grads


# ------------------------------------------------------------------
# 4. ε 隱私帳本（用 opacus 的 RDP accountant，只算帳不包 model）
# ------------------------------------------------------------------
class EpsilonAccountant:
    """
    薄包裝 opacus.RDPAccountant，用來把「σ / sample_rate / steps」換算成 ε。

    我們手刻 clip+noise，但隱私會計（RDP composition + subsampling 放大）數值上
    很容易出錯，所以這部分沿用 opacus 經過驗證的帳本。只 import accountant，
    完全不碰它的 make_private / vmap，因此在 MPS 上也能用。
    """

    def __init__(self) -> None:
        from opacus.accountants import RDPAccountant
        self._acc = RDPAccountant()

    def step(self, noise_multiplier: float, sample_rate: float) -> None:
        """記錄一個 DP-SGD step（σ 與這一步的抽樣率 q = batch / dataset）。"""
        if noise_multiplier > 0:
            self._acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    def epsilon(self, delta: float = 1e-5) -> float:
        """回傳目前累積的 ε（σ=0 全程未加噪時隱私無界，回 inf）。"""
        try:
            return float(self._acc.get_epsilon(delta=delta))
        except (ValueError, IndexError):
            return float("inf")


def compute_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
) -> float:
    """
    便利函式：給定固定 σ、抽樣率 q、總步數，直接回傳 ε。
    σ=0 視為不加噪，ε=inf（無隱私保證）。
    """
    if noise_multiplier <= 0 or steps <= 0:
        return float("inf")
    acc = EpsilonAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acc.epsilon(delta=delta)


if __name__ == "__main__":
    # 自我測試：BN 替換、梯度擾動、per-sample DP 梯度、ε 計算
    from model import create_model

    print("[1] BN -> GroupNorm")
    m = create_model()
    n_bn_before = sum(isinstance(mod, nn.BatchNorm2d) for mod in m.modules())
    replace_bn_with_groupnorm(m)
    n_bn_after = sum(isinstance(mod, nn.BatchNorm2d) for mod in m.modules())
    n_gn = sum(isinstance(mod, nn.GroupNorm) for mod in m.modules())
    print(f"    BN: {n_bn_before} -> {n_bn_after}, GroupNorm: {n_gn}")

    print("[2] perturb_gradients (clip + noise)")
    fake_grads = [torch.randn(64, 3, 3, 3), torch.randn(64)]
    out = perturb_gradients(fake_grads, max_grad_norm=1.0, noise_multiplier=1.0)
    print(f"    global norm after clip+noise = {_global_l2_norm(out).item():.3f}")

    print("[3] dp_sgd_gradient (per-sample clip + noise)")
    m.eval()
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    dp_g = dp_sgd_gradient(m, x, y, nn.CrossEntropyLoss(),
                           max_grad_norm=1.0, noise_multiplier=1.0)
    print(f"    produced {len(dp_g)} param grads, "
          f"fc2.weight grad shape = {tuple(dp_g['fc2.weight'].shape)}")

    print("[4] compute_epsilon")
    eps = compute_epsilon(noise_multiplier=1.0, sample_rate=0.01, steps=1000)
    print(f"    sigma=1.0 q=0.01 steps=1000 -> eps={eps:.3f}")
    print("All self-tests passed.")
