"""
Step 3 — DP Flower Client（忠於 Step 1 Flower 系統的對接版）

這是把 DP-SGD 接回 Step 1「真正的 Flower client/server」的版本，對應 README
要 Step 3 組員寫的 client_dp.py。和單進程模擬版 dp_experiment.py 用同一套 DP
機制（dp_mechanism），差別只在這支是真的連上 Flower server 跑。

和 Step 1 client.py 的差異：
  1. 模型用 create_dp_model（BN -> GroupNorm），否則 per-sample DP-SGD 不合法。
     ⚠️ 因此 server 端也必須用同一個模型 -> 請搭配 server_dp.py 啟動。
  2. 本地訓練改用 DP-SGD（per-sample clip + Gaussian noise），而不是普通 SGD。
  3. fit() 回傳當前累積的 ε，方便在 server log 觀察隱私預算。

啟動方式（搭配 server_dp.py）：
  # 視窗 1
  python server_dp.py --rounds 10 --min-clients 2 --server-eval
  # 視窗 2、3 ...
  python client_dp.py --client-id 0 --num-clients 2 --sigma 1.0
  python client_dp.py --client-id 1 --num-clients 2 --sigma 1.0
"""

import argparse

import flwr as fl
import torch
import torch.nn as nn

from data import load_dataset
from device_utils import get_device, device_name
from client import FlowerClient, get_parameters, set_parameters, evaluate
from dp_mechanism import create_dp_model, dp_sgd_gradient, EpsilonAccountant


class DPFlowerClient(FlowerClient):
    """
    在 Step 1 FlowerClient 基礎上，把模型換成 GroupNorm 版、本地訓練換成 DP-SGD。
    """

    def __init__(self, client_id: int, num_clients: int, sigma: float,
                 max_grad_norm: float = 1.0, delta: float = 1e-5,
                 mode: str = "iid", alpha: float = 0.5, batch_size: int = 16):
        # 不呼叫 super().__init__（它會建 BN 模型），自己建 DP 相容模型與資料
        self.client_id = client_id
        self.device = get_device()
        print(f"[DP Client {client_id}] device = {device_name(self.device)}  σ={sigma}")

        self.model = create_dp_model().to(self.device)
        self.trainloader, self.testloader = load_dataset(
            num_clients=num_clients, client_id=client_id,
            batch_size=batch_size, mode=mode, alpha=alpha,
        )

        # DP 超參數與隱私帳本（跨 round 持續累積）
        self.sigma = sigma
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.accountant = EpsilonAccountant()
        # 抽樣率 q = batch / 本地樣本數，供 ε 會計用
        self.sample_rate = batch_size / max(1, len(self.trainloader.dataset))
        self.generator = torch.Generator(device=self.device).manual_seed(1 + client_id)

    def fit(self, parameters, config):
        """收到 global 權重 -> 本地 DP-SGD 訓練 -> 回傳新權重 + 當前 ε。"""
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.05))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0)
        self.model.train()
        for _ in range(local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                dp_grads = dp_sgd_gradient(
                    self.model, x, y, criterion,
                    max_grad_norm=self.max_grad_norm,
                    noise_multiplier=self.sigma,
                    generator=self.generator,
                )
                optimizer.zero_grad(set_to_none=True)
                for name, param in self.model.named_parameters():
                    param.grad = dp_grads[name]
                optimizer.step()
                self.accountant.step(noise_multiplier=self.sigma,
                                     sample_rate=self.sample_rate)

        eps = self.accountant.epsilon(delta=self.delta)
        eps = eps if eps != float("inf") else -1.0   # Flower metrics 不能傳 inf
        print(f"  [DP Client {self.client_id}] local DP-SGD done, ε={eps:.4f}")
        return get_parameters(self.model), len(self.trainloader.dataset), {"epsilon": eps}


def main():
    parser = argparse.ArgumentParser(description="DP Flower Client (Step 3)")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="noise multiplier σ（0 = 不加噪）。")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--mode", default="iid", choices=["iid", "noniid"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--server", default="127.0.0.1:8080")
    args = parser.parse_args()

    client = DPFlowerClient(
        client_id=args.client_id, num_clients=args.num_clients,
        sigma=args.sigma, max_grad_norm=args.max_grad_norm, delta=args.delta,
        mode=args.mode, alpha=args.alpha, batch_size=args.batch_size,
    )
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
