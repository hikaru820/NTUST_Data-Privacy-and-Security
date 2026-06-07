"""
Step 3 — DP Flower Server（搭配 client_dp.py）

和 Step 1 server.py 幾乎相同，唯一差別：global model 用 create_dp_model
（BN -> GroupNorm）。因為 DP client 也是 GroupNorm 模型，兩邊的 state_dict
key 必須一致，否則 set_parameters 會對不上。

server 本身不做 DP（DP 發生在 client 的本地訓練），這裡只是用相容的模型結構
做初始化與 server-side 評估。聚合仍是標準 FedAvg。

啟動：
  python server_dp.py --rounds 10 --min-clients 2 --server-eval
"""

import argparse
import os

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader

from data import load_cifar100
from device_utils import get_device, device_name
from client import get_parameters, set_parameters, evaluate
from server import fit_config, evaluate_config, weighted_average  # 沿用 Step 1 設定
from dp_mechanism import create_dp_model


def get_evaluate_fn(testloader, device, save_path, total_rounds):
    """server-side 評估：每輪用乾淨 testset 評估聚合後的 DP global model。"""
    def evaluate_fn(server_round, parameters, config):
        model = create_dp_model().to(device)   # DP 相容（GroupNorm）模型
        set_parameters(model, parameters)
        loss, acc = evaluate(model, testloader, device)
        print(f"[DP Server eval | round {server_round}] loss={loss:.4f} acc={acc:.4f}")
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            if server_round == total_rounds:
                print(f"  -> Final DP global model saved to {save_path}")
        return loss, {"accuracy": acc}
    return evaluate_fn


def main():
    parser = argparse.ArgumentParser(description="DP Flower Server (Step 3)")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--server-eval", action="store_true")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--save-path", default="../checkpoints/dp_global_flower.pth")
    args = parser.parse_args()

    # 用 DP 相容模型取得初始權重，保證 client/server 結構一致
    init_model = create_dp_model()
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    evaluate_fn = None
    if args.server_eval:
        device = get_device()
        print(f"[DP Server] eval device = {device_name(device)}")
        _, testset = load_cifar100()
        testloader = DataLoader(testset, batch_size=128, shuffle=False)
        evaluate_fn = get_evaluate_fn(testloader, device,
                                      save_path=args.save_path, total_rounds=args.rounds)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=init_params,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate_fn,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
