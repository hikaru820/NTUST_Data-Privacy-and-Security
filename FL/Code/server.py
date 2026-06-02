import argparse
import os
from typing import Dict, List, Tuple, Optional

import flwr as fl
import torch
from flwr.common import Metrics, ndarrays_to_parameters
from torch.utils.data import DataLoader

from model import create_model
from data import load_cifar100
from device_utils import get_device, device_name
from client import get_parameters, set_parameters, evaluate

def fit_config(server_round: int) -> Dict:
    """Server 每輪用這個 dict 控制 client 的 fit() 超參數。"""
    return {
        "server_round": server_round,
        "local_epochs": 1,      # 每個 client 在本地跑幾個 epoch
        "lr": 0.01,             # 也可以做 lr schedule（如 round 大時降低）
    }


def evaluate_config(server_round: int) -> Dict:
    return {"server_round": server_round}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    把多個 client 的 evaluate 結果用樣本數加權平均(樣本多的client影響力越大)
    metrics = [(num_examples_client0, {"accuracy": ...}), ...]
    """
    accs = [num * m["accuracy"] for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return {"accuracy": sum(accs) / total}

def get_evaluate_fn(testloader, device, save_path: Optional[str], total_rounds: int):
    """
    回傳一個 closure 給 Flower：每輪聚合完 global model 後會被呼叫。
    這比 client-side evaluate 更可信（同一個乾淨 testset）。
    """
    def evaluate_fn(server_round: int, parameters, config):
        model = create_model().to(device)
        set_parameters(model, parameters)
        loss, acc = evaluate(model, testloader, device)
        print(f"[Server-side eval | round {server_round}] "
              f"loss={loss:.4f}  acc={acc:.4f}")

        # 每輪覆寫存檔；訓練結束時就是最終 model（給 Step 2 用）
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            if server_round == total_rounds:
                print(f"  -> Final global model saved to {save_path}")

        return loss, {"accuracy": acc}

    return evaluate_fn

def main():
    parser = argparse.ArgumentParser(description="Flower FL Server")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=2,
                        help="開始訓練/評估前最少要連幾個 client")
    parser.add_argument("--server-eval", action="store_true",
                        help="啟用 server-side evaluation (建議開)")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--save-path", default="../checkpoints/global_model.pth",
                        help="最終 global model 儲存路徑")
    args = parser.parse_args()

    # 用一個新 model 取得初始權重（保證 client 跟 server 結構一致）
    init_model = create_model()
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    # 設定 server-side evaluation（可選但強烈建議）
    evaluate_fn = None
    if args.server_eval:
        device = get_device()
        print(f"[Server] eval device = {device_name(device)}")
        _, testset = load_cifar100()
        testloader = DataLoader(testset, batch_size=128, shuffle=False)
        evaluate_fn = get_evaluate_fn(
            testloader, device,
            save_path=args.save_path,
            total_rounds=args.rounds,
        )

    # FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                       # 每輪用 100% 可用 client 訓練
        fraction_evaluate=1.0,                  # 每輪用 100% 可用 client 評估
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients, # 等到至少這麼多 client 才開始
        initial_parameters=init_params,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate_fn,
    )

    # 啟動 server，會 block 直到所有 round 跑完
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()