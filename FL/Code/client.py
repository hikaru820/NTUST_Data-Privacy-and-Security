import argparse
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_dataset
from model import create_model
from device_utils import get_device, device_name

# Parameter <-> NumPy conversion
# Flower 在 server/client 間用 List[np.ndarray] 傳輸權重，
# 所以要把 PyTorch state_dict 跟 NumPy 互轉。
def get_parameters(model: nn.Module):
    """把 model 的權重轉成 list of numpy arrays (給 server 傳輸用）。"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    """把 server 傳回的 numpy weights 灌回 model。"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ------------------------------------------------------------
# Optimizer / criterion factories
#
# 抽出來方便 Step 3 (Differential Privacy) 銜接：
# DP 組員只要在他們的 client_dp.py 裡 override 這兩個 function，
# 例如用 opacus 的 PrivacyEngine 包過 optimizer，train() 就會自動套用，
# 整個 FlowerClient 跟 server.py 都不用動。
#
# 範例 (Step 3)：
#   from opacus import PrivacyEngine
#   def make_optimizer(model, lr, momentum=0.9):
#       opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#       # PrivacyEngine.make_private(...) 會回傳新的 optimizer
#       return opt  # 加上 DP 包裝
# ------------------------------------------------------------
def make_criterion() -> nn.Module:
    """Loss function factory。Step 3 通常不用改。"""
    return nn.CrossEntropyLoss()


def make_optimizer(model: nn.Module, lr: float, momentum: float = 0.9) -> optim.Optimizer:
    """Optimizer factory。Step 3 在這裡換成 opacus 包過的 DP-SGD。"""
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train(model: nn.Module, trainloader, device, epochs: int = 1,
          lr: float = 0.01, momentum: float = 0.9):
    """單一 client 的本地訓練迴圈。"""
    model.train()
    criterion = make_criterion()
    optimizer = make_optimizer(model, lr, momentum)

    last_loss, last_acc = 0.0, 0.0
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        last_loss = running_loss / total
        last_acc = correct / total
        print(f"  [Local Epoch {epoch+1}/{epochs}] "
              f"loss={last_loss:.4f} acc={last_acc:.4f}")

    return last_loss, last_acc

def evaluate(model: nn.Module, testloader, device):
    """在 testset 上計算 loss/accuracy(不更新權重)。"""
    model.eval()
    criterion = make_criterion()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            running_loss += loss.item() * x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

    return running_loss / total, correct / total

class FlowerClient(fl.client.NumPyClient):
    """
    包裝 model + dataloaders 成 Flower client。
    Flower server 會呼叫 get_parameters / fit / evaluate。
    """

    def __init__(self, client_id: int, num_clients: int,
                 mode: str = "iid", alpha: float = 0.5,
                 batch_size: int = 32):
        self.client_id = client_id
        self.device = get_device()
        print(f"[Client {client_id}] device = {device_name(self.device)}")

        # 建 model 並搬到對應裝置
        self.model = create_model().to(self.device)

        # 載入這個 client 的資料分片
        self.trainloader, self.testloader = load_dataset(
            num_clients=num_clients,
            client_id=client_id,
            batch_size=batch_size,
            mode=mode,
            alpha=alpha,
        )

    # ---- Flower API ----
    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """收到 global weights -> 本地訓練 -> 回傳新 weights。"""
        set_parameters(self.model, parameters)

        # server 可以透過 config 傳超參數來
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))

        train(self.model, self.trainloader, self.device,
              epochs=local_epochs, lr=lr)

        # 回傳：新權重、本地樣本數、額外 metrics
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """收到 global weights -> 在本地 testset 評估。"""
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}
    
def main():
    parser = argparse.ArgumentParser(description="Flower FL Client")
    parser.add_argument("--client-id", type=int, required=True,
                        help="Client ID (0 到 num-clients-1)")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--mode", default="iid", choices=["iid", "noniid"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha (僅 noniid 模式)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--server", default="127.0.0.1:8080",
                        help="Flower server address")
    args = parser.parse_args()

    client = FlowerClient(
        client_id=args.client_id,
        num_clients=args.num_clients,
        mode=args.mode,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )

    # Flower 1.13 用 start_client + .to_client()
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()