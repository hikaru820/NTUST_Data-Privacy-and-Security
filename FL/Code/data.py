import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter

# 定義並回傳訓練與測試資料的預處理轉換
def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return train_transform, test_transform

# 下載並回傳完整訓練集與測試集 (完整未切分)
def load_cifar100(data_dir: str = "./data"):
    train_transform, test_transform = get_transforms()
    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    return trainset, testset

# 將資料集進行 IID (獨立同分布) 切分，回傳一個包含各 client 樣本索引的 list
def partition_iid(trainset, num_clients: int, seed: int = 1):
    np.random.seed(seed)
    total_samples = len(trainset)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # 均分索引路徑
    client_indices = np.array_split(indices, num_clients)
    # 轉換成 list of list 格式
    return [c.tolist() for c in client_indices]

# 使用 Dirichlet 分布進行 Non-IID 切分，回傳一個包含各 client 樣本索引的 list
def partition_dirichlet(trainset, num_clients: int, alpha: float = 0.5, seed: int = 1):
    np.random.seed(seed)
    labels = np.array(trainset.targets) # CIFAR100 的 targets 是 list of int
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(100):  # 100 個 class
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        # 依據 alpha 產生每個 client 的分配比例
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # 將比例映射成實際切分索引
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, splits)
        
        for i, p in enumerate(parts):
            client_indices[i].extend(p.tolist())
            
    return client_indices

# 回傳 DataLoader
def load_dataset(num_clients: int, client_id: int, batch_size: int = 32, mode: str = "iid", alpha: float = 0.5, seed: int = 1):
    # 載入原始資料
    trainset, testset = load_cifar100(data_dir="./data")
    
    # 根據模式選用不同的切分演算法獲取所有 Client 的索引清單
    if mode.lower() == "iid":
        all_client_indices = partition_iid(trainset, num_clients, seed)
    elif mode.lower() in ["noniid", "dirichlet"]:
        all_client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
    else:
        raise ValueError(f"不支援的切分模式: {mode}。請選擇 'iid' 或 'noniid'。")
        
    # 提取當前指定 client_id 的索引，並用 Subset 包裝
    client_indices = all_client_indices[client_id]
    client_trainset = Subset(trainset, client_indices)
    
    # 封裝成 DataLoader 回傳
    trainloader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

if __name__ == "__main__":
    num_clients = 5
    
    print("==================================================")
    print("  驗證一:IID 獨立同分布切分模式")
    print("==================================================")
    for cid in range(num_clients):
        trainloader, _ = load_dataset(num_clients=num_clients, client_id=cid, mode="iid", seed=1)
        # 透過 trainloader.dataset 獲取 Subset，並撈出真正的 targets 標籤
        subset_indices = trainloader.dataset.indices
        full_trainset = trainloader.dataset.dataset
        client_targets = [full_trainset.targets[i] for i in subset_indices]
        
        class_counts = Counter(client_targets)
        print(f"Client {cid} - 樣本總數: {len(subset_indices)}")
        # 印出前 5 個類別的樣本數作為分佈代表（預期極度平均，每類大約 100 張圖片左右）
        top_5_classes = sorted(class_counts.items())[:5]
        print(f"   ↳ 前5類別分佈範例: {top_5_classes} ...\n")

    print("==================================================")
    print("  驗證二:Non-IID 狄利克雷切分模式 (alpha=0.1 強異質)")
    print("==================================================")
    for cid in range(num_clients):
        trainloader, _ = load_dataset(num_clients=num_clients, client_id=cid, mode="noniid", alpha=0.1, seed=1)
        subset_indices = trainloader.dataset.indices
        full_trainset = trainloader.dataset.dataset
        client_targets = [full_trainset.targets[i] for i in subset_indices]
        
        class_counts = Counter(client_targets)
        print(f"Client {cid} - 樣本總數: {len(subset_indices)}")
        # 印出前 5 個類別的樣本數（預期會落差極大，某些類別極多，某些類別甚至為 0）
        top_5_classes = sorted(class_counts.items())[:5]
        print(f"   ↳ 前5類別分佈範例: {top_5_classes} ...\n")