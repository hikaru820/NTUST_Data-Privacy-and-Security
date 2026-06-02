import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR100CNN(nn.Module):
    """
    Medium CNN 給 CIFAR-100, 以平衡:
      - FL 訓練速度（夠小，DirectML/CUDA 都能跑）
      - 合理準確率（IID 約 50~60% test acc）
      - 適合 Step 2 Gradient Leakage 攻擊（不用殘差連接）

    Architecture: 3 conv blocks (2 conv each) + 2 FC layers
    Total parameters: ~3.3M
    Input:  (B, 3, 32, 32)
    Output: (B, 100) logits

    ⚠️ Step 3 (Differential Privacy) 注意事項：
       本架構使用 BatchNorm2d，但 opacus 的 DP-SGD 不支援 BN
       （違反 sample-level DP，因為 BN 用 batch 內統計量）。
       Step 3 組員需要先呼叫：
           from opacus.validators import ModuleValidator
           model = ModuleValidator.fix(model)  # 自動把 BN 換成 GroupNorm
       這會讓 baseline accuracy 微幅下降，屬正常現象。
    """
   
    def __init__(self, num_classes: int = 100):
            super().__init__()
            
            # Block 1: 3x32x32 -> 64x16x16
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            
            # Block 2: 64x16x16 -> 128x8x8
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4   = nn.BatchNorm2d(128)
            
            # Block 3: 128x8x8 -> 256x4x4
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5   = nn.BatchNorm2d(256)
            self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn6   = nn.BatchNorm2d(256)
            
            # Pooling & regularization
            self.pool    = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            
            # Fully-connected head
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, num_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Block 1
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            
            # Block 2
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)
            
            # Block 3
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = self.pool(x)
            
            # Flatten + FC
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        
def create_model(name: str = "cifar100_cnn", num_classes: int = 100) -> nn.Module:
    """
    模型建構，根據名稱回傳對應的模型實例
    
    之後可建構更多模型(e.g., ResNet, MobileNet)，只要在這裡擴充即可。
    
    攻擊者模型的程式碼也會使用到這個函式來創建模型
    因此要確保這裡的模型定義與 FL 客戶端使用的模型保持一致
    才能正確模擬攻擊者對客戶端模型的攻擊
    """
    name = name.lower()
    if name == "cifar100_cnn":
        return CIFAR100CNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")
    
if __name__ == "__main__":
    model = create_model()
    
    # 參數數量統計
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # 簡單的 forward pass 測試
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"\nInput shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (4, 100), f"Expected (4, 100), got {tuple(out.shape)}"
    print("Forward pass: OK")
    
    # 輸出模型結構
    print("\n--- Architecture ---")
    print(model)