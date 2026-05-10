import torch
import torch.nn as nn

class FaceAttackCNN(nn.Module):
    def __init__(self, num_classes=40, input_channels=1):
        """
        AI Re-identification Attack 模型架構
        :param num_classes: 總共的類別數量 (AT&T dataset 預設為 40 人)
        :param input_channels: 影像通道數 (AT&T dataset 為灰階，預設 1)
        """
        super(FaceAttackCNN, self).__init__()
        
        # 建立卷積層提取特徵
        self.features = nn.Sequential(
            # 第一層卷積
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二層卷積
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三層卷積
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 使用 AdaptiveAvgPool2d 確保能適應不同的輸入圖片大小 (例如 112x92 或 64x64)
        # 固定輸出為 4x4 大小的特徵圖
        self.adap_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # 防止 Overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adap_pool(x)
        x = torch.flatten(x, 1)  # 攤平為一維向量
        x = self.classifier(x)
        return x

# 測試用程式碼 (可以執行此檔案確保模型架構無誤)
if __name__ == "__main__":
    # 假設輸入影像為 batch_size=8, channels=1, height=112, width=92
    dummy_input = torch.randn(8, 1, 112, 92)
    model = FaceAttackCNN(num_classes=40)
    output = model(dummy_input)
    print(f"輸入大小: {dummy_input.shape}")
    print(f"輸出大小: {output.shape} (預期應為 [8, 40])")
