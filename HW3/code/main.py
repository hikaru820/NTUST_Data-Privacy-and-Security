import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from attack_model import FaceAttackCNN
from utils import get_dataloaders
from evaluate import run_all_evaluations

def train_model(model, train_loader, criterion, optimizer, num_epochs=30, device: str | torch.device = 'cpu'):
    """
    訓練 CNN 模型
    """
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # 計算訓練準確率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            
    return model

def main():
    # 設定參數
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    NUM_CLASSES = 40
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = '../results/best_model.pth'
    
    print(f"Using device: {DEVICE}")
    
    # 建立結果資料夾
    os.makedirs('../results', exist_ok=True)
    
    # 攻擊者列表。要新增就在這裡加一項即可。
    attackers = [
        {'name': 'orig',     'train_source': 'original'},   # 原本的 naive 攻擊者
        {'name': 'adaptive_blur_k99', 'train_source': 'blur_k99'},   # 自適應攻擊者
    ]

    all_results = {}
    for atk in attackers:
        name, src = atk['name'], atk['train_source']
        print(f"\n========== 訓練攻擊者: {name}  (來源: {src}) ==========")

        train_loader, test_loaders_dict = get_dataloaders(
            batch_size=BATCH_SIZE, train_source=src
        )

        model = FaceAttackCNN(num_classes=NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        trained = train_model(model, train_loader, criterion, optimizer,
                              num_epochs=NUM_EPOCHS, device=DEVICE)

        model_path = f'../results/best_model_{name}.pth'
        torch.save(trained.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        df = run_all_evaluations(model_path, test_loaders_dict,
                                 num_classes=NUM_CLASSES, device=DEVICE, tag=name)
        all_results[name] = df

    # 把所有攻擊者的結果合成一張對照表
    merged = None
    for name, df in all_results.items():
        sub = df.rename(columns={'準確率': f'準確率_{name}'})
        if merged is None:
            merged = sub.copy()
        else:
            # left join 保留第一個 df 的順序，且兩邊的 ===== 分隔列會自動對齊
            merged = merged.merge(sub, on='資料集', how='left')

    merged.to_csv('../results/step2_accuracy_comparison.csv', index=False)
    print("\n[完成] 對照表已存到 ../results/step2_accuracy_comparison.csv")
    print(merged)

if __name__ == "__main__":
    main()
