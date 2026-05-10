import os
import torch
import torch.nn as nn
import torch.optim as optim
from attack_model import FaceAttackCNN
from utils import get_dataloaders
from evaluate import run_all_evaluations

def train_model(model, train_loader, criterion, optimizer, num_epochs=30, device='cpu'):
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
    
    # 1. 取得資料
    print("Loading data...")
    train_loader, test_loaders_dict = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. 建立模型與訓練元件
    model = FaceAttackCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. 訓練模型
    print("\n--- Start Training ---")
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)
    
    # 儲存模型權重
    torch.save(trained_model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # 4. 進行評估 (呼叫 evaluate.py 的函式)
    print("\n--- Start Evaluation ---")
    df_results = run_all_evaluations(MODEL_PATH, test_loaders_dict, num_classes=NUM_CLASSES, device=DEVICE)
    
    print("\n最終攻擊準確率結果:")
    print(df_results)

if __name__ == "__main__":
    main()
