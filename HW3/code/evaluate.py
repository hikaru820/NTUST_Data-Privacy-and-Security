import torch
import pandas as pd
import os
from attack_model import FaceAttackCNN

def evaluate_model(model, dataloader, device):
    """
    評估模型在給定 dataloader 上的準確率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # 取得預測結果 (類別 index)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def run_all_evaluations(model_path, dataloaders_dict, num_classes=40, input_channels=1, device: str | torch.device = 'cpu'):
    """
    載入 trained model，測試不同版本的圖片並輸出各組 accuracy
    
    :param model_path: 訓練好的模型權重檔案路徑 (例如: 'models/best_model.pth')
    :param dataloaders_dict: 包含各組不同圖片版本 (Original, Pixel, Blur) dataloader 的字典
    :param num_classes: 類別數量
    """
    # 1. 載入 trained model 架構與權重
    print(f"Loading trained model from {model_path} ...")
    model = FaceAttackCNN(num_classes=num_classes, input_channels=input_channels)
    
    # 處理無 GPU 或 model path 不存在的情況
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[警告] 找不到模型路徑: {model_path}，目前使用未經訓練的隨機權重進行測試！")
        
    model.to(device)
    
    # 2. 測試 original / pixelization / blur 不同版本的圖片，並計算 accuracy
    results = []
    print("\n--- 準備開始評估各組 Dataset 的 Accuracy ---")
    
    for dataset_name, loader in dataloaders_dict.items():
        acc = evaluate_model(model, loader, device)
        results.append({
            'Dataset': dataset_name,
            'Accuracy': f"{acc:.2f}%"
        })
        print(f"[{dataset_name}] Accuracy: {acc:.2f}%")
        
    # 加入 Random Guess 作為基準參考
    random_guess_acc = (1 / num_classes) * 100
    results.append({
        'Dataset': 'Random Guess',
        'Accuracy': f"{random_guess_acc:.2f}%"
    })
    print(f"[Random Guess] Accuracy: {random_guess_acc:.2f}% (Baseline)")
    
    # 3. 將結果轉換為 DataFrame 並儲存為 CSV，方便交給 Eden 做分析
    df_results = pd.DataFrame(results)
    
    # 確保 results 資料夾存在
    os.makedirs('../results', exist_ok=True)
    output_csv = '../results/step2_accuracy.csv'
    df_results.to_csv(output_csv, index=False)
    
    print(f"\n[Done] 測試完成！Accuracy table 已儲存至: {output_csv}")
    return df_results

# 範例執行程式碼 (需等 teammate 完成 loader 後串接)
if __name__ == "__main__":
    print("此腳本提供評估涵式 run_all_evaluations。")
    print("待負責 training pipeline 的同學完成 dataloader 後，即可在 main.py 呼叫此函式進行全面評估。")
