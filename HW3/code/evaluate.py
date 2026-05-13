import torch
import pandas as pd
import os
from attack_model import FaceAttackCNN
import re

# 群組顯示順序 — 未列出的項目會被排在最底部
_GROUP_ORDER = [
    'original',
    'pixel',           # pixel_b4, b8, b16  (無 DP)
    'blur',            # blur_k15, k45, k99 (無 DP)
    'dp_original',
    'dp_pixel_b4',
    'dp_blur_k15',
    'dp_pixel_b16',
    'dp_blur_k99',
]

def _parse_name(name):
    """回傳 (群組, 排序鍵) 用於群組化以及群組內的排序。
       排序鍵 (sort_key) 是數值參數（如區塊大小、核大小或 epsilon）。"""
    if name == 'original':
        return 'original', 0
    m = re.match(r'^pixel_b(\d+)$', name)
    if m: return 'pixel', int(m.group(1))
    m = re.match(r'^blur_k(\d+)$', name)
    if m: return 'blur',  int(m.group(1))
    m = re.match(r'^dp_(.+)_eps([\d.]+)$', name)
    if m: return f'dp_{m.group(1)}', float(m.group(2))
    return name, 0   # 未知格式 -> 獨立成組，不進行內部排序

def _group_index(g):
    return _GROUP_ORDER.index(g) if g in _GROUP_ORDER else len(_GROUP_ORDER)


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


def run_all_evaluations(model_path, dataloaders_dict, num_classes=40, input_channels=1, device: str | torch.device = 'cpu', tag=''):
    """
    載入訓練好的模型，測試不同版本的圖片並輸出各組準確率 (Accuracy)
    
    :param model_path: 訓練好的模型權重檔案路徑 (例如: 'models/best_model.pth')
    :param dataloaders_dict: 包含各組不同圖片版本 (原始、像素化、模糊) 資料加載器的字典
    :param num_classes: 類別數量
    """
    # 1. 載入訓練好的模型架構與權重
    print(f"正在從 {model_path} 載入訓練好的模型 ...")
    model = FaceAttackCNN(num_classes=num_classes, input_channels=input_channels)
    
    # 處理無 GPU 或模型路徑不存在的情況
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"[警告] 找不到模型路徑: {model_path}，目前使用未經訓練的隨機權重進行測試！")
        
    model.to(device)
    
    # 2. 測試原始 / 像素化 / 模糊等不同版本的圖片，並計算準確率
    results = []
    raw = []
    print("\n--- 準備開始評估各組資料集 (Dataset) 的準確率 ---")
    for dataset_name, loader in dataloaders_dict.items():
        acc = evaluate_model(model, loader, device)
        group, key = _parse_name(dataset_name)
        raw.append({
            '資料集':  dataset_name,
            '準確率': f"{acc:.2f}%",
            '_group':   group,
            '_key':     key,
        })

    # 排序：先按群組順序，再按各組內的數值參數排序
    raw.sort(key=lambda r: (_group_index(r['_group']), r['_key']))

    # 輸出分組後的資料列，並在每個新群組前加入分隔符號
    results = []
    last_group = None
    for r in raw:
        if r['_group'] != last_group:
            results.append({
                '資料集':  f"===== {r['_group']} =====",
                '準確率': '',
            })
            print(f"\n===== {r['_group']} =====")
            last_group = r['_group']
        results.append({'資料集': r['資料集'], '準確率': r['準確率']})
        print(f"[{r['資料集']}] 準確率: {r['準確率']}")

    # 最後加上基準值 (Baseline)
    random_guess_acc = (1 / num_classes) * 100
    results.append({'資料集': '===== 基準值 (Baseline) =====', '準確率': ''})
    results.append({'資料集': '隨機猜測 (Random Guess)', '準確率': f"{random_guess_acc:.2f}%"})
    print(f"\n===== 基準值 (Baseline) =====\n[隨機猜測] 準確率: {random_guess_acc:.2f}%")

    # 3. 儲存為 CSV 檔案
    df_results = pd.DataFrame(results)
    os.makedirs('../results', exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    output_csv = f'../results/step2_accuracy{suffix}.csv'
    df_results.to_csv(output_csv, index=False)
    
    print(f"\n[完成] 準確率表格已儲存至: {output_csv}")
    return df_results

# 範例執行程式碼 (需等組員完成 loader 後進行串接)
if __name__ == "__main__":
    print("此腳本提供評估函式 run_all_evaluations。")
    print("待負責訓練流程 (Training Pipeline) 的同學完成 dataloader 後，即可在 main.py 呼叫此函式進行全面評估。")