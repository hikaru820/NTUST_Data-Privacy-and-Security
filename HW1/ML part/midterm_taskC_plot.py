import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(filepath):
    """讀取 logtxt 獲取 AUC 與 Accuracy"""
    auc, acc = 0.0, 0.0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("AUC:"):
                auc = float(line.split(":")[1].strip())
            elif line.startswith("Accuracy:"):
                acc = float(line.split(":")[1].strip())
    return auc, acc

def get_latest_log_for_label(log_dir, label_pattern):
    """找尋特定標籤 (例如 k10_MLPresult 或 DP_eps_1.0_MLPresult) 最新輸出的 log 檔案"""
    files = glob.glob(os.path.join(log_dir, f"{label_pattern}*.txt"))
    if not files:
        return None
    # 取時間最新的檔案
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "MLPlogs")

    if not os.path.exists(log_dir):
        print(f"找無日誌資料夾 {log_dir}")
        return

    # 定義我們要比較的組別
    k_labels = ['original', 'k2', 'k5', 'k10', 'k25', 'k50']
    dp_labels = ['DP_eps_0.1', 'DP_eps_1.0', 'DP_eps_10.0']

    results = []

    # 抓取 HW1 的 K-Anonymity 實驗結果
    for label in k_labels:
        f = get_latest_log_for_label(log_dir, f"{label}_MLPresult")
        if f:
            auc, acc = parse_log_file(f)
            type_str = "Baseline" if label == 'original' else "K-Anonymity (HW1)"
            x_label = "Original" if label == 'original' else f"K={label.replace('k', '')}"
            results.append({"Method": type_str, "Parameter": x_label, "AUC": auc, "Accuracy": acc})

    # 抓取期中作業的三組 DP 合成實驗結果
    for label in dp_labels:
        f = get_latest_log_for_label(log_dir, f"{label}_MLPresult")
        if f:
            auc, acc = parse_log_file(f)
            eps_val = label.split("_")[-1]
            results.append({"Method": "DP Synthetic (Task C)", "Parameter": f"Eps={eps_val}", "AUC": auc, "Accuracy": acc})

    if not results:
        print("未找到任何符合條件的模型評估日誌，無法繪製圖表 =(")
        return

    df = pd.DataFrame(results)
    
    print("=== 解析後的評估數據 ===")
    print(df.to_string(index=False))

    # --- 開始繪圖 (使用 Seaborn) ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUC Chart
    sns.barplot(data=df, x="Parameter", y="AUC", hue="Method", ax=axes[0], palette="muted")
    axes[0].set_title("MLP Model Performance Comparison: AUC")
    axes[0].set_ylim(0.5, 1.0) # AUC範圍
    axes[0].tick_params(axis='x', rotation=45)

    # Accuracy Chart
    sns.barplot(data=df, x="Parameter", y="Accuracy", hue="Method", ax=axes[1], palette="muted")
    axes[1].set_title("MLP Model Performance Comparison: Accuracy")
    axes[1].set_ylim(0.5, 1.0) # Accuracy 範圍
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(base_dir, "HW1_Midterm_Comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n對比圖表繪製成功！已儲存為 {plot_path}")

if __name__ == "__main__":
    main()
