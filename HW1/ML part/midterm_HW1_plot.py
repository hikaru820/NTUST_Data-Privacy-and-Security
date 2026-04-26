import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def marginal_tvd(real_df, syn_df, col):
    """計算單一欄位的 TVD（Total Variation Distance），衡量合成與真實分佈的差異"""
    if real_df[col].dtype == 'object':
        real_dist = real_df[col].value_counts(normalize=True)
        syn_dist  = syn_df[col].value_counts(normalize=True)
        idx = real_dist.index.union(syn_dist.index)
        return 0.5 * abs(
            real_dist.reindex(idx, fill_value=0) -
            syn_dist.reindex(idx, fill_value=0)
        ).sum()
    else:
        bins = np.linspace(real_df[col].min(), real_df[col].max(), 20)
        r, _ = np.histogram(real_df[col], bins=bins, density=True)
        s, _ = np.histogram(syn_df[col],  bins=bins, density=True)
        return 0.5 * abs(r - s).sum() * (bins[1] - bins[0])


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
    dp_labels = ['DP_eps_0.1', 'DP_eps_0.5', 'DP_eps_1.0', 'DP_eps_5.0', 'DP_eps_10.0']


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
            results.append({"Method": "DP Synthetic (midterm-HW1)", "Parameter": f"Eps={eps_val}", "AUC": auc, "Accuracy": acc})

    if not results:
        print("未找到任何符合條件的模型評估日誌，無法繪製圖表 =(")
        return

    df = pd.DataFrame(results)

    print("=== 解析後的評估數據 ===")
    print(df.to_string(index=False))

    # --- 計算 TVD（從合成 CSV 與真實資料直接算）---
    dp_dir       = os.path.join(base_dir, '..', 'DP_outputs')
    real_path    = os.path.join(base_dir, '..', 'Adult', 'adult_cleaned.csv')
    epsilons     = [0.1, 0.5, 1.0, 5.0, 10.0]
    tvd_values   = []
    eps_with_tvd = []

    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        if 'fnlwgt' in real_df.columns:
            real_df = real_df.drop(columns=['fnlwgt'])

        for eps in epsilons:
            syn_path = os.path.join(dp_dir, f'adult_dp_{eps}.csv')
            if os.path.exists(syn_path):
                syn_df  = pd.read_csv(syn_path)
                avg_tvd = np.mean([marginal_tvd(real_df, syn_df, c) for c in real_df.columns if c in syn_df.columns])
                tvd_values.append(avg_tvd)
                eps_with_tvd.append(eps)

    # --- 取 DP 折線圖資料（從 results 裡篩出 DP 那幾筆）---
    dp_df = df[df['Method'] == 'DP Synthetic (midterm-HW1)'].copy()
    dp_df['Epsilon'] = dp_df['Parameter'].str.replace('Eps=', '').astype(float)
    dp_df = dp_df.sort_values('Epsilon')

    # --- 開始繪圖（2×2 版面）---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 圖1（左上）：Accuracy vs Epsilon 折線圖
    if not dp_df.empty:
        axes[0, 0].plot(dp_df['Epsilon'], dp_df['Accuracy'], 'b-o', label='MLP Accuracy')
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Epsilon (ε)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('ML Utility vs Privacy Budget')
        axes[0, 0].legend()

    # 圖2（右上）：TVD vs Epsilon 折線圖
    if tvd_values:
        axes[0, 1].plot(eps_with_tvd, tvd_values, 'g-s')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel('Epsilon (ε)')
        axes[0, 1].set_ylabel('Average TVD')
        axes[0, 1].set_title('Statistical Similarity vs Privacy Budget')

    # 圖3（左下）：AUC 比較長條圖（k-anonymity vs DP）
    sns.barplot(data=df, x="Parameter", y="AUC", hue="Method", ax=axes[1, 0], palette="muted")
    axes[1, 0].set_title("Model Performance Comparison: AUC")
    axes[1, 0].set_ylim(0.5, 1.0)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 圖4（右下）：Accuracy 比較長條圖（k-anonymity vs DP）
    sns.barplot(data=df, x="Parameter", y="Accuracy", hue="Method", ax=axes[1, 1], palette="muted")
    axes[1, 1].set_title("Model Performance Comparison: Accuracy")
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(base_dir, "HW1_Midterm_Comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n對比圖表繪製成功！已儲存為 {plot_path}")

if __name__ == "__main__":
    main()
