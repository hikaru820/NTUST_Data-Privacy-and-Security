import pandas as pd
import numpy as np
from pathlib import Path
import os
import time

# DataSynthesizer 內部使用舊版 numpy API，需在 import 前先 patch
import DataSynthesizer.DataGenerator
DataSynthesizer.DataGenerator.np = np  # type: ignore
from DataSynthesizer.DataDescriber import DataDescriber   # 學習資料分佈並加入 DP 雜訊
from DataSynthesizer.DataGenerator import DataGenerator   # 從學到的分佈中抽樣合成資料

# 沿用 HW1 的分類器，方便直接與 HW1 結果比較
from SVM import train_svm
from MLP import train_mlp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 測試五個 epsilon 值，涵蓋強隱私（小 ε）到高效用（大 ε）的範圍
EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]
base_dir        = Path(__file__).resolve().parent
clean_data_path = base_dir.parent / 'Adult' / 'adult_cleaned.csv'  # HW1 資料集（已移除含 ? 的列）

# 合成 CSV 與 description JSON 的輸出目錄
out_dir = base_dir.parent / 'DP_outputs'
out_dir.mkdir(parents=True, exist_ok=True)

def load_and_preprocess_dp_data(syn_df, real_test_df):
    """
    以合成資料為訓練集、固定真實資料為測試集，進行特徵工程處理（TSTR 標準做法）。
    所有 epsilon 共用同一份真實測試集，確保 accuracy 數字可以公平比較。
    """
    train_df = syn_df.copy()
    test_df  = real_test_df.copy()

    # 移除在 HW1 k-anonymity 實驗中同樣被排除的欄位
    # fnlwgt 是人口普查抽樣權重，education 與 educational-num 重複，native-country 資料不一致
    drop_cols = ['fnlwgt', 'education', 'native-country']
    for df_ in (train_df, test_df):
        df_.drop(columns=[c for c in drop_cols if c in df_.columns], inplace=True)

    # 將 income 標籤編碼為 0/1（<=50K → 0，>50K → 1）
    # 合成資料有時會將 income 輸出為浮點數，以 0.5 為閾值強制二值化
    for df_ in (train_df, test_df):
        if df_['income'].dtype == object:
            df_['income'] = df_['income'].map({'<=50K': 0, '>50K': 1})
        else:
            df_['income'] = (df_['income'] > 0.5).astype(int)

    train_df = train_df.dropna()
    test_df  = test_df.dropna()

    y_train = train_df['income']
    x_train = train_df.drop('income', axis=1)
    y_test  = test_df['income']
    x_test  = test_df.drop('income', axis=1)

    # One-hot encoding（與 HW1 使用相同的類別欄位）
    cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
    x_train = pd.get_dummies(x_train, columns=[c for c in cat_cols if c in x_train.columns])
    x_test  = pd.get_dummies(x_test,  columns=[c for c in cat_cols if c in x_test.columns])

    # 合成與真實資料的類別值可能不同，導致 dummy 欄位數不一致，需對齊
    all_columns = x_train.columns.union(x_test.columns)
    x_train = x_train.reindex(columns=all_columns, fill_value=0)
    x_test  = x_test.reindex(columns=all_columns,  fill_value=0)

    # DataSynthesizer 有時輸出 'education-num' 而非 'educational-num'，統一欄位名稱
    for df_ in (x_train, x_test):
        if 'education-num' in df_.columns and 'educational-num' not in df_.columns:
            df_.rename(columns={'education-num': 'educational-num'}, inplace=True)

    # 數值欄位標準化；scaler 只在訓練集上 fit，避免資料洩漏到測試集
    scaler   = StandardScaler()
    num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for c in num_cols:
        x_train[c] = x_train[c].fillna(x_train[c].mean())
        x_test[c]  = x_test[c].fillna(x_train[c].mean())  # 用訓練集平均值填補，避免洩漏

    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols]  = scaler.transform(x_test[num_cols])

    return x_train, x_test, y_train, y_test

# ── 統計相似性評估（TVD）────────────────────────────
def marginal_tvd(real_df, syn_df, col):
    """
    計算單一欄位的 Total Variation Distance（TVD）。
    TVD 介於 0～1，越接近 0 代表合成資料的該欄位分佈越接近真實資料。
    """
    if real_df[col].dtype == 'object':
        # 類別欄位：比較各類別的出現比例差異
        real_dist = real_df[col].value_counts(normalize=True)
        syn_dist  = syn_df[col].value_counts(normalize=True)
        idx = real_dist.index.union(syn_dist.index)
        tvd = 0.5 * abs(
            real_dist.reindex(idx, fill_value=0) -
            syn_dist.reindex(idx, fill_value=0)
        ).sum()
    else:
        # 數值欄位：用直方圖近似分佈後計算差異
        bins = np.linspace(real_df[col].min(), real_df[col].max(), 20)
        r, _ = np.histogram(real_df[col], bins=bins, density=True)
        s, _ = np.histogram(syn_df[col],  bins=bins, density=True)
        tvd  = 0.5 * abs(r - s).sum() * (bins[1] - bins[0])
    return tvd


def run_midterm_HW1():
    print("=== 初始化期中作業：差異隱私合成數據生成 (PrivBayes 模式) ===")
    
    if not clean_data_path.exists():
        print(f"錯誤：找不到我們 HW1 清洗好的原始數據 {clean_data_path}。請確保它存在！")
        return

    # 載入並前處理：移除 fnlwgt（人口普查抽樣權重，非個人屬性，不需合成）
    clean_df = pd.read_csv(clean_data_path)
    print(f"原始資料: {clean_df.shape}")
    if 'fnlwgt' in clean_df.columns:
        clean_df = clean_df.drop(columns=['fnlwgt'])
    num_tuples_to_generate = len(clean_df)

    # 存成前處理後的暫存檔，供 DataDescriber 使用
    preprocessed_path = out_dir / 'adult_preprocessed.csv'
    clean_df.to_csv(preprocessed_path, index=False)

    # 切出固定真實測試集，供所有 epsilon 的 SVM/MLP 評估使用（TSTR）
    _, real_test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

    dp_results  = {}
    svm_results = {}
    mlp_results = {}

    for eps in EPSILONS:
        print(f"\n--- 開始生成 DP 合成數據 (Epsilon = {eps}) ---")
        start = time.time()
        description_file = out_dir / f'description_eps_{eps}.json'
        synthetic_data_file = out_dir / f'adult_dp_{eps}.csv'
        
        # ----- 1. 描述數據結構 (PrivBayes 訓練過程) -----
        describer = DataDescriber(category_threshold=15) 
        # mode='correlated_attribute_mode' 也就是基於貝氏網路的 PrivBayes 方法
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=str(preprocessed_path),
            epsilon=eps,
            k=2 # k 是 PrivBayes 的 max parents in Bayesian network
        )
        describer.save_dataset_description_to_file(str(description_file))

        # ----- 2. 生成合成數據 -----
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(
            num_tuples_to_generate,
            str(description_file)
        )
        generator.save_synthetic_data(str(synthetic_data_file))

        elapsed = time.time() - start
        synthetic_df = pd.read_csv(synthetic_data_file)
        dp_results[eps] = {'df': synthetic_df, 'time': elapsed}
        print(f"  完成！耗時 {elapsed:.1f}s，生成 {len(synthetic_df)} 筆，已保存至 {synthetic_data_file}")
        
        # ----- 3. 機器學習訓練 (套用 HW1 邏輯) -----
        print(f"\n--- 開始使用 DP 合成數據 (Epsilon = {eps}) 訓練模型 ---")
        x_train, x_test, y_train, y_test = load_and_preprocess_dp_data(dp_results[eps]['df'], real_test_df)
        
        # SVM
        print("訓練 SVM 模組...")
        svm_res = train_svm(x_train, x_test, y_train, y_test, label=f'DP_eps_{eps}')
        svm_results[eps] = svm_res
        
        # MLP
        print("訓練 MLP 模組...")
        mlp_res = train_mlp(x_train, x_test, y_train, y_test, label=f'DP_eps_{eps}')
        mlp_results[eps] = mlp_res
        
    # ── 步驟五：計算各 epsilon 的平均 TVD ───────────────────────
    # 對所有欄位取平均，得到整體統計相似性的單一指標
    print("\n── 統計相似性評估（TVD）────────────────────────────────")
    tvd_results = {}
    for eps, r in dp_results.items():
        avg_tvd = np.mean([marginal_tvd(clean_df, r['df'], col) for col in clean_df.columns])
        tvd_results[eps] = avg_tvd
        print(f"  ε={eps} 平均 TVD: {avg_tvd:.4f}")

    # ── 結果摘要 ─────────────────────────────────────────────
    print("\n=== 期中作業 DP 合成數據 結果摘要 (midterm-HW1) ===")
    print(f"{'Epsilon':<10} {'Runtime(s)':>10} {'Avg TVD':>9} {'SVM AUC':>8} {'SVM ACC':>8} {'MLP AUC':>8} {'MLP ACC':>8}")
    for eps in EPSILONS:
        print(f"{eps:<10} "
              f"{dp_results[eps]['time']:>10.1f} "
              f"{tvd_results[eps]:>9.4f} "
              f"{svm_results[eps]['auc']:>8.4f} {svm_results[eps]['accuracy']:>8.4f} "
              f"{mlp_results[eps]['auc']:>8.4f} {mlp_results[eps]['accuracy']:>8.4f}")
              
if __name__ == "__main__":
    run_midterm_HW1()
