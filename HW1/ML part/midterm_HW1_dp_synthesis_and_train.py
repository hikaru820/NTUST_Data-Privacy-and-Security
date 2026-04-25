import pandas as pd
import numpy as np
from pathlib import Path
import os
import time

import DataSynthesizer.DataGenerator
DataSynthesizer.DataGenerator.np = np
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

# 引入 HW1 的 ML 模型
from SVM import train_svm
from MLP import train_mlp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPSILONS = [0.1, 1.0, 10.0]
base_dir = Path(__file__).resolve().parent
clean_data_path = base_dir.parent / 'adult_cleaned.csv'

# 設定 DataSynthesizer 輸出目錄
out_dir = base_dir.parent / 'DP_outputs'
out_dir.mkdir(parents=True, exist_ok=True)

def load_and_preprocess_dp_data(epsilon):
    """
    載入剛生成的 DP 合成數據並進行與 HW1 相同的特徵工程處理
    """
    csv_file = out_dir / f'adult_dp_{epsilon}.csv'
    df = pd.read_csv(csv_file)
    
    # 保持與 HW1 中 DataConversion 的一致性
    if 'fnlwgt' in df.columns:
        df = df.drop(['fnlwgt'], axis=1)
    if 'education' in df.columns:
        df = df.drop(['education'], axis=1)
    if 'native-country' in df.columns:
        df = df.drop(['native-country'], axis=1) # 跟隨 HW1 Kanon 的做法，丟棄這個不一致的特徵
        
    if df['income'].dtype == object or df['income'].dtype == str:
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    else:
        # 有時候合成完如果被當成連續數值，我們強制讓它大於 0.5 就是 1
        df['income'] = (df['income'] > 0.5).astype(int)

    # 確保不會有 NaN
    df = df.dropna()

    y = df['income']
    x = df.drop('income', axis=1)

    # 分割資料集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # One-hot encoding
    cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
    # 確保兩邊的 dummy 特徵完全對齊
    x_train = pd.get_dummies(x_train, columns=cat_cols)
    x_test = pd.get_dummies(x_test, columns=cat_cols)
    
    all_columns = x_train.columns.union(x_test.columns)
    x_train = x_train.reindex(columns=all_columns, fill_value=0)
    x_test = x_test.reindex(columns=all_columns, fill_value=0)

    # 標準化
    scaler = StandardScaler()
    num_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    # 防止合成資料產生極少數缺失，做個防呆
    for c in num_cols:
        if c not in x_train.columns:
            # 如果不小心生成的名字變成 education-num，找一下
            alt_c = c.replace('educational-num', 'education-num')
            if alt_c in x_train.columns:
                x_train.rename(columns={alt_c: c}, inplace=True)
                x_test.rename(columns={alt_c: c}, inplace=True)
        x_train[c] = x_train[c].fillna(x_train[c].mean())
        x_test[c] = x_test[c].fillna(x_train[c].mean())
        
    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])

    return x_train, x_test, y_train, y_test

def run_midterm_HW1():
    print("=== 初始化期中作業：差異隱私合成數據生成 (PrivBayes 模式) ===")
    
    if not clean_data_path.exists():
        print(f"錯誤：找不到我們 HW1 清洗好的原始數據 {clean_data_path}。請確保它存在！")
        return

    # 計算 tuple 數量
    clean_df = pd.read_csv(clean_data_path)
    num_tuples_to_generate = len(clean_df)

    svm_results = {}
    mlp_results = {}

    for eps in EPSILONS:
        print(f"\n--- 開始生成 DP 合成數據 (Epsilon = {eps}) ---")
        description_file = out_dir / f'description_eps_{eps}.json'
        synthetic_data_file = out_dir / f'adult_dp_{eps}.csv'
        
        # ----- 1. 描述數據結構 (PrivBayes 訓練過程) -----
        describer = DataDescriber(category_threshold=15) 
        # mode='correlated_attribute_mode' 也就是基於貝氏網路的 PrivBayes 方法
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=str(clean_data_path), 
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
        
        print(f"合成數據生成完畢，已保存至 {synthetic_data_file}")
        
        # ----- 3. 機器學習訓練 (套用 HW1 邏輯) -----
        print(f"\n--- 開始使用 DP 合成數據 (Epsilon = {eps}) 訓練模型 ---")
        x_train, x_test, y_train, y_test = load_and_preprocess_dp_data(eps)
        
        # SVM
        print("訓練 SVM 模組...")
        svm_res = train_svm(x_train, x_test, y_train, y_test, label=f'DP_eps_{eps}')
        svm_results[eps] = svm_res
        
        # MLP
        print("訓練 MLP 模組...")
        mlp_res = train_mlp(x_train, x_test, y_train, y_test, label=f'DP_eps_{eps}')
        mlp_results[eps] = mlp_res
        
    print("\n=== 期中作業 DP 合成數據 結果摘要 (midterm-HW1) ===")
    print(f"{'Epsilon':<10} {'SVM AUC':>8} {'SVM ACC':>8} {'MLP AUC':>8} {'MLP ACC':>8}")
    for eps in EPSILONS:
        print(f"{eps:<10} "
              f"{svm_results[eps]['auc']:>8.4f} {svm_results[eps]['accuracy']:>8.4f} "
              f"{mlp_results[eps]['auc']:>8.4f} {mlp_results[eps]['accuracy']:>8.4f}")
              
if __name__ == "__main__":
    run_midterm_HW1()
