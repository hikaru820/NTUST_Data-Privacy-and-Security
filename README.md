# HW1: K-Anonymity Experiment

## 專案說明

本作業實作基於 **Mondrian 多維度 K-匿名化演算法**，對 Adult Census Income 資料集進行隱私保護處理，並比較不同 K 值下機器學習模型的效能變化，以觀察隱私保護強度對資料可用性的影響。

> 參考論文：K. LeFevre, D. J. DeWitt, and R. Ramakrishnan, "Mondrian Multidimensional K-Anonymity," ICDE, Vol. 6, 2006.

---

## 資料集

| 項目 | 說明 |
|------|------|
| 來源 | UCI Machine Learning Repository — Adult Census Income |
| 原始筆數 | 48,842 筆 |
| 前處理後 | 45,222 筆（移除含缺值列） |
| 目標變數 | `income`（`<=50K` → 0，`>50K` → 1） |
| 正負比例 | 75.2% (0) / 24.8% (1) |

---

## 欄位設計

### Quasi-Identifier（QI）欄位（匿名化對象）

| 欄位 | 類型 | 匿名化方式 | 範例 |
|------|------|-----------|------|
| `age` | 數值 | Mondrian range | `38` → `36-40` |
| `educational-num` | 數值 | Mondrian range | `9` → `7-9` |
| `hours-per-week` | 數值 | Mondrian range | `40` → `35-40` |
| `gender` | 類別 | Generalization hierarchy | `Male` / `Female` → `*` |
| `race` | 類別 | Generalization hierarchy | `Black` → `Non-White` → `*` |
| `marital-status` | 類別 | Generalization hierarchy | `Divorced` → `Separated-Divorced` → `*` |

### 特殊處理欄位

| 欄位 | 處理方式 | 原因 |
|------|---------|------|
| `native-country` | Full suppression（全部設為 `*`） | 國家種類繁多，泛化效果有限，直接抑制 |
| `education` | 從資料集移除（不參與 ML） | 與 `educational-num` 資訊重複，保留數值版本 |

---

## Generalization Hierarchy

```
gender:
  Male / Female → *

race:
  Black / Asian-Pac-Islander / Amer-Indian-Eskimo / Other → Non-White
  White → White
  Non-White / White → *

marital-status:
  Married-civ-spouse / Married-AF-spouse → Married
  Never-married / Widowed → Single
  Divorced / Separated / Married-spouse-absent → Separated-Divorced
  Married / Single / Separated-Divorced → *
```

---

## Mondrian 演算法說明

採用 **Top-down greedy strict multidimensional partitioning**：

1. 從整個資料集開始作為一個 partition
2. 每次選擇 **normalized range 最寬**的 QI 維度
3. 以 **中位數（median）** 做切割
4. 若兩側各自包含 ≥ K 筆資料，遞迴繼續切割
5. 無法繼續切割時，對該 partition 套用 summary statistics：
   - 數值型：以 range 字串表示（如 `25-40`）
   - 類別型：以 hierarchy 的最低共同祖先（LCA）表示
6. 全部完成後，每個等價類大小 ≥ K（K-Anonymity 保證）

**時間複雜度**：O(n log n)

---

## 實驗設計

### K 值設定

| K 值 | 等價類數量 | 最小等價類大小 |
|------|-----------|--------------|
| 2 | ~6,667 | 2 |
| 5 | ~3,401 | 5 |
| 10 | ~2,031 | 10 |
| 25 | ~966 | 25 |
| 50 | ~554 | 50 |

### 機器學習模型

分別在**原始資料**與各 **K 值匿名化資料**上訓練：

- **SVM**（Support Vector Machine）
- **Deep Learning**（Neural Network，使用 PyTorch / Keras）

### 評估指標

| 指標 | 說明 |
|------|------|
| Accuracy | 整體正確率 |
| Misclassification Rate | 1 − Accuracy |
| Precision | 預測為正的精確率 |
| Recall | 實際為正的召回率 |
| AUC | ROC 曲線下面積 |

---

## 檔案結構

```
HW1/
├── README.md                          # 本文件
├── project1.ipynb                     # 主程式（含所有步驟）
├── adult.csv                          # 原始資料集
├── adult_cleaned.csv                  # 前處理後資料
├── adult_k2.csv                       # K=2 匿名化資料
├── adult_k5.csv                       # K=5 匿名化資料
├── adult_k10.csv                      # K=10 匿名化資料
├── adult_k25.csv                      # K=25 匿名化資料
├── adult_k50.csv                      # K=50 匿名化資料
├── MultiDim.pdf                       # 參考論文
└── HW_K-anonymity_v3_20250316.pdf     # 作業說明
```

---

## 執行方式

### 環境需求

```bash
pip install pandas numpy scikit-learn torch matplotlib
```

### 執行步驟

開啟 `project1.ipynb`，依序執行各 Step：

| Step | 內容 |
|------|------|
| Step 1 | 資料前處理（移除缺值、encode target） |
| Step 2 | 定義 Generalization Hierarchy |
| Step 3 | 實作 Mondrian K-Anonymity |
| Step 4 | 對 K=2,5,10,25,50 產生匿名化資料集 |
| Step 5 | Feature Engineering（range → 數值、one-hot encoding） |
| Step 6 | 訓練 ML 模型（SVM + Deep Learning） |
| Step 7 | 評估並比較各 K 值的效能 |

---

## 參考文獻

1. K. LeFevre, D. J. DeWitt, and R. Ramakrishnan, "Mondrian Multidimensional K-Anonymity," *ICDE*, Vol. 6, 2006.
2. H. Wimmer and L. Powell, "A Comparison of the Effects of K-Anonymity on Machine Learning Algorithms," *CONISAR*, Vol. 2167, 2014.
3. UCI Machine Learning Repository — Adult Dataset.
