# midterm-HW1: 差異隱私合成數據與機器學習效能比較 (Midterm Extension)

本專案目錄負責期中作業的延伸任務 (midterm-HW1)。主要目標是引入嚴謹的 **差異隱私 (Differential Privacy, DP)** 數學架構，利用 `DataSynthesizer` 動態產生毫無隱私洩漏風險的「合成數據」，並將其與 HW1 實作的 **K-匿名性 (K-Anonymity)** 進行相同的預測效能對比 (SVM / MLP 分類器)。

---

## ✨ 實作特色與亮點
* **PrivBayes 結構引入**：採用 `DataSynthesizer` 內建之 Correlated Attribute Mode 生成具備條件機率分佈的貝氏網路，完美保留 Adult 資料集的關聯特徵。
* **嚴謹基線對比 (Controlled Experiment)**：完整對齊 HW1 的清洗後數據 (`adult_cleaned.csv`) 以及特徵轉換邏輯，實現真正的控制變因對比實驗，而非不同起點的比較。

---

## 📂 核心檔案結構說明
新建立的 Python 程式檔已悉數置於 `ML part/` 資料夾下，各自的職責如下：

1. `midterm_HW1_dp_synthesis_and_train.py`
   - **功能**：核心合成與訓練腳本。會讀取 HW1 的乾淨數據，利用 PrivBayes 模型代入隱私預算 $\epsilon \in \{0.1, 1.0, 10.0\}$，產出合成資料表後，自動將新資料拋進 HW1 的 SVM 與 MLP 檔案中進行特徵工程及訓練。
   - **產出物**：模型 (pkl) 自動存入 `MLPmodels/` 與 `SVMmodels/`；其成果成績將紀錄於 `.txt` 並存放在相關 `logs/` 資料夾下。

2. `midterm_HW1_plot.py`
   - **功能**：對比繪圖腳本。自動巡覽並抓取 `MLPlogs/` 內最新版本的 K-Anon (K=2~50) 與 DP 合成 (Epsilon=0.1~10) 訓練日誌解析出 AUC 與 Accuracy 分數。
   - **產出物**：利用 `seaborn` 繪製雙軸直方圖，並自動匯出為 `HW1_Midterm_Comparison.png` 以利簡報貼圖。

3. `../create_word_report.py` (位於 HW1 根目錄)
   - **功能**：一鍵圖文報告產生器。將上述產出之圖表、對比表格數字，以及預先撰寫好的學術級報告大綱轉換編譯並輸出。
   - **產出物**：一份排版整齊的正式 Microsoft Word 文件 (`Midterm_Differentially Private Synthetic Data-HW1.docx`)，打開就能做 PPT 的文案庫。

---

## 🚀 如何自己跑一次 (Quick Start)
請確保您的 Python 執行環境已擁有這些工具：
```bash
pip install pandas numpy scikit-learn matplotlib seaborn DataSynthesizer python-docx
```

接著請在 `HW1/ML part/` 目錄底下依序執行指令：

1. **第一步：啟動合成與實驗**（⚠️ 此步驟將建構厚重的貝氏網路，執行可能需用時 10~20 分鐘）：
   ```bash
   python midterm_HW1_dp_synthesis_and_train.py
   ```
2. **第二步：繪製成績圖表**：
   ```bash
   python midterm_HW1_plot.py
   ```
3. **第三步：如果教授要求紙本報告或詳細文字解說，請在 HW1 根目錄生成 Word 文件**：
   ```bash
   cd ..
   python create_word_report.py
   ```

---

## 📊 實驗結論導覽 (The Privacy-Utility Trade-off)
在對比圖表中，證實了差異隱私架構的特性：
* 在適中預算 ($\epsilon = 10.0$) 的狀況下，數據保留了大部分特徵網路，MLP 模型 AUC 獲得 87.7%，精準度與 K=10 時的 91% 極度逼近，展現了強大的商業潛力。
* 當隱私保護力度極大化 ($\epsilon = 0.1$) 時，為了抵禦數學假設中的單一個體差異性，演算法被迫加入大量破壞性 Lapalce 噪音，使 AUC 雪崩式下降至 69.5%。
* 完美總結了防護與效能如同翹翹板般的對應關係。
