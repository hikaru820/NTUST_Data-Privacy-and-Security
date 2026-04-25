from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os

def create_report():
    doc = Document()

    # 設定中文字體：微軟正黑體
    doc.styles['Normal'].font.name = u'Microsoft JhengHei'
    doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')
    doc.styles['Normal'].font.size = Pt(12)

    # 標題
    title = doc.add_heading('期中作業 Task C - 差異隱私合成數據生成與效能對比分析報告', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.name = u'Microsoft JhengHei'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')

    # 1. 實作任務與研究動機
    h1 = doc.add_heading('1. 實作任務與研究動機 (Task Overview)', level=1)
    for run in h1.runs: run.font.name = u'Microsoft JhengHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')
    
    doc.add_paragraph(
        "在現代數據分享情境中，如何在保證「個體隱私不被洩漏」的前提下，最大化保留資料的「機器學習分析價值 (Utility)」，"
        "是一項極具挑戰的任務。本報告針對 HW1 實作之 K-匿名性 (K-Anonymity) 進行延伸比對，"
        "引進嚴謹的數學定義——差異隱私 (Differential Privacy, DP)。我們透過生成差異隱私合成數據 (DP Synthetic Data) "
        "作為機器學習的新世代訓練集，並探討不同的隱私預算 (Privacy Budget, ε) 投入時，"
        "對多層感知器 (MLP) 模型預測效能的實際衝擊。"
    )

    # 2. 工具與演算法選擇
    h2 = doc.add_heading('2. 工具與演算法選擇 (Tool & Algorithm Selection)', level=1)
    for run in h2.runs: run.font.name = u'Microsoft JhengHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')

    doc.add_heading('2.1 採用開源工具', level=2)
    doc.add_paragraph("本研究不採直覺的「隨機加噪」，而是導入了國際知名套件 DataSynthesizer。")

    doc.add_heading('2.2 核心演算法：PrivBayes (NIST 競賽獲獎演算法)', level=2)
    p = doc.add_paragraph()
    p.add_run("作法與理由：").bold = True
    p.add_run("DataSynthesizer 內建之 Correlated Attribute Mode 實作了著名的 PrivBayes 演算法。對於如 Adult Income 這種具高度複雜連帶關係的表格型資料 (Tabular Data)，如果只對單一欄位獨立加入雜訊，會完全摧毀機器學習最重視的「特徵關聯」（例如：教育程度與收入的強耦合）。")
    
    p = doc.add_paragraph()
    p.add_run("優勢：").bold = True
    p.add_run("PrivBayes 會先透過學習真實資料，建構出多維度的貝氏網路 (Bayesian Network)。演算法在計算條件機率與網路權重時，會精確投入拉普拉斯噪音 (Laplace Noise) 來滿足 ε-DP 保護。如此產生的假資料，不僅從根本上切斷了與「真實個體」的直接連結，又能在總體統計分佈上，高度保留原始資料的關聯特性。")

    # 3. 實作流程與實踐步驟
    h3 = doc.add_heading('3. 實作流程與實踐步驟 (Methodology)', level=1)
    for run in h3.runs: run.font.name = u'Microsoft JhengHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')

    doc.add_paragraph("為確保對比實驗的嚴謹性 (Controlled Experiment)，本次實驗架構嚴格遵循「控制變因」原則：")
    doc.add_paragraph("1. 基準對齊 (Baseline Alignment)：延續 HW1 階段已清洗之 adult_cleaned.csv 作為唯一基準資料 (共 45,222 筆)。確保特徵刪除等邏輯與 HW1 完全一致。")
    doc.add_paragraph("2. 隱私預算分配 (Epsilon Setup)：設計三個級別的差異隱私保護強度，分別為 ε = 0.1 (極高度保護)、ε = 1.0 (中度保護)、ε = 10.0 (輕度保護/高實用性)。利用 PrivBayes 動態生成三組等量的高仿合成數據集。")
    doc.add_paragraph("3. 模型訓練與復現 (Model Training)：將這三組「完全無真實隱私風險」的合成數據，重新拋入 HW1 架構之 MLP 分類器 (兩層隱藏層 64-32，Adam 優化器) 中擬合，並由未被污染的 Test-set 驗證其泛化能力。")

    # 4. 比較結果分析
    h4 = doc.add_heading('4. 比較結果分析 (Results Analysis)', level=1)
    for run in h4.runs: run.font.name = u'Microsoft JhengHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')
    
    doc.add_paragraph("根據 MLP 模型之輸出，在相同測試條件下取得的數值如下表所示：")

    # 加入表格
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '評估方法分類'
    hdr_cells[1].text = '參數設定 (Parameter)'
    hdr_cells[2].text = 'AUC 分數'
    hdr_cells[3].text = '準確度 (Accuracy)'

    records = [
        ('無保護 (基準)', 'Original Dataset', '0.9166', '0.8628'),
        ('取平均參考 (HW1)', 'K-Anonymity (K=10)', '0.9120', '0.8579'),
        ('合成數據 (期中)', 'DP (ε = 10.0)', '0.8774', '0.8322'),
        ('合成數據 (期中)', 'DP (ε = 1.0)', '0.8364', '0.7855'),
        ('合成數據 (期中)', 'DP (ε = 0.1)', '0.6953', '0.7343')
    ]
    for m, p, a, acc in records:
        row_cells = table.add_row().cells
        row_cells[0].text = m
        row_cells[1].text = p
        row_cells[2].text = a
        row_cells[3].text = acc

    # 5. 自動化對比圖表解讀指南
    h5 = doc.add_heading('5. 自動化對比圖表解讀指南 (How to Interpret the Visualizations)', level=1)
    for run in h5.runs: run.font.name = u'Microsoft JhengHei'; run._element.rPr.rFonts.set(qn('w:eastAsia'), u'Microsoft JhengHei')
    
    doc.add_paragraph("在這份雙軸柱狀圖中（圖表為 HW1_Midterm_Comparison.png），我們向團隊展示了「隱私-效能權衡 (Privacy-Utility Trade-off)」的強大證據。報告重點如下：")
    
    doc.add_paragraph("1. 軸線定義與視覺觀察：", style='List Number')
    doc.add_paragraph("X 軸代表不同的防護力度設計；Y 軸則顯示該防護力度下的模型表現（左圖為 AUC 模型鑑別度，右圖為 Accuracy 預測準確度）。")

    doc.add_paragraph("2. K-匿名性 (HW1) vs. DP合成表現：", style='List Number')
    doc.add_paragraph("從圖中左側群集可見，K-Anonymity 的衰退曲線非常平緩。因為它的本質是「模糊化 (Generalization)」，數值仍反映真實世界的分佈邊界，故其 AUC 仍能死守在 90% 以上。相比之下，DP 合成數據 (淺藍色區塊) 則是採用了更嚴苛的數學防護，直接從源頭抹除真實資料個體。")

    doc.add_paragraph("3. 差異隱私內部的「長尾衰退效應」解讀 (核心報告亮點)：", style='List Number')
    p = doc.add_paragraph()
    p.add_run("適度保護 (ε=10.0)：").bold = True
    p.add_run("在圖上可以看到，當預算放寬時，PrivBayes 合成的資料能讓機器學習維持 87.7% 的 AUC 以及 83% 的正確率。這個效能表現極具實務應用價值，相當逼近 HW1 的能力，此結論證明合成數據技術足以代替高度敏感的原始資料發布給第三方做初期研發。")
    
    p = doc.add_paragraph()
    p.add_run("極端防禦的代價 (ε=0.1)：").bold = True
    p.add_run("當預算趨近於極小，代表嚴格要求任何獨立個體的加入都不能改寫整體的貝氏機率分佈。圖中最右側柱體發生斷崖式下墜 (AUC 跌破至 0.69)。為滿足嚴苛的差異隱私數學假設，機率模型被迫加入過多的拉普拉斯噪音，破壞了變數關聯性，印證了絕對的隱私等同於部分資料開發價值的犧牲。")

    # 插入圖片
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML part", "HW1_Midterm_Comparison.png")
    if os.path.exists(img_path):
        doc.add_paragraph("附圖：模型評估結果之柱狀圖對比。")
        doc.add_picture(img_path, width=Pt(400))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        doc.add_paragraph("\n[圖片 HW1_Midterm_Comparison.png 請在此處插入]")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "期中作業_TaskC_差異隱私專題報告.docx")
    doc.save(output_path)
    print("Docx created:", output_path)

if __name__ == '__main__':
    create_report()
