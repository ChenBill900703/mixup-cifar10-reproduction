# CIFAR-10 / PreAct ResNet-18 複現結果

狀態：COMPLETE。更新：2026-10-06T17:03:36.884397+08:00。

Primary metric 固定為第 200 epoch 的 full-test error；所有數值來自實際 checkpoint 與完整紀錄。

| Method | Paper Test Error (%) | Reproduction Final Error (%) | Difference (pp) |
|---|---:|---:|---:|
| ERM | 5.60 | 5.60 | 0.00 |
| Mixup | 4.20 | 4.27 | +0.07 |

## ERM

Final accuracy：94.40%；final error：5.60%。
相對論文 5.60%，差距 0.00 個百分點。
Best error：5.43%（epoch 148），僅診斷，未取代 final。
訓練與評估累計 98.2 分鐘，不含所有啟動／存檔／繪圖時間。

來源：[summary](../outputs/ERM/summary.json)、[200-epoch history](../outputs/ERM/training_history.csv)。

## Mixup

Final accuracy：95.73%；final error：4.27%。
相對論文 4.20%，差距 +0.07 個百分點。
Best error：3.96%（epoch 173），僅診斷，未取代 final。
訓練與評估累計 94.8 分鐘，不含所有啟動／存檔／繪圖時間。

來源：[summary](../outputs/Mixup/summary.json)、[200-epoch history](../outputs/Mixup/training_history.csv)。

同一預定 seed 下，ERM final error 減去 Mixup final error = +1.33 個百分點。
此為單次訓練比較，不代表多 seed 的平均效果，也未建立統計顯著性。

## 忠實度、差異與解讀

兩組使用同一官方模型、seed 20170922、batch 128、200 epochs、SGD momentum .9、weight decay 1e-4、FP32。ERM alpha=0；Mixup alpha=1。原始訓練程式和設定未為了接近論文而調整。

以下是可解釋 reproduction gap 的候選因素，不是已證實的原因：

- Paper seed: UNKNOWN；此處固定 seed 取自官方 README 範例，未依 test error 挑選。
- 論文使用 Tesla P100；此處是 RTX 3070 Ti。
- PyTorch/CUDA/cuDNN 與歷史版本可能不同；原始版本未公布，尤其 BN gamma 初始化預設跨版本有差異。
- cuDNN benchmark=True，不能保證 kernel 或跨程序 bitwise deterministic。
- 官方實際 LR schedule 在 human epoch 102/152 切換；論文文字自然解讀是 101/151，依使用者指定採官方行為。
- Windows 原本 8 workers 載入 DLL 時發生 WinError1455，改 workers=0；transforms 不變，但亂數流消耗順序不同。
- Figure 3(a) 未明示 final-vs-best；本複現遵守 epoch200 endpoint。Table5 的 last-10 median 不混用。

完整來源核對：[source_boundary.md](source_boundary.md)。工程驗證：[validation_report.md](validation_report.md)。

Quick Tests、失敗的初始化測試與 8-worker 資源失敗紀錄均保留；未當成正式結果，也未用於調整 seed 或科學超參數。

## 獨立重新評估核對

2026-10-06 17:02 Asia/Taipei，重新載入兩個 final_model_state_dict.pt，對完整 10,000 張 test images 評估：ERM 答對 9,440 張、錯誤 560 張（5.60%）；Mixup 答對 9,573 張、錯誤 427 張（4.27%）。與 epoch200 紀錄一致，兩組均 PASS。證據：[independent_evaluation.json](../validation/independent_evaluation.json)。

ERM 在 epoch142 後曾因 CPU 記憶體配置失敗中斷，已從保存的 optimizer/RNG/checkpoint 接續 epoch143，完成至200；完整歷史連續且沒有重複或缺失 epoch。失敗原因與恢復紀錄見 [setup_failures.md](setup_failures.md)。
