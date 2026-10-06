# v1.0.0 — 已驗證的 CIFAR-10 Mixup 複現

兩組正式訓練均已完成200 epochs，primary metric固定為最後一輪：

| Method | Paper error | Reproduction final error | Difference (pp) |
|---|---:|---:|---:|
| ERM | 5.60% | 5.60% | 0.00 |
| Mixup α=1 | 4.20% | 4.27% | +0.07 |

重新載入最終模型、完整測試10000張影像：ERM錯560張、Mixup錯427張，與紀錄一致。未以best checkpoint取代final；這是單seed結果。

## 下載內容

- **mixup-reproduction-complete.zip**：完整發布程式、繁體中文說明、兩組200輪CSV與圖表、驗證證據、原始錯誤與恢復日誌、兩組正式final weights與resume checkpoints，以及Quick Test模型。
- **SHA256SUMS.txt**：下載包的SHA-256。ZIP內另外提供逐檔manifest。

不含Python虛擬環境、CIFAR-10原始資料、下載快取或私人對話；requirements與官方資料下載方式完整保留。

## 重要邊界

使用原作者模型檔。FP32、batch128、SGD momentum.9、weight decay1e-4、seed20170922。官方LR按實際程式在human epochs102/152切換；與論文文字差異已揭露。Windows workers=0。Paper seed與歷史framework版本未知，BN初始化版本差異已記錄。

ERM曾在epoch142後遇到CPU allocation失敗，從完整checkpoint恢復至epoch143並完成200；未重跑整個實驗，也未改seed或科學參數。完整過程見README連結的白話訓練紀錄。

GitHub自動提供的Source code ZIP不含大型模型；要保存含模型的完整實驗，請下載上述complete.zip。
