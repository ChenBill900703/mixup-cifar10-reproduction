# 驗證報告：READY FOR FORMAL RUN

日期：2026-10-06（Asia/Taipei）。本報告記錄正式執行前的 preflight。後續使用者要求直接執行，正式 Mixup 已於 11:54 啟動；尚無正式完成數字。啟動證據：outputs/Mixup/launch.json。只接受真實觀測結果，不編造報告數值。

最終驗證識別碼：`20261006T034926`（識別碼以 UTC 命名）。機器可讀證據：[readiness.json](../validation/readiness.json)。完整日誌：[驗證目錄](../validation/20261006T034926/)。

| 檢查 | 結果與證據 |
|---|---|
| 論文 | 全文擷取、p.5 圖與實驗段落目視核對；Figure 3(a) 5.6 / 4.2 已確認 |
| 官方 clone / 模型 | revision eaff31ab397a90fbc0a4aac71fb5311144b3608b；模型原檔 SHA 相等；11,171,274 個參數 |
| 架構運算 | 分別載入專案模型與官方模型，比對初始化、forward 及 backward：PASS |
| 語法 / import / CUDA | PASS；真實 CUDA tensor forward/backward 成功 |
| Mixup | weighted CE 與 soft-label CE 的值和梯度一致；alpha=0 對應 ERM：PASS |
| LR | 實際執行官方調整函式，逐一比對 200 epochs：PASS |
| 資料 | CIFAR-10 原始檔 MD5 完整性檢查：PASS；50,000 / 10,000 |
| ERM Quick Test | epoch 1 存檔後停止，再由另一程序接續 epoch 2：PASS |
| Mixup Quick Test | epoch 1 存檔後停止，再由另一程序接續 epoch 2：PASS |
| 恢復狀態 | Python / NumPy / CPU / CUDA / DataLoader RNG digest 恢復相等；optimizer momentum 存在：PASS |
| 拒絕錯誤設定 | model/dataset/batch/optimizer/seed/alpha 單元測試；實際重啟修改 seed 被拒，checkpoint SHA 未變：PASS |
| 原子存檔 | 注入寫檔中斷後，前一份 checkpoint 仍可讀且位元組相同：PASS |
| 完成後重啟 | 不重新訓練，checkpoint SHA 未变：ERM 與 Mixup 均 PASS |
| 正式執行防護 | 未提供明確批准旗標，拒絕正式執行：PASS |
| 輸出 | CSV/summary/checkpoint/final weights/曲線已由真實短測試產生；曲線標示 QUICK TEST - NOT A RESULT |

Quick Test 使用完整 architecture、batch size 128 及正式 transforms，但每 epoch 只跑 3 個 train batches（384 張）與 2 個 test batches（200 張）。因此它只驗證工程流程，不衡量模型泛化；其 summary 的 result_claim_allowed=False，與論文差距欄位為 null。正式模式會使用完整資料、200 epochs。

## 已驗證的環境

- Python 3.12.14；PyTorch 2.7.1+cu126；torchvision 0.22.1+cu126。
- PyTorch CUDA runtime 12.6；cuDNN 90701（9.7.1）；NVIDIA driver 591.86。
- NVIDIA GeForce RTX 3070 Ti，CUDA 回報 8,589,410,304 bytes VRAM（約 8 GiB），單一可見 GPU。
- torch.cuda.is_available() = True；正式與短測試皆 AMP=False、TF32=False。
- cuDNN benchmark=True 跟隨官方，因此未承諾跨程序／硬體 bitwise 相同。RNG 還原成功不代表所有 kernel 必然 deterministic。

## 正式預定設定

RUN_ERM=False；RUN_MIXUP=True；命令列 --formal 將 QUICK_TEST=False；USE_AMP=False。CIFAR-10 / 官方 PreActBlock ResNet18 / alpha=1 / 200 epochs / batch 128 / SGD momentum .9 / weight decay 1e-4 / seed 20170922 / no dropout。test batch=100。

依使用者指定的官方 boundary 優先例外：**epoch 1–101 LR=.1；102–151 LR=.01；152–200 LR=.001**。Crop32+padding4、horizontal flip .5、mean (.4914,.4822,.4465)、std (.2023,.1994,.2010)。Primary endpoint 固定 epoch 200；best 僅診斷。

## 必須保留的差異

1. 官方 LR 實際邊界與論文文字相差一個 epoch，已明列採用官方行為。
2. 官方 8 workers 在本機實测出現 WinError1455，改為 workers=0；未更改 batch、transforms 或系統分頁檔。失敗紀錄完整保留。
3. 論文 seed 和 PyTorch 版本未公布；新版 BatchNorm 初始化不一定等同歷史環境。保留官方原碼、使用鎖定環境，沒有假裝歷史細節已知。
4. Figure 3(a) 未明示 final/best；依使用者要求採 final，不能把 Table 5 的 last-10 median 混用。

以上屬於來源與執行邊界；完整逐項 audit 見 [source_boundary.md](source_boundary.md)。本報告只確認工程 preflight 通過，不宣稱已複現 4.2%。下一步僅為取得使用者對正式 Mixup 執行的確認。
