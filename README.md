# Mixup 論文複現：程式、完整訓練紀錄與真實結果

這個專案複現 **《mixup: Beyond Empirical Risk Minimization》在 CIFAR-10 上的 PreAct ResNet-18 實驗**。我們使用原作者的模型程式，在 Windows、單張 RTX 3070 Ti 8 GB 上，分別完成一般訓練（ERM）與 Mixup 的 **200 epochs**。

這裡的重點是「照事先決定的設定訓練，再誠實公布得到的結果」。沒有為了接近論文而換 seed、調整 weight decay，或把表現最好的中途模型當成最後結果。

**狀態：兩組正式訓練完成；checkpoint、完整紀錄與重新評估全部核對通過。**

## 先看最後結果

以下都使用 **第 200 epoch 結束時的模型**，單位為 test error（錯誤率）：

| 方法 | 論文 Figure 3(a) | 本次 final error | 與論文的差距 | 重新評估：錯誤張數／10,000 |
|---|---:|---:|---:|---:|
| ERM，一般訓練 | 5.60% | **5.60%** | **0.00 個百分點** | 560 |
| Mixup，α=1 | 4.20% | **4.27%** | **+0.07 個百分點** | 427 |

白話來說：同樣考 10,000 張圖片，一般訓練錯 560 張，Mixup 錯 427 張；本次 Mixup 的錯誤總數少了 133 張，錯誤率降低 **1.33 個百分點**。

這是一個固定 seed 的結果，**不是多次訓練的平均，也不是統計顯著性的證明**。數字接近論文，不代表所有歷史環境細節都已完全還原；重要差異在下面公開說明。

| 僅供診斷的最佳中途結果 | best error | 出現 epoch |
|---|---:|---:|
| ERM | 5.43% | 148 |
| Mixup | 3.96% | 173 |

上表的 best **沒有拿來替代**正式 final。原始 JSON 中出現 `5.599999999999994` 或極接近零的差距，是浮點數儲存現象；閱讀版按兩位小數呈現，原始資料保留不改。

## 從哪裡開始看

- **想了解訓練怎麼做、為什麼中斷又能繼續：**[完整白話訓練過程](docs/training_walkthrough_zh.md)。
- **想核對數字與限制：**[結果與差異分析](docs/reproduction_results.md)、[論文／官方程式逐項對照](docs/source_boundary.md)。
- **想看每一輪發生什麼事：**[ERM 200 筆紀錄](outputs/ERM/training_history.csv)、[Mixup 200 筆紀錄](outputs/Mixup/training_history.csv)。
- **想下載完整模型與 checkpoint：**[v1.0.0 完整實驗包](https://github.com/ChenBill900703/mixup-cifar10-reproduction/releases/tag/v1.0.0)。
- **想自行操作：**[下載、驗證與重新訓練說明](docs/reproduce_zh.md)。
- **想快速確認證據：**[結果一致性檢查](validation/final_integrity.json)、[重新載入模型的完整 test-set 評估](validation/independent_evaluation.json)。

## ERM 和 Mixup 到底差在哪裡？

**ERM** 可以理解成一般的學習方式：給模型一張圖片，例如貓，要求模型把它判成貓。模型預測錯了，就依照錯誤程度更新參數。

**Mixup** 則先把同一個 minibatch 裡的圖片順序打亂，配成另一組，然後混合兩張圖片。例如 70% 的貓加上 30% 的狗，學習目標也同時改成 70% 貓、30% 狗。這不是把混合圖片硬說成某一個類別。

```text
λ ~ Beta(α, α)，本次 α = 1
混合圖片 = λ × 圖片 A + (1 − λ) × 圖片 B
訓練 loss = λ × CE(預測, A 的標籤) + (1 − λ) × CE(預測, B 的標籤)
```

α=1 時，λ 在 0 到 1 之間均勻取樣。這份官方實作是**每個 minibatch 取一個 λ**，同批圖片一起使用；配對在同一批內隨機打亂，不建立第二個獨立 DataLoader。

## 訓練設定：在看到正式結果前就固定

| 項目 | 本次設定 |
|---|---|
| 資料集 | 官方 CIFAR-10，50,000 張 train、10,000 張 test |
| 模型 | 原作者 `ResNet18()`，實際由 PreActBlock 組成；原檔直接保留 |
| 參數量 | 11,171,274 |
| 每組訓練長度 | 200 epochs |
| Train batch size / test batch size | 128 / 100 |
| Optimizer | SGD，momentum 0.9，weight decay 1e-4，無 Nesterov |
| Seed | 20170922，取自官方 README 範例；**Paper seed: UNKNOWN** |
| Mixup / ERM | α=1 / α=0 |
| 學習率 | epochs 1–101：0.1；102–151：0.01；152–200：0.001 |
| 精度 | FP32；AMP=False、TF32=False |
| Dropout / warmup / cosine / label smoothing | 都沒有 |
| 訓練圖片處理 | RandomCrop(32, padding=4)、RandomHorizontalFlip(p=0.5)、ToTensor、Normalize |
| Normalize mean | (0.4914, 0.4822, 0.4465) |
| Normalize std | (0.2023, 0.1994, 0.2010) |
| 測試圖片處理 | 只做 ToTensor 與相同 Normalize |
| DataLoader workers | 0；原本 8 在本機 Windows 載入 DLL 時失敗，調整經過有保留 |
| cuDNN benchmark | True，跟隨官方；不承諾逐位元完全相同 |

**Epoch** 是把全部訓練圖片走過一輪。每輪 50,000 張，batch 128，會有 391 批，最後一批 80 張也保留。因此每組 200 epochs 有 78,200 次參數更新。測試時，每輪都用完整 10,000 張圖片、100 批做評估。

## 必須注意的來源差異

### 1. 官方 LR 實際切換點是 102、152

論文文字寫在 100、150 epochs 後降低學習率；但官方程式的 epoch 從 0 起算，而且在該輪訓練和測試**結束後**才調整。所以換成從 1 起算，實際影響的是第 **102、152** 輪。

本專案依預先確定的規則採官方程式行為，並用測試逐一比對全部 200 輪，沒有默默把它「修正」成另一個 schedule。

### 2. 不是 torchvision 的 ResNet18

官方模型保留了自己的 stem BatchNorm/ReLU、PreActBlock 和 shortcut 行為。現在常見的其他 PreActResNet18 實作也不一定相同，因此這裡保留原作者模型原檔，不凭印象換架構。

### 3. 歷史環境並非完全已知

作者沒有公布完整 PyTorch/CUDA/cuDNN 版本和 paper seed。尤其舊版與新版 BatchNorm 的初始化預設值不同。本次保留官方模型建構方式，使用固定的現代 runtime defaults；這是**明確揭露的 assumption**，不是宣稱已重建當年所有細節。

### 4. 最後一輪、最佳一輪、最後十輪中位數不同

Figure 3(a) 報告 5.6%／4.2%，沒有明確說明 final-vs-best 的選取方式。本專案事先固定使用 **epoch 200 final**。Table 5 的 last-10-epochs median 是另一種統計量，不能拿來混用。

## 用什麼電腦跑？

- Windows 11，單張 NVIDIA GeForce RTX 3070 Ti，約 8 GB VRAM。
- 使用者提供的主機規格：Intel i7-11700K、32 GB RAM、SSD。
- 實測 Python 3.12.14、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126。
- CUDA runtime 12.6、cuDNN 9.7.1、NVIDIA driver 591.86。
- 正式開始前確認 `torch.cuda.is_available() == True`。

Mixup 累計訓練與評估約 **94.8 分鐘**；ERM 約 **98.2 分鐘**。這不包含所有安裝、啟動、寫檔、繪圖與中斷等待時間，不能把這兩個數字當成整個專案的牆鐘時間。

## 有沒有失敗？有，而且沒有刪掉紀錄

1. CUDA 套件下載遇到系統暫存空間不足，改到 E 槽暫存。
2. 初期測試的一個手填參數量預期值寫錯，修正測試，沒有改模型。
3. Windows 8 個資料載入程序載入 cuDNN 時出現 WinError1455，改成 0 workers。batch 128、模型與 transforms 都維持。
4. 正式 ERM 在保存 epoch142 後，下一輪組 batch 時 CPU 配置約 1.5 MiB 記憶體失敗。後來確認 checkpoint 完整，從 epoch143 接續完成200。

這些是可追溯的工程問題。**沒有把失敗當成結果、沒有因為 test error 不理想而換 seed 重跑。** 詳細時間、錯誤與恢復方法在[白話訓練過程](docs/training_walkthrough_zh.md)及[失敗紀錄](docs/setup_failures.md)。

## 為什麼能相信這份結果？

我們做了以下核對，而不是只看一張表：

- 官方模型原檔與使用中的模型檔 SHA-256 相同；forward/backward 比對通過。
- Mixup loss 與 soft-label cross-entropy 的值、梯度一致。
- Quick Test 故意在 epoch1 存檔後停止，再以另一個程序接續 epoch2。
- 每個 epoch 原子寫入模型、optimizer、LR、所有 RNG、DataLoader generator、history 和環境。
- 兩組的完整 CSV 都有連續 200 輪，每輪使用 50,000/10,000 張資料。
- 最終 weights 與 epoch200 checkpoint 的 tensor 逐項一致。
- 重新載入模型，完整評估 10,000 張 test images，得到 ERM 9,440 張正確、Mixup 9,573 張正確，與紀錄吻合。

重新評估沒有訓練，也沒有選新的 checkpoint。模型與紀錄的 SHA-256 可供他人核對。

## 結果圖

![ERM 與論文比較](outputs/ERM/paper_vs_reproduction.png)
![Mixup 與論文比較](outputs/Mixup/paper_vs_reproduction.png)
![ERM test error 完整曲線](outputs/ERM/test_error_curve.png)
![Mixup test error 完整曲線](outputs/Mixup/test_error_curve.png)

每組資料夾另外提供 test accuracy 與 training loss 圖。Mixup 的 label 本身是混合目標，因此不要直接把它的 train loss 與 ERM 的 train loss 高低當成誰學得比較好；正式比較用相同 test set 的分類錯誤率。

## 檔案分工

```text
train.py                         正式訓練與安全續跑
config.json                      集中的預設設定；預設 QUICK_TEST=True
models/official_resnet.py         原作者模型原檔
utils/                           資料、checkpoint、報表
tests/                           結構、loss、LR、resume 驗證
outputs/ERM/                     正式 ERM 200 輪紀錄、圖表、summary
outputs/Mixup/                   正式 Mixup 200 輪紀錄、圖表、summary
outputs/quick/                   與正式結果隔離的短測試證據
validation/                     各項檢查與獨立重新評估證據
references/                     論文與固定版本的官方／上游來源
licenses/                       原始授權文字
docs/                           詳細說明、來源對照與失敗紀錄
finalize_results.py              比對兩組完整結果並產生報告
verify_final_evaluation.py       載入最終模型、重新測試全部 10000 張
run_remaining.py                 已授權續跑 ERM 後自動完成結果核對
prepare_new_run.py               建立乾淨的新實驗目錄，保護已發表結果
```

Git 儲存庫保存程式與可直接閱讀的實驗證據。大型 `.pt` 檔放在 [Release 完整實驗包](https://github.com/ChenBill900703/mixup-cifar10-reproduction/releases/tag/v1.0.0)；其中包含兩組 final weights、兩組 resume checkpoint、Quick Test checkpoint，以及同版程式／報表。下載包提供 SHA-256；它不包含 Python 安裝環境、CIFAR-10 原始資料或私人對話。環境可按 requirements 建立，資料由 torchvision 從官方來源下載。

若只按 GitHub 的「Download ZIP」，得到的是原始碼與小型證據，**不含 Release 裡的模型**。要完整保存本次實驗，請下載 Release 的 `mixup-reproduction-complete.zip`。

## 自己再跑一次之前

請先讀[操作說明](docs/reproduce_zh.md)。不要直接在已發表的 `outputs/ERM` 或 `outputs/Mixup` 覆寫新實驗；用 `prepare_new_run.py` 建立新目錄。正式程式保留來源、設定與環境不相容時拒絕 resume 的保護。

## 可以怎麼解讀這次結果？

本次固定 seed 的複現得到接近論文的結果，並觀察到 Mixup 的 final test error 低於 ERM。差距可能與未公開的 seed、歷史軟體版本、硬體、BN 初始化與非確定性 kernel 有關，但這份單次實驗**沒有證明哪個因素造成差距**。

這是可核對的單次複現，不是所有機器、所有 seed 都會得到同樣數字的保證。

## 論文、來源與授權

- Zhang, Cisse, Dauphin and Lopez-Paz. **mixup: Beyond Empirical Risk Minimization.** ICLR 2018. [arXiv:1710.09412v2](https://arxiv.org/abs/1710.09412v2)。
- [原作者 repository](https://github.com/facebookresearch/mixup-cifar10)，固定 revision `eaff31ab397a90fbc0a4aac71fb5311144b3608b`。
- [原作者引用的上游](https://github.com/kuangliu/pytorch-cifar)，核對 revision `49b7aa97b0c12fe0d4054e670403a16b6b834ddd`；不以目前上游模型取代官方版本。
- 官方 Mixup repository 的 CC BY-NC 4.0 及上游 MIT 文字保留於 [licenses/](licenses/)。這份專案沒有把第三方來源重新宣告成另一種授權；詳見 [來源與發布範圍](docs/publication_scope.md)。

維護者：ChenBill900703。實驗日期：2026-10-06。訓練數值保留原始觀測，不作人工美化。
