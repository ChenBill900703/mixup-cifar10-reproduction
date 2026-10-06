# 如何下載、核對與重新跑一次

請先分清楚兩種目的：**核對本次已完成模型**，或**自己重新訓練一組新結果**。不要把新的訓練寫進本次已發表的outputs資料夾。

## A. 只看報告、紀錄與圖

直接瀏覽GitHub即可。兩組各有200輪CSV、summary、環境、原始log、resume證據，以及accuracy/error/loss曲線。

GitHub一般的「Download ZIP」或`git clone`會取得程式與小型結果檔，**不會取得Release中的大型.pt檔**。

## B. 下載含模型的完整實驗包

到[v1.0.0 Release](https://github.com/ChenBill900703/mixup-cifar10-reproduction/releases/tag/v1.0.0)，下載：

1. `mixup-reproduction-complete.zip`：程式、文件、所有發布證據、兩組正式weights/checkpoint，以及Quick Test weights/checkpoint。
2. `SHA256SUMS.txt`：下載包的SHA-256。

PowerShell檢查：

```powershell
Get-FileHash .\mixup-reproduction-complete.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

兩者ZIP雜湊應一致。解壓縮後，`ARTIFACT_MANIFEST.json`列出每個發布檔案的相對路徑、大小與SHA-256；`publication_manifest.json`專門記錄模型原檔的hash及所在位置。SHA-256相同表示下載檔案與發布時的位元組相同，不代表科學結論自動成立；科學核對另有測試證據。

完整包不附CIFAR-10原始資料或Python虛擬環境；它們可按下面步驟從官方來源取得。沒有附私人對話。

## C. 用最終模型重新測試10,000張圖片

以下在完整包解壓縮後的專案根目錄執行。需要可用的NVIDIA CUDA環境及Python3.12；本次觀測使用3.12.14，但不代表其他版本一定會逐位元相同。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True); CIFAR10('data', train=False, download=True)"
.\.venv\Scripts\python.exe verify_final_evaluation.py
```

這只做inference，不訓練。腳本會核對結果是否等於本次報告，並保存`validation/independent_evaluation.json`。執行會更新這份核對檔，若要保留下載版原始證據，先保留完整ZIP或另用副本操作。

本次實際得到ERM9440/10000正確、Mixup9573/10000正確。不同GPU／套件可能存在浮點與kernel差異，腳本遇到不一致會報錯，不會把報告數字偷偷改成新的值。

## D. 在新目錄重新訓練，保護已發表結果

在clone或解壓縮後的專案根目錄執行：

```powershell
py -3.12 prepare_new_run.py --destination ..\mixup-new-run
Set-Location ..\mixup-new-run
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True); CIFAR10('data', train=False, download=True)"
.\.venv\Scripts\python.exe tests\validate.py
```

helper只複製科學程式、設定、參考來源、文件與授權，不複製既有outputs或readiness通過狀態。若目的資料夾已存在，會拒絕操作。複製的docs包含本次历史報告，不能誤當成新實驗結果；新結果由新目錄的outputs產生。

Quick Test全部PASS後，再明確啟動正式實驗：

```powershell
# 只跑Mixup
.\.venv\Scripts\python.exe -u train.py --formal --approve-formal --method Mixup

# 或：只跑ERM
.\.venv\Scripts\python.exe -u train.py --formal --approve-formal --method ERM

# 或：依序跑兩组，不是同時占用GPU
.\.venv\Scripts\python.exe -u train.py --formal --approve-formal --method both
```

上面三個選一個符合目的的命令，不必全部執行。預設config.json是QUICK_TEST=True；`--formal`會在有效設定中將它改為False，因此不必手動改十幾個地方。

正式程式會檢查CUDA、來源模型hash、驗證報告、設定與環境。來源變動會使舊readiness失效，必須重新驗證。不要为了讓舊檔案「通過」而手動改hash或刪掉不相容檢查。

這個發布版本的官方來源已放在`references/official-mixup-cifar10/`，是固定revision的檔案快照，不帶Git內部歷史；訓練及測試所需的來源對照不需要再clone一次。若另行clone，要checkout文件指定revision。

## E. 如果中斷，要如何resume？

在**同一個新實驗目錄、同一套環境與設定**重複原本的正式命令即可。程式會自動找checkpoint：

```text
RESUME CHECKPOINT FOUND
LAST COMPLETED EPOCH = 142
NEXT EPOCH = 143
```

完成的epoch保留，沒有完成的那一輪會重做。每輪先完整train/test，再存checkpoint。已完成200輪時不會從頭重訓。

公開包裡的checkpoint是本次已完成200輪的checkpoint，用於保存與核對。它不是「換一台電腦一定能直接從中間繼續」的承諾；原程式會拒絕關鍵環境差異。只想評估請用final weights，想新跑請建立乾淨目錄。

## F. 訓練完成，如何產生總比較？

兩組新實驗都完成後：

```powershell
.\.venv\Scripts\python.exe finalize_results.py
.\.venv\Scripts\python.exe verify_final_evaluation.py
```

第一個命令會檢查200輪歷史、LR、模型檔、兩組設定與資料量，生成比較；第二個命令重新載入模型評估完整test set。兩個命令都不會重訓，也不會改成使用best checkpoint。

## G. 紀錄中的負零與重現限制

原始CSV/JSON保留Python浮點值，因此可能看到很小的數字例如`-5.3e-15`。這不代表實際有可解讀的負差距；兩位小數顯示為0.00即可，保留raw數值有助核對。

本次軟體完整版本可查requirements-lock.txt和environment.json。但套件鎖定也無法補回原論文沒公布的paper seed或歷史framework設定，更無法保證所有平台bitwise一致。詳見assumptions與source_boundary。
