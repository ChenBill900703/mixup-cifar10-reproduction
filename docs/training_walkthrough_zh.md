# 這次實驗到底怎麼跑？完整白話紀錄

本文把原始日誌、設定與核對資料串起來。所有正式結果來自實際訓練；遇到的錯誤也保留。時間均為 **2026-10-06，Asia/Taipei**，除非檔案明寫 UTC。

## 一、先決定要比什麼，而不是先追求漂亮數字

我們要比較兩件事：

1. ERM：照一般方式，以原始圖片與它的標籤訓練。
2. Mixup：把同批的兩張圖片及其標籤按比例混合後訓練。

兩組用同一個 CIFAR-10 資料集、同一份官方 PreAct ResNet-18、同一個 seed 與 200 epochs。最重要的不同是 alpha：ERM=0，Mixup=1。

在訓練前就決定：最後拿 **epoch200 的錯誤率**比論文。即使 epoch173 比 epoch200 好，也不能因為比較漂亮就把173拿來當正式結果。

## 二、讀論文，再真的看官方程式

論文第5頁 Section3.2 說明 CIFAR 實驗，Figure3(a) 給出 ERM 5.6%、Mixup 4.2%。我們實際 clone 官方 repository，固定 commit，逐項看模型、資料處理、optimizer、loss 和 LR。

這一步發現兩個很容易憑印象做錯的地方：

- 名字雖然叫 ResNet18，但官方這份實作使用 PreActBlock。它的 stem、shortcut 細節與其他常見版本不同，不能直接換 torchvision ResNet18。
- 官方 LR 是在一輪結束後才更新，epoch 又從0起算，於是從1起算時，第102輪才開始用0.01，第152輪才開始用0.001。

我們把這些列在來源對照表，保留原碼，並執行官方 LR 函式，確認本專案每一輪的 LR 都吻合。不是因為結果好壞才決定選哪個版本。

## 三、建立可用的本機環境

先確認電腦真的看到 RTX3070Ti，再建立獨立 Python 環境，安裝固定 PyTorch／torchvision 版本。`nvidia-smi` 顯示的 driver 支援上限與 PyTorch 自帶的 CUDA runtime 是兩回事；本次 PyTorch runtime 是12.6。

安裝時第一次下載約2.7GB的 CUDA wheel，系統暫存空間不夠而失敗。後來把下載暫存放在 E 槽，停用下載快取，再完成安裝。這只是安裝位置調整，沒有改模型或超參數。

正式訓練之前，實際做 CUDA tensor 的 forward/backward，確認不是只有「看得到 GPU 名字」，而是計算真的能在 GPU 上執行。

## 四、資料怎麼處理？

使用官方 CIFAR-10 Python archive。資料檔經 torchvision 的 MD5 完整性檢查，確認 train=50,000、test=10,000。

每張 train 圖片是32×32彩色圖片：

1. 四周先補4格，再隨機裁成32×32。
2. 有一半機率左右翻轉。
3. 轉成模型可處理的 tensor。
4. 使用官方指定 mean/std 做 normalization。

Test 圖片不做隨機裁切或翻轉，只轉 tensor 並 normalization。這讓每輪評估使用同樣的測試資料規則。沒有新增 RandAugment、CutMix、ColorJitter、Random Erasing 等方法。

一輪完整 train 有391批，最後一批80張；不是把不足128張的尾批丟掉。每次測試有100批、每批100張，合計完整10,000張。

## 五、Mixup 每一批實際做什麼？

假設一批有128張圖片。程式先抽出一個 λ，再把128個位置隨機打亂。例如原本第1張與打亂後第1張配對，原本第2張與打亂後第2張配對，依此類推。

新的輸入是 `λ*A + (1-λ)*B`。如果 λ=.7，便是70% A、30% B。標籤的要求也相同：不是硬判A，而是讓loss同時考慮A和B，權重分別是.7和.3。

官方程式不需要真的建立一個完整 one-hot 混合向量，而是計算兩個 cross-entropy 的加權和。單元測試核對過：它與混合 one-hot 標籤的cross-entropy，在loss和梯度上都一致。

ERM 的 alpha=0，令 λ=1，等於只看原圖與原標籤。為保留官方路徑，仍保留相同函式與 permutation 呼叫，沒有另換一套 optimizer 或模型。

## 六、先短測試，再正式200輪

正式跑之前，兩組都做了2輪 Quick Test。用的是真實 CIFAR-10、完整模型、batch128與正式 transforms，只把每輪縮短成3個train batches、2個test batches。

所以每個短測試 epoch 只用384張train、200張test。這些accuracy/error只用來確認程式路徑，**沒有拿來宣稱複現成果**。

測試刻意做以下流程：

1. 從epoch1開始，完成後存checkpoint並停止程序。
2. 重新啟動另一個程序，顯示最後完成epoch1、下一輪epoch2。
3. 恢復model、optimizer momentum、LR與全部RNG，核對亂數狀態digest。
4. 跑完epoch2，確認沒有重複epoch1。
5. 故意用不同seed嘗試resume，程式必須拒絕，而且不能覆寫原checkpoint。
6. 在寫檔過程注入失敗，確認上一份checkpoint仍完整可讀。

我們也核對了官方模型forward/backward、loss梯度與200輪LR，全部通過才開始正式訓練。

## 七、第一個記憶體問題：8個workers載入DLL失敗

官方原程式使用8個DataLoader workers。Windows 啟動這些子程序時，實際發生 WinError1455，無法載入cuDNN相關DLL。它指向記憶體提交／分頁檔限制，不等於已證明8GB GPU顯存被用完。

處理方式是把 workers 改為0，也就是讓主程序負責讀取、整理圖片。**batch128沒有變，模型、augmentation、optimizer與FP32也沒有變。** 沒有修改Windows分頁檔設定。

要誠實說明的是：雖然圖像處理方法相同，workers改變會影響亂數的消耗順序，所以不能說數值軌跡與原本8workers逐位元完全一樣。這個調整是在資源失敗後、正式結果出來前做的，不是用來追求4.2%。

## 八、正式Mixup：11:54開始，13:29完成

正式Mixup在11:54啟動，200輪的train與test都使用完整資料。每轮約28–32秒，最後累計train/test計時為5689.39秒，約94.8分鐘。

訓練並不是每一輪都單調變好。權重更新有隨機性，test error上下浮動是可能發生的；評估指標在每輪記錄，但不根據它更換seed或參數。

Mixup最後三輪是：

| Epoch | LR | Test accuracy | Test error |
|---:|---:|---:|---:|
| 198 | .001 | 95.85% | 4.15% |
| 199 | .001 | 95.90% | 4.10% |
| 200 | .001 | 95.73% | **4.27%** |

中途best是epoch173的3.96%。但正式規則早已固定，所以報告使用4.27%，而不是3.96%、4.10%或4.15%。

## 九、正式ERM：真的中斷一次，靠checkpoint接回來

ERM在13:35開始。14:46前，已完成並保存epoch142；下一輪組合batch時，CPU allocator無法配置1,572,864 bytes，約1.5MiB，程式退出。

「連1.5MiB都配不到」不代表模型只需要1.5MiB，也不代表我們掌握了所有當時的記憶體使用狀態。錯誤只能證明那次CPU記憶體配置失敗；當時系統壓力的根本原因未被完整量測，不能斷言是哪個程式造成。

16:14檢查時，實體RAM又有約8.7GiB可用。這個較晚的快照不能反推14:46的峰值。處理時沒有關閉使用者的其他程式，也沒有更改分頁檔。

恢復前先檢查：

- checkpoint能載入，最後完成的epoch確實是142。
- history有1到142連續紀錄，每輪資料量完整。
- 模型tensor是有限數值，没有NaN或Inf。
- 訓練原碼與設定的hash沒變。
- RNG digest完整，下一輪LR仍是.01。

另外把背景「等待訓練結束」的輔助程式改成不預先載入PyTorch，減少它自己的記憶體占用，並加入每30秒的系統記憶體記錄。**真正訓練程式及科學參數沒有改。**

16:16重新啟動時，畫面清楚顯示：

```text
RESUME CHECKPOINT FOUND
LAST COMPLETED EPOCH = 142
NEXT EPOCH = 143
```

接著恢復optimizer與所有RNG，核對digest相同，從epoch143一路跑到200。16:44完成訓練與自動核對。累計train/test約5891.16秒，98.2分鐘；中間等待時間不算在這個數字裡。

ERM最後三輪是：

| Epoch | LR | Test accuracy | Test error |
|---:|---:|---:|---:|
| 198 | .001 | 94.47% | 5.53% |
| 199 | .001 | 94.49% | 5.51% |
| 200 | .001 | 94.40% | **5.60%** |

中途best是epoch148的5.43%，同樣沒有用來替代final。

## 十、checkpoint到底存了什麼？

不是只存一個「模型圖片」。它包括：最後完成的epoch、模型參數、optimizer和momentum、下一輪LR、Python/NumPy/torch CPU/CUDA亂數狀態、train/test DataLoader generator、完整history、best診斷資訊、config、environment和來源hash。

寫檔先寫同資料夾的temporary file，flush/fsync後，再用原子replace取代正式checkpoint。這樣在寫新檔失敗時，比直接覆寫唯一檔案更容易保留上一個完整epoch。

它不是對任何斷電或硬碟故障的絕對保證，也不承諾不同GPU／cuDNN版本能逐位元相同。這次有實際證據證明：epoch142的checkpoint被成功用來恢復，完整history最後為200輪。

## 十一、完成後，再用模型真的考一次

自動核對先確認CSV、summary、checkpoint互相吻合，最終weights與epoch200 checkpoint tensor一致，兩組設定只有method/alpha不同。

17:02再獨立重新載入兩個final_model_state_dict.pt，對完整10,000張test images評估，没有做任何參數更新：

- ERM：9,440張正確、560張錯誤，error=5.60%。
- Mixup：9,573張正確、427張錯誤，error=4.27%。

這與訓練紀錄完全一致，因此結果不只是手動填在表格裡，而是保存的模型本身可以重算得到。

## 十二、要怎麼向老師解釋？

可以說：「我們以固定seed，在已公開說明的Windows與現代PyTorch環境下，使用原作者模型與主要訓練設定，完成ERM和Mixup各200輪。Final test error分別為5.60%與4.27%，與論文5.6%及4.2%接近。本次觀察到Mixup比ERM低1.33個百分點，且已用最終模型重新評估完整測試集驗證。」

同時要補充：這是一個seed，原作者歷史環境和paper seed未完整公開，LR的官方實際邊界、workers調整與BN初始化版本差異都已記錄。不能把「結果接近」說成所有細節完全一致，也不能說已證明哪個因素導致0.07個百分點差距。

## 證據入口

- [完整結果報告](reproduction_results.md)
- [來源逐項核對](source_boundary.md)
- [環境／實作assumptions](assumptions.md)
- [ERM原始紀錄與stderr](../outputs/ERM/)
- [Mixup原始紀錄](../outputs/Mixup/)
- [200輪完整性核對](../validation/final_integrity.json)
- [模型重新評估證據](../validation/independent_evaluation.json)
- [ERM epoch142的RNG恢復證據](../outputs/ERM/resume_verification_epoch_142.json)

這些檔案中較早的preflight計畫／狀態是歷史快照；最終完成狀態以`final_integrity.json`、各組`summary.json`和本結果報告為準。
