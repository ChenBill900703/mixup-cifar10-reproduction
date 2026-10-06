# 發布範圍與來源保留

本發布是已完成實驗的獨立副本，沒有修改原訓練工作目錄。公開前比對validation/readiness.json中所有training source hashes，保證train.py、config.json、models/、utils/及原有tests/與實際訓練版本一致。

## Git儲存庫

包含訓練／續跑／驗證程式、固定設定、requirements、官方與上游檔案快照、論文參考、原始授權、兩組完整epoch歷史、summary、圖表、診斷日誌、Quick Test紀錄、失敗與恢復證據，以及繁體中文教學文件。

原始技術日誌和checkpoint可能包含當時的本機路徑、版本、硬體資訊與程序識別碼；保留這些技術metadata是為了維持原始實驗證據。沒有收錄GitHub憑證或私人對話。

preflight計畫、較早的狀態與assumptions文件是歷史快照。最終完成狀態請看outputs各組summary、validation/final_integrity.json及validation/independent_evaluation.json。

## Release

`mixup-reproduction-complete.zip`包含同版發布副本，加上原始.pt檔：正式ERM/Mixup各自的final_model_state_dict.pt、resume_checkpoint.pt，以及Quick Test保存的模型。模型檔逐位元複製，沒有重新初始化、重新訓練、挑選其他epoch或重新序列化。

訓練資料的官方download URL與MD5記錄在dataset_provenance.json；資料集本體可由torchvision重新下載。完整包不收錄.venv、下載快取、CIFAR-10本體、.git內部歷史、私人對話或內部任務分派文件。

## 雜湊與修改範圍

- `publication_manifest.json`：原始訓練source hash與模型hash。
- `ARTIFACT_MANIFEST.json`：發布副本每個檔案的大小與hash；在完整ZIP中也涵蓋.pt。
- Release的`SHA256SUMS.txt`：完整ZIP的hash。
- `.gitattributes`停用自動換行轉換，避免checkout時改動已封存科學程式的位元組。

README及新增白話文件是發布時整理的說明；它們不是另一輪訓練。prepare_new_run.py僅建立乾淨目錄，不改動原科學設定。

## 授權與引用

官方Mixup repository的授權文字保留於`licenses/LICENSE-official-mixup`；引用上游pytorch-cifar的授權保留於`licenses/LICENSE-pytorch-cifar`。官方與上游原有授權／註解也留在references快照中。論文與資料集歸原作者，引用時請參考README列出的原始來源。

本發布沒有宣稱第三方內容全部改成MIT，也沒有擅自替原作者擴張商業使用權限。使用、修改或再散布時，應依各來源的原授權處理。
