# Precommitted experiment specification

Mode: COMPUTATIONAL_AI. Objective: reproduce the CIFAR-10 / official PreAct ResNet-18 ERM and mixup alpha=1 experiments in Section 3.2/Figure 3(a), retaining the paper/code source boundary.

Unit: one complete 200-epoch training run per method, seed 20170922. Image-level test correctness defines the endpoint; no confidence interval over seeds is implied by a single run. Comparator: ERM via the same official mixing path with alpha=0 (lambda=1); intervention: alpha=1. No ablation or tuning stage is authorized.

Primary outcome: 100 - full CIFAR-10 test accuracy at epoch 200. Secondary diagnostic: minimum test error and epoch. Fixed paper reference: ERM 5.6%, mixup 4.2%. Difference is reproduction minus paper in percentage points. Missing experiments stay NOT_EXECUTED; no zero or Quick Test placeholder is accepted as a result.

Fixed scientific settings: official source architecture; 200 epochs; train batch 128/test batch 100; SGD .9 momentum, 1e-4 decay on all parameters; LR .1/.01/.001 with official boundaries 102/152 (human epoch); no dropout; FP32; one GPU; crop 32, constant pad 4, random horizontal flip .5, official normalization. See source_boundary.md for each provenance label.

Data: official CIFAR-10 Python archive, MD5 `c58f30108f718f92721af3b95e74349a`; 50,000 train/10,000 test, file-level torchvision integrity checks. Test labels are used for the fixed evaluation only; no test-driven configuration or seed changes.

Hardware USER_REPORTED: RTX 3070 Ti 8 GB, i7-11700K, 32 GB RAM, SSD, Windows. Read-only scan matched GPU model/VRAM. Authorized allocation: local setup and short validation; formal 200-epoch wall time not authorized. Formal duration estimate remains UNKNOWN until representative full-epoch timing; capped Quick Tests are not reliable throughput estimates due to initialization and cuDNN algorithm selection overhead. No paid compute. Windows DataLoader workers=0 after the documented 8-worker resource failure.

Storage planning estimates: CUDA environment/download peak roughly 12–20 GB, CIFAR archive+expanded roughly .35 GB, each model ~45 MB FP32 plus ~45 MB SGD momentum; atomic save temporarily needs two checkpoint files. E drive used for download staging following a system-temp disk-space failure. These are planning estimates, not measured resource results.

Acceptance: source audit; official model equality; loss+gradient equivalence; all 200 LR values match executed official schedule; syntax/import; actual CUDA forward/backward; real-data ERM and Mixup 2-epoch capped runs with epoch-1 stop and separate-process resume to epoch 2; exact RNG digest restoration; optimizer momentum present; config mismatch rejection; injected atomic-save failure retains previous checkpoint; completed run does not retrain. Required checks are automated in tests/validate.py.

Stopping: non-finite loss, unavailable CUDA, incompatible checkpoint/config/environment, changed model/source, failed validation, or user interruption. Mid-epoch interruption resumes from the last completed epoch. Preserve failed logs. Formal runs require --formal --approve-formal after the user approves; changing protected scientific settings requires a new audited protocol.

Evidence: validation timestamp folders and outputs/quick contain EXECUTED_LOCAL code-path tests only; outputs/ERM and outputs/Mixup are reserved for formal runs. Result claims require status COMPLETE, final_epoch=200 and result_claim_allowed=true.
