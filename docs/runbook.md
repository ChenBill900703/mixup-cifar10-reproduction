# Windows PowerShell runbook

## Authorized remaining work and automatic final report

On 2026-10-06 the user authorized the remaining ERM run after Mixup completed. The background launcher `run_remaining.py` executes ERM under the already validated formal settings, then calls `finalize_results.py` to audit both complete checkpoints, check all 200 history rows, verify final model weights and produce docs/reproduction_results.md plus validation/final_integrity.json. Live state is outputs/remaining_work.json; launcher logs are located by outputs/ERM/launch.json. COMPLETE means both training and the final audit succeeded; FAILED retains an error and all prior epoch checkpoints.

If this authorized continuation is interrupted, restart it with `.\.venv\Scripts\python.exe -u run_remaining.py`. ERM automatically resumes; no Mixup retraining occurs. To rebuild only the final report after both methods finish, run `.\.venv\Scripts\python.exe finalize_results.py`. The reporting scripts do not change training settings or choose best checkpoints.

Open PowerShell in `E:\mixup BEYOND EMPIRICAL RISK MINIMIZATION\mixup-reproduction`. No activation is needed: use the environment executable directly.

```powershell
Set-Location 'E:\mixup BEYOND EMPIRICAL RISK MINIMIZATION\mixup-reproduction'
.\.venv\Scripts\python.exe tests\validate.py
```

Settings live in config.json. Default RUN_ERM=False, RUN_MIXUP=True, QUICK_TEST=True, USE_AMP=False. `--formal` changes the effective QUICK_TEST to False without editing files; `--method ERM|Mixup|both` overrides selectors only. CLI avoids modifying config.json and invalidating validated source hashes.

To manually test resume (new quick-id to preserve older tests):

```powershell
.\.venv\Scripts\python.exe -u train.py --quick --method Mixup --quick-id manual_demo --stop-after-epoch 1
.\.venv\Scripts\python.exe -u train.py --quick --method Mixup --quick-id manual_demo
```

The first invocation prints FRESH RUN - STARTING EPOCH 1. The second prints RESUME CHECKPOINT FOUND, LAST COMPLETED EPOCH = 1, NEXT EPOCH = 2. Each quick epoch uses the complete model and actual formal data pipeline, capped at 3 training batches (384 images) and 2 test batches (200 images). Dataset files still have all 50,000/10,000 examples. These metrics must never enter the paper comparison.

## Formal Mixup — only after user approval

```powershell
.\.venv\Scripts\python.exe -u train.py --formal --approve-formal --method Mixup
```

This means RUN_ERM=False, RUN_MIXUP=True, QUICK_TEST=False, 200 complete epochs. The runtime requires a passing readiness file for the current source hashes. To resume, run exactly the same command. It restores all saved states automatically. If epoch 200 already completed, it reexports derived files and does not retrain. No special --resume flag or filename editing.

Formal ERM, only when separately requested:

```powershell
.\.venv\Scripts\python.exe -u train.py --formal --approve-formal --method ERM
```

outputs/ERM and outputs/Mixup are separate. Launching Mixup does not load or change ERM checkpoints. The root comparison file can reflect both methods, keeping unexecuted cells empty. Do not copy quick checkpoints into formal directories.

## Restore guarantees and limits

Checkpoint includes completed epoch, model, optimizer and momentum, next LR, Python/NumPy/torch CPU/all visible CUDA RNG, separate train/test DataLoader generators, full history, best diagnostic metric, config, source hashes, environment and immutable run ID. Writes use same-directory temporary file, flush/fsync, then os.replace. There is never an intentional deletion of the last good checkpoint before replacement. Atomic replacement reduces partial-write risk; it cannot guarantee survival of all hardware/storage failures. Keep an external backup if needed.

An OS file lock prevents concurrent writes to a method directory. Lock automatically releases when the process exits, even after a crash; the small .run.lock file may remain. Config or source differences are refused; framework/GPU environment differences are refused. Never bypass checks to force an incompatible resume. Keep trusted checkpoints only: loading optimizer/NumPy states requires torch.load(weights_only=False).

If interrupted mid-epoch, repeat the command: the incomplete epoch is replayed. CSV/summary are derived from checkpoint history, preventing duplicated rows. RNG restoration is checked by digest before training resumes. cuDNN nondeterminism still prevents a cross-process bitwise guarantee.

## Rebuild on another Windows machine

Install a compatible Python 3.12 separately if absent; current .venv is local and not portable. Then:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/mixup-cifar10.git references/official-mixup-cifar10
git -C references/official-mixup-cifar10 checkout eaff31ab397a90fbc0a4aac71fb5311144b3608b
.\.venv\Scripts\python.exe -u train.py --quick --method Mixup --quick-id initial_download --download --stop-after-epoch 1
.\.venv\Scripts\python.exe tests\validate.py
```

requirements-lock.txt records all locally resolved package versions. Official model SHA equality is enforced. Revalidate on the new machine. Keep licenses/ and attribution when sharing source; do not upload datasets, environment, checkpoints or private logs accidentally.

## Outputs after formal completion

Per method: config.json, environment.json, dataset_provenance.json, source_boundary.md, assumptions.md, training_history.csv, summary.json, paper_comparison.csv, test_accuracy_curve.png, test_error_curve.png, training_loss_curve.png, paper_vs_reproduction.png, final_model_state_dict.pt, resume_checkpoint.pt, README.md and invocation records. Root outputs/paper_comparison.csv combines methods. Best diagnostic results never replace final weights.

On a setup/test failure, retain the timestamped log and readiness=FAIL, fix the verified cause, then rerun validation into a new timestamp folder. Do not begin formal training until readiness=PASS and user confirmation.
