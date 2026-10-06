# Preserved setup failure record

2026-10-06, RESOURCE: first CUDA wheel installation failed during download with `OSError: [Errno 28] No space left on device`, around 2.4/2.7 GB downloaded. It used default system temp/cache. No training had started.

Recovery: same torch 2.7.1 / torchvision 0.22.1 CUDA 12.6 target, set TEMP/TMP to project-local `.install-temp` on E drive and disable pip download cache. Version-pinned numerical/reporting dependencies were installed in the same isolated .venv. Retry succeeded. No scientific parameter was changed.

Initial Git network request from the sandbox could not connect to GitHub. The explicitly user-requested public clone succeeded with network permission; both official and referenced upstream revisions are retained. No credentials were requested.

Initial invariant test found an incorrect manually entered expected parameter count in the test (11,172,170). The actual unchanged official model has 11,171,274 parameters. Corrected the test oracle to compare directly to the separately imported official model, with the measured count as an additional check. No model code changed; failed log retained at validation/invariants_initial.log.

Sandbox writes to new output directories were denied; validation was rerun under normal user permissions. This is an execution-environment limitation, not a CUDA or training failure.

Validation `20261006T034729`, RESOURCE: actual 8-worker ERM quick test failed during Windows spawned-worker torch import with WinError 1455, unable to load cudnn_heuristic64_9.dll. No epoch was completed. Parent process remained waiting on workers, so this test process tree was terminated. Full traceback: validation/20261006T034729/ERM_epoch1.log. Fix: num_workers=0 on both quick/formal paths; same dataset, batch size, model, augmentation and optimizer. No OS paging settings changed. Fresh validation uses a new timestamp, and the failed output folder remains intact.

Formal ERM interruption, 2026-10-06 14:46 Asia/Taipei: after epoch142 was saved, the next epoch failed in DataLoader collate/torch.stack with DefaultCPUAllocator unable to allocate 1,572,864 bytes. This is a CPU allocation failure; it is not evidence of CUDA VRAM exhaustion. The root cause of the system memory pressure at that time is unknown. Original stderr and failed run record remain under outputs/ERM; the launcher failure state was archived before restart.

At 16:14, read-only inspection showed about 8.7 GiB free physical RAM and 8.5 GiB available commit. This later snapshot cannot reconstruct the failure-time peak. Before restarting, checkpoint epoch142 was loaded and verified: contiguous 142 complete epochs, full 50000/10000 counts, finite model tensors, source hashes unchanged, RNG digest intact, next LR=.01. At 16:16, resumed from epoch143. Changed only the non-training waiting launcher to avoid importing torch while waiting and to log Windows physical/commit memory every 30 seconds. Training source, scientific configuration, seed, batch size and checkpoint states were not changed. No other programs were closed and no OS page-file setting was modified.
