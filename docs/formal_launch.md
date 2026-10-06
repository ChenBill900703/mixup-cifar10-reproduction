# Formal Mixup launch amendment

2026-10-06 11:54 Asia/Taipei. User asked about the memory failure and explicitly requested direct execution. The assistant explained that fabricated experiment numbers would not be provided and proceeded with actual training only.

Preflight status PASS: validation/20261006T034926/readiness.json. Official code and all fixed scientific settings unchanged. Windows num_workers=0 was already tested and disclosed. Command: `python -u train.py --formal --approve-formal --method Mixup` using the local .venv. No ERM formal run launched. Launch process and log paths: outputs/Mixup/launch.json.

The post-test read-only snapshot reported 33,346,680 KiB OS-visible RAM and 9,167,756 KiB free (roughly 23 GiB in use, 8.7 GiB free). Windows reported 47,330,676 KiB total virtual memory and 9,354,384 KiB free. The automatically managed C:\pagefile.sys was 13,656 MiB allocated, 147 MiB current usage. GPU memory was 246/8192 MiB after testing. These readings do not reconstruct the failed 8-worker run's peak or prove physical RAM was exhausted; the retained error is WinError1455 during cuDNN DLL load. No OS page-file setting was modified.

Formal allocation now authorized: one single-GPU Mixup run, up to 200 full epochs, with epoch-boundary atomic checkpoints and fixed seed/settings. No hyperparameter/seed search and no fabricated or selected-best primary results. Full-run status and final results are authoritative only in outputs/Mixup/summary.json and resume_checkpoint.pt. This amendment supersedes the preflight-only authorization scope in the historical planning documents.
