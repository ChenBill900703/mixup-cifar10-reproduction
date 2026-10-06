# Unknowns and precommitted implementation assumptions

- **Paper seed: UNKNOWN.** Fixed project seed 20170922 is taken from the official README example, not claimed to be the paper seed. No seed selection based on test accuracy.
- Original PyTorch, torchvision, CUDA, cuDNN, compiler and complete machine image: UNKNOWN. Use Windows Python 3.12.14, PyTorch 2.7.1+cu126 and torchvision 0.22.1+cu126. Exact observed details are in environment.json and requirements-lock.txt.
- Framework defaults are version dependent. In particular historical BN gamma defaults differ; we keep the official model file unchanged and explicitly use the chosen runtime defaults. We do not claim historical initialization was reconstructed.
- Figure 3(a) final-versus-best rule: UNKNOWN. The user-selected primary endpoint is epoch 200, with the last-epoch checkpoint always exported. Table 5 uses another estimator (last-10 median) and is not the primary target.
- The paper-time official commit and upstream commit are UNKNOWN. Both inspected clones are pinned and their file hashes retained.
- DataLoader generators and explicit worker seeds are added for restart control. Workers=8 is retained, persistent_workers=False; workers restart each epoch from the saved generator state. Prefetched work in an interrupted epoch is discarded, and the entire incomplete epoch is replayed.
- cuDNN benchmark=True follows the official code. Deterministic algorithms are not forced. Seeds, checkpoints and saved RNG control random streams, but kernel nondeterminism and library versions limit bitwise reproducibility.
- TF32=False and AMP=False make the intended arithmetic FP32 on Ampere. The original P100 did not support TF32; no speed-oriented precision substitution.
- Reporting fixes the official loss denominator bug and uses sample-weighted losses. It changes reporting, not the minibatch losses used for backward.
- The user authorized setup and capped Quick Tests only. Formal training remains blocked until explicit confirmation; no 200-epoch result currently exists.
- Official LR boundary is followed: epochs 1–101=.1, 102–151=.01, 152–200=.001. This is a documented code-versus-prose difference, not hyperparameter tuning.

If a final gap is observed, seed uncertainty, P100 vs RTX 3070 Ti, software changes, historical defaults and nondeterministic kernels are hypotheses. None may be declared causal without controlled evidence. Do not retune using the test set to approach 4.2%.
