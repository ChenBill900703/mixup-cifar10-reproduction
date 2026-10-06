# Source audit: PAPER / OFFICIAL CODE / ASSUMPTION

Audit date: 2026-10-06. Primary source is the supplied full 13-page arXiv:1710.09412v2 PDF (27 Apr 2018). All pages were extracted; the relevant page 5 figure and section were also rendered and inspected. Scope: Section 2 (p.3), Section 3.2 and Figure 3(a) (p.5), Section 3.8/Table 5 (pp.9-10), reference Liu (2017) (p.12).

Official clone: https://github.com/facebookresearch/mixup-cifar10

Pinned official HEAD: `eaff31ab397a90fbc0a4aac71fb5311144b3608b` (2018-09-08). This is the actual cloned default-branch HEAD, not an inferred paper-time revision. The repository has other remote history; do not equate its archival date with the pinned code date.

Referenced upstream clone: https://github.com/kuangliu/pytorch-cifar ; inspected HEAD `49b7aa97b0c12fe0d4054e670403a16b6b834ddd`. Its current `models/preact_resnet.py` is comparison evidence, not the implementation to substitute. The paper does not pin an upstream commit.

## Decision matrix

| Detail | PAPER | OFFICIAL CODE | Project decision / evidence class |
|---|---|---|---|
| CIFAR-10 PreAct ResNet-18 | p.5 Section 3.2, cites Liu | `models/resnet.py:ResNet18` builds `ResNet(PreActBlock,[2,2,2,2])` | Preserve official file byte-for-byte; never torchvision ResNet18 |
| Stem | Exact operators not enumerated | 3x3 3→64, stride 1, pad 1, bias False; BN; ReLU; no max pool | OFFICIAL-CODE-SPECIFIC |
| Residual stages | PreAct ResNet-18 | widths 64/128/256/512; blocks 2/2/2/2; first strides 1/2/2/2 | OFFICIAL-CODE-SPECIFIC |
| Residual block | Reference only | BN(input)→ReLU→3x3 conv→BN→ReLU→3x3 conv; addition; no post-add ReLU | OFFICIAL-CODE-SPECIFIC |
| Shortcut | Reference only | Always `shortcut(ReLU(BN(x)))`; empty Sequential when unchanged, otherwise 1x1 projection without BN | OFFICIAL-CODE-SPECIFIC; preserve even the preactivated identity |
| Classifier | Not enumerated | fixed 4x4 average pool, flatten, Linear(512,10), bias True; no final extra BN | OFFICIAL-CODE-SPECIFIC |
| Epochs / train batch | 200 / 128 | 200 / 128 | CONFIRMED |
| Optimizer | Not stated for CIFAR in Section 3.2 | SGD(momentum=.9, weight_decay=1e-4); no Nesterov or dampening specified | SGD/momentum OFFICIAL-CODE-SPECIFIC; runtime defaults nesterov=False, dampening=0 explicitly recorded |
| Weight decay | 1e-4 | 1e-4 applied to all parameters | CONFIRMED; no parameter exclusions |
| LR | Start .1, divide by 10 after 100/150 epochs | zero-based `adjust_learning_rate` is called AFTER each epoch | Follow literal code as explicitly authorized for LR boundary; see below |
| Mixup | Section 2: Beta(alpha,alpha), convex input/label combinations, shuffled same minibatch; Section 3.2 alpha=1 | One NumPy Beta scalar per minibatch; CPU randperm then CUDA; two weighted cross-entropies | CONFIRMED; mathematically equivalent soft-label CE, no one-hot allocation needed |
| ERM | Comparator | alpha≤0 gives lambda=1 but still generates permutation and calls weighted loss | Retain alpha=0 official path, including RNG consumption |
| Crop | Not specified for CIFAR | RandomCrop(32,padding=4) | OFFICIAL-CODE-SPECIFIC; padding default constant zero |
| Flip | Not specified for CIFAR | RandomHorizontalFlip() | OFFICIAL-CODE-SPECIFIC; p=.5 |
| Normalization | Not specified for CIFAR | mean (.4914,.4822,.4465), std (.2023,.1994,.2010), after ToTensor | OFFICIAL-CODE-SPECIFIC |
| Test transform | Evaluated on test set | ToTensor+same normalization; no random augmentation | OFFICIAL-CODE-SPECIFIC |
| Test loader | Not specified | batch 100, shuffle=False, workers=8 | batch/shuffle retained; workers=0 Windows adaptation after WinError 1455 |
| Train loader | Batch 128 | shuffle=True, workers=8, no drop_last/pin_memory arguments | workers=0 after observed Windows commit-memory failure; drop_last=False, pin_memory=False; explicit generators added for resume |
| Dropout | None | None on this model path | CONFIRMED |
| GPU | Single Tesla P100, PyTorch | CUDA, DataParallel, cuDNN benchmark=True | Single user-reported RTX 3070 Ti; no one-GPU wrapper (same computation); benchmark=True retained |
| Initialization | Not specified | No custom initialization: Conv/Linear/BN framework defaults | ASSUMPTION: use pinned runtime defaults; historical version UNKNOWN, see initialization caveat |
| BatchNorm | Not specified | `nn.BatchNorm2d(...)` defaults | eps=1e-5, momentum=.1, affine=True, track_running_stats=True in pinned runtime; no tuning |
| Seed | Not published | default 0 skips explicit seed; README example 20170922; only torch.manual_seed called | Paper seed: UNKNOWN. Precommit 20170922 from README, now seed Python/NumPy/torch/CUDA/workers too; ASSUMPTION |
| Evaluation | Figure 3(a): 5.6 / 4.2; selection rule not explicit | eval mode, correct/total; saves best and last | Primary = final epoch 200 by USER REQUIREMENT; best only diagnostic |
| Table 5 | Median error of last 10 epochs: ERM 5.53 / mixup 4.24 at wd=1e-4 | Not separately implemented | Do not conflate with Figure 3(a) or rename median as final |
| Loss logging | Not specified | returns sum of batch means divided by last zero-based batch index (off by one) | Correct reporting to sample-weighted mean; training gradients unchanged; disclosed implementation repair |
| Resume | Not specified | partial state; no optimizer/NumPy/CUDA states | Added robust checkpoint, state restore, atomic replacement and incompatibility checks per USER REQUIREMENT |

## LR boundary: explicit conflict resolution

PAPER: naturally read as 1–100=.1, 101–150=.01, 151–200=.001.

OFFICIAL CODE: train/test run first; `adjust_learning_rate(optimizer, epoch)` then checks `epoch >=100` and `>=150`; loop indexes start at zero. Training with index 100 (human epoch 101) still uses .1. The adjustment affects index 101 (human epoch 102).

ASSUMPTION / resolution: no hidden scheduler correction. Follow official semantics under the user's explicit exception for boundary differences. The reproduced schedule is **1–101=.1, 102–151=.01, 152–200=.001**. Tests execute the extracted official function for all 200 epochs and compare against the project mapping.

| Human epoch | Paper reading | Official / selected |
|---:|---:|---:|
| 100 | .1 | .1 |
| 101 | .01 | .1 |
| 102 | .01 | .01 |
| 150 | .01 | .01 |
| 151 | .001 | .01 |
| 152 | .001 | .001 |

## Initialization and software-era caveat

The authors did not pin a PyTorch/torchvision version. Unmodified model source does not prove identical historical initialization. In PyTorch v0.3.1, BatchNorm gamma used `uniform_()`; in our pinned PyTorch 2.7.1 it is ones. See the primary historical source: https://github.com/pytorch/pytorch/blob/v0.3.1/torch/nn/modules/batchnorm.py . We do not assert the paper used v0.3.1. This project explicitly chooses unmodified official constructors under 2.7.1 instead of inventing a paper-time version. This is a known reproduction limitation, not an exact environment recreation.

Conv2d initialization in the selected runtime is Kaiming uniform with a=sqrt(5), equivalent distribution to uniform ±1/sqrt(fan_in); conv biases are disabled. Linear weight/bias use ±1/sqrt(fan_in). BN gamma=1, beta=0, running_mean=0, running_var=1. Initializer draws and augmentation RNG implementation can differ across framework versions. No additional custom initialization is applied.

## Modern compatibility adaptations

Replace deprecated Variable/volatile and `.data[0]` with normal tensors, no_grad and item; explicit FP32, disable AMP and both TF32 backends; no compile, cosine, warmup, label smoothing or added augmentation. SGD foreach/fused are explicitly False. The architecture, loss, optimizer equation and minibatch size are preserved. CUDNN benchmark may select nondeterministic kernels; restoring RNG is not a promise of bit-identical results across processes/devices.

The first actual 8-worker test failed while spawned Windows processes loaded cuDNN DLLs (`WinError 1455`, paging-file/commit-memory limit). Retry uses num_workers=0 for both quick and formal paths; no paging-file/OS setting was modified. Transforms and sample counts are unchanged, but random augmentation draws now use the main CPU RNG stream. This is a declared implementation assumption and can affect a fixed-seed trajectory. It was chosen to resolve a resource failure before any valid test accuracy existed, not to tune results.

Source priority: supplied PDF > official implementation > referenced upstream > disclosed assumptions; explicit user exceptions (LR semantics and final metric) apply. Code-specific choices above are not falsely attributed to the paper.
