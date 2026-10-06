"""Local trusted checkpoints, atomic replacement, and complete RNG state."""
import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision


def atomic_write(path, writer, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb' if binary else 'w', **({} if binary else {'encoding': 'utf-8', 'newline': ''})) as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path, value):
    atomic_write(path, lambda f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False), False)


def save_checkpoint(path, value):
    atomic_write(path, lambda f: torch.save(value, f))


def capture_rng(generators):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
            'dataloader': {k: g.get_state() for k, g in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    for k, g in generators.items():
        g.set_state(state['dataloader'][k])


def rng_digest(state):
    h = hashlib.sha256()
    h.update(repr(state['python']).encode())
    n = state['numpy']
    h.update(repr((n[0], n[2:])).encode())
    h.update(n[1].tobytes())
    for t in [state['torch_cpu'], *state['torch_cuda'], *state['dataloader'].values()]:
        h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_hashes(root):
    paths = [root / 'train.py', root / 'config.json']
    for folder in ('models', 'utils', 'tests'):
        paths.extend(sorted((root / folder).glob('*.py')))
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths}


def environment():
    available = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0) if available else None
    try:
        smi = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        smi = str(exc)
    return {'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
            'pytorch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'cuda_available': available,
            'gpu': props.name if props else None, 'gpu_vram_bytes': props.total_memory if props else None,
            'visible_gpu_count': torch.cuda.device_count(), 'nvidia_smi': smi,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32}


def validate_resume(checkpoint, config, env):
    if checkpoint['config'] != config:
        keys = sorted(k for k in set(config) | set(checkpoint['config'])
                      if config.get(k) != checkpoint['config'].get(k))
        raise ValueError('REFUSING RESUME: config mismatch: ' + ', '.join(keys))
    for key in ('python', 'pytorch', 'torchvision', 'numpy', 'cuda_runtime', 'cudnn', 'gpu'):
        if checkpoint['environment'][key] != env[key]:
            raise ValueError('REFUSING RESUME: environment mismatch: ' + key)
    history = checkpoint['history']
    if [h['epoch'] for h in history] != list(range(1, checkpoint['epoch'] + 1)):
        raise ValueError('REFUSING RESUME: inconsistent checkpoint history')


@contextlib.contextmanager
def run_lock(path):
    """OS-managed lock releases even when the process crashes; lock file may remain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a+b')
    f.seek(0)
    if not f.read(1):
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError('Another process owns this run directory') from None
    try:
        yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()
