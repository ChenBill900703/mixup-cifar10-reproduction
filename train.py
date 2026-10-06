"""Faithful official mixup-cifar10 computation with Windows-safe epoch resume."""
import argparse
import copy
import itertools
import json
import math
import os
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import ResNet18
from utils.data import build_loaders
from utils.report import comparison, export_run
from utils.state import (capture_rng, environment, restore_rng, rng_digest, run_lock,
                         save_checkpoint, save_json, sha256, source_hashes, validate_resume)

ROOT = Path(__file__).resolve().parent


def lr_for_epoch(epoch, initial=0.1):
    """1-based epoch; official adjusts AFTER train/test with 0-based index."""
    if epoch < 1:
        raise ValueError('epoch must be >= 1')
    lr = initial
    if epoch >= 102:
        lr /= 10
    if epoch >= 152:
        lr /= 10
    return lr


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    # Preserve official CPU randperm followed by CUDA transfer, including alpha=0.
    index = torch.randperm(x.size(0)).to(x.device)
    return lam*x + (1-lam)*x[index, :], y, y[index], lam


def mixup_loss(outputs, ya, yb, lam):
    return lam*nn.functional.cross_entropy(outputs, ya) + (1-lam)*nn.functional.cross_entropy(outputs, yb)


def configure(c):
    random.seed(c['seed'])
    np.random.seed(c['seed'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.benchmark = c['cudnn_benchmark']
    torch.backends.cudnn.deterministic = c['deterministic_algorithms']
    torch.use_deterministic_algorithms(c['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = c['allow_tf32']
    torch.backends.cudnn.allow_tf32 = c['allow_tf32']
    torch.set_float32_matmul_precision('highest')


def effective_config(base, method):
    c = copy.deepcopy(base)
    # Run selectors do not alter an existing method's scientific configuration.
    c.pop('RUN_ERM')
    c.pop('RUN_MIXUP')
    c['method'] = method
    c['alpha'] = 0.0 if method == 'ERM' else base['alpha']
    c['effective_epochs'] = c['quick_epochs'] if c['QUICK_TEST'] else c['epochs']
    c['source_hashes'] = source_hashes(ROOT)
    return c


def check_fixed(c):
    fixed = {'model': 'official_ResNet18_PreActBlock', 'dataset': 'CIFAR10', 'epochs': 200,
             'batch_size': 128, 'test_batch_size': 100, 'optimizer': 'SGD', 'lr': 0.1,
             'momentum': .9, 'weight_decay': 1e-4, 'lr_policy': 'official_post_epoch_zero_based',
             'USE_AMP': False, 'allow_tf32': False, 'nesterov': False, 'dampening': 0,
             'drop_last': False, 'persistent_workers': False, 'alpha': 1.0,
             'normalization_mean': [.4914, .4822, .4465],
             'normalization_std': [.2023, .1994, .2010], 'crop_size': 32,
             'crop_padding': 4, 'horizontal_flip_probability': .5}
    for k, v in fixed.items():
        if c[k] != v:
            raise ValueError(f'Locked reproduction setting {k}: expected {v}, got {c[k]}')
    if c['quick_epochs'] != 2 or min(c['quick_train_batches'], c['quick_test_batches']) < 1:
        raise ValueError('Quick test requires 2 epochs and positive batch caps')
    source = ROOT/'references/official-mixup-cifar10/models/resnet.py'
    if sha256(ROOT/'models/official_resnet.py') != sha256(source):
        raise ValueError('Model differs from vendored official source')


def formal_gate(c, approved):
    if not approved:
        raise RuntimeError('Formal training requires explicit user approval: --approve-formal')
    p = ROOT/'validation/readiness.json'
    if not p.exists():
        raise RuntimeError('Formal training blocked: validation/readiness.json missing')
    r = json.loads(p.read_text(encoding='utf-8'))
    if r.get('status') != 'PASS' or r.get('source_hashes') != source_hashes(ROOT):
        raise RuntimeError('Formal training blocked: validation failed or code changed; rerun tests/validate.py')
    expected = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    for k in ('RUN_ERM', 'RUN_MIXUP', 'QUICK_TEST'):
        expected.pop(k)
    actual = {k:v for k,v in c.items() if k not in ('RUN_ERM','RUN_MIXUP','QUICK_TEST')}
    if actual != expected:
        raise RuntimeError('Formal config differs from validated configuration')
    observed = environment()
    for key in ('python', 'pytorch', 'torchvision', 'numpy', 'cuda_runtime', 'cudnn', 'gpu'):
        if observed[key] != r['environment'][key]:
            raise RuntimeError('Formal environment changed since validation: ' + key)


def epoch_pass(model, loader, c, optimizer=None):
    training = optimizer is not None
    model.train(training)
    limit = c['quick_train_batches' if training else 'quick_test_batches'] if c['QUICK_TEST'] else None
    batches = itertools.islice(loader, limit) if limit is not None else loader
    loss_total, correct, total = 0., 0, 0
    with torch.set_grad_enabled(training):
        for x, y in batches:
            x, y = x.cuda(), y.cuda()
            if training:
                x, ya, yb, lam = mixup_data(x, y, c['alpha'])
                outputs = model(x)
                loss = mixup_loss(outputs, ya, yb, lam)
                optimizer.zero_grad(set_to_none=False)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(x)
                loss = nn.functional.cross_entropy(outputs, y)
                correct += outputs.argmax(1).eq(y).sum().item()
            if not math.isfinite(loss.item()):
                raise FloatingPointError('Non-finite loss; last completed checkpoint preserved')
            loss_total += loss.item()*y.size(0)
            total += y.size(0)
    return loss_total/total, 100.*correct/total, total


def run_method(base, method, args):
    c = effective_config(base, method)
    configure(c)
    env = environment()
    if not env['cuda_available']:
        raise RuntimeError('CUDA REQUIRED: torch.cuda.is_available() is False')
    if env['visible_gpu_count'] != 1:
        raise RuntimeError('Exactly one visible GPU required; set CUDA_VISIBLE_DEVICES=0')
    out = ROOT/'outputs'/('quick' if c['QUICK_TEST'] else '')
    if c['QUICK_TEST']:
        out = out/args.quick_id
    out = out/method
    with run_lock(out/'.run.lock'):
        checkpoint_path = out/'resume_checkpoint.pt'
        cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
        if cp:
            validate_resume(cp, c, env)
            print('RESUME CHECKPOINT FOUND', flush=True)
            print(f"LAST COMPLETED EPOCH = {cp['epoch']}\nNEXT EPOCH = {cp['epoch']+1}", flush=True)
        else:
            if (out/'config.json').exists():
                previous = json.loads((out/'config.json').read_text(encoding='utf-8'))
                if previous != c:
                    raise ValueError('Existing run directory config differs; choose a new run directory')
            print('FRESH RUN - STARTING EPOCH 1', flush=True)
        loaders, generators, provenance = build_loaders(c, ROOT/'data', args.download)
        model = ResNet18().cuda()  # Official source file copied byte-for-byte.
        optimizer = torch.optim.SGD(model.parameters(), lr=c['lr'], momentum=c['momentum'],
                                    weight_decay=c['weight_decay'], nesterov=c['nesterov'],
                                    dampening=c['dampening'], foreach=False, fused=False)
        history, best, last = [], {'accuracy': -1., 'epoch': 0}, 0
        run_id = uuid.uuid4().hex if not cp else cp['run_id']
        if cp:
            model.load_state_dict(cp['model_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            history, best, last = cp['history'], cp['best'], cp['epoch']
            if cp['lr_state']['next_epoch_lr'] != lr_for_epoch(last+1, c['lr']):
                raise ValueError('Checkpoint LR state mismatch')
            restore_rng(cp['rng_state'], generators)
            actual_digest = rng_digest(capture_rng(generators))
            if actual_digest != cp['rng_digest']:
                raise RuntimeError('RNG restore verification failed')
            save_json(out/f'resume_verification_epoch_{last}.json',
                      {'last_completed_epoch': last, 'next_epoch': last+1,
                       'rng_digest_saved': cp['rng_digest'], 'rng_digest_restored': actual_digest,
                       'optimizer_restored': True, 'config_match': True})
        save_json(out/'config.json', c)
        save_json(out/'environment.json', cp['environment'] if cp else env)
        save_json(out/'dataset_provenance.json', provenance)
        for name in ('source_boundary.md', 'assumptions.md'):
            shutil.copyfile(ROOT/'docs'/name, out/name)
        invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record_path = out/f'run_record_{invocation}.json'
        record = {'run_id': run_id, 'started_utc': invocation, 'method': method,
                  'mode': 'COMPUTATIONAL_AI', 'quick_test': c['QUICK_TEST'],
                  'result_claim_allowed': False, 'status': 'RUNNING', 'start_epoch': last+1,
                  'source_hashes': c['source_hashes'], 'environment': env}
        save_json(record_path, record)
        try:
            elapsed = history[-1]['elapsed_seconds'] if history else 0.
            for epoch in range(last+1, c['effective_epochs']+1):
                start = time.perf_counter()
                lr = lr_for_epoch(epoch, c['lr'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                train_loss, _, train_n = epoch_pass(model, loaders['train'], c, optimizer)
                test_loss, accuracy, test_n = epoch_pass(model, loaders['test'], c)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                elapsed += seconds
                history.append(dict(epoch=epoch, lr=lr, train_loss=train_loss, test_loss=test_loss,
                                    test_accuracy=accuracy, test_error=100.-accuracy,
                                    epoch_seconds=seconds, elapsed_seconds=elapsed,
                                    train_examples=train_n, test_examples=test_n))
                if accuracy > best['accuracy']:
                    best = {'accuracy': accuracy, 'epoch': epoch}
                for group in optimizer.param_groups:
                    group['lr'] = lr_for_epoch(epoch+1, c['lr'])
                rng = capture_rng(generators)
                cp = {'format_version': 1, 'run_id': run_id, 'epoch': epoch,
                      'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'lr_state': {'policy': c['lr_policy'], 'next_epoch_lr': lr_for_epoch(epoch+1, c['lr'])},
                      'rng_state': rng, 'rng_digest': rng_digest(rng), 'history': history,
                      'best': best, 'config': c, 'environment': env}
                save_checkpoint(checkpoint_path, cp)
                export_run(out, cp)
                print(f"{method} | Epoch {epoch}/{c['effective_epochs']} | LR {lr:.4f} | "
                      f"Train loss {train_loss:.4f} | Test acc {accuracy:.2f} | Error {100-accuracy:.2f} | {seconds:.1f}s", flush=True)
                if args.stop_after_epoch and epoch >= args.stop_after_epoch:
                    print(f'CONTROLLED STOP AFTER COMPLETED EPOCH {epoch}', flush=True)
                    break
            if cp:
                export_run(out, cp)  # Rebuild derived files even after a crash during export.
                if cp['epoch'] == c['effective_epochs']:
                    save_checkpoint(out/'final_model_state_dict.pt', model.state_dict())
            if not c['QUICK_TEST']:
                comparison(ROOT/'outputs')
                shutil.copyfile(ROOT/'outputs/paper_comparison.csv', out/'paper_comparison.csv')
            record.update(status='COMPLETE' if cp and cp['epoch']==c['effective_epochs'] else 'STOPPED',
                          last_completed_epoch=cp['epoch'] if cp else 0,
                          result_claim_allowed=bool(cp and not c['QUICK_TEST'] and cp['epoch']==200),
                          max_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        except BaseException as exc:
            record.update(status='FAILED_OR_INTERRUPTED', error=repr(exc))
            raise
        finally:
            save_json(record_path, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--method', choices=['ERM', 'Mixup', 'both'])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true')
    mode.add_argument('--formal', action='store_true')
    parser.add_argument('--approve-formal', action='store_true')
    parser.add_argument('--quick-id', default='manual')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    if not args.quick_id or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in args.quick_id):
        parser.error('--quick-id must contain letters, digits, underscores or hyphens')
    base = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.quick or args.formal:
        base['QUICK_TEST'] = args.quick
    if args.method:
        base['RUN_ERM'] = args.method in ('ERM', 'both')
        base['RUN_MIXUP'] = args.method in ('Mixup', 'both')
    check_fixed(base)
    if not base['QUICK_TEST']:
        formal_gate(base, args.approve_formal)
    methods = [m for m, flag in [('ERM', 'RUN_ERM'), ('Mixup', 'RUN_MIXUP')] if base[flag]]
    if not methods:
        parser.error('Both RUN_ERM and RUN_MIXUP are false')
    for method in methods:
        run_method(base, method, args)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
