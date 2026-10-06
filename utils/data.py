import random
import numpy as np
import torch
from torchvision import datasets, transforms


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(config, root, download=False):
    c = config
    normalize = transforms.Normalize(c['normalization_mean'], c['normalization_std'])
    train_transform = transforms.Compose([
        transforms.RandomCrop(c['crop_size'], padding=c['crop_padding']),
        transforms.RandomHorizontalFlip(p=c['horizontal_flip_probability']),
        transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train = datasets.CIFAR10(root=str(root), train=True, download=download, transform=train_transform)
    test = datasets.CIFAR10(root=str(root), train=False, download=download, transform=test_transform)
    if len(train) != 50000 or len(test) != 10000:
        raise ValueError('CIFAR-10 cardinality must be 50000 train / 10000 test')
    if not train._check_integrity() or not test._check_integrity():
        raise ValueError('CIFAR-10 integrity check failed')
    generators = {k: torch.Generator().manual_seed(c['seed'] + i) for i, k in enumerate(('train', 'test'))}
    common = dict(num_workers=c['num_workers'], pin_memory=c['pin_memory'],
                  persistent_workers=c['persistent_workers'], worker_init_fn=seed_worker,
                  drop_last=c['drop_last'])
    loaders = {
        'train': torch.utils.data.DataLoader(train, batch_size=c['batch_size'], shuffle=True,
                                            generator=generators['train'], **common),
        'test': torch.utils.data.DataLoader(test, batch_size=c['test_batch_size'], shuffle=False,
                                           generator=generators['test'], **common)}
    provenance = {'dataset': 'CIFAR10', 'train_count': len(train), 'test_count': len(test),
                  'integrity': 'torchvision CIFAR10 official per-file MD5 PASS',
                  'url': datasets.CIFAR10.url, 'archive_md5': datasets.CIFAR10.tgz_md5,
                  'train_files': train.train_list, 'test_files': test.test_list,
                  'train_transform': repr(train_transform), 'test_transform': repr(test_transform)}
    return loaders, generators, provenance
