import ast
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models import ResNet18
from train import lr_for_epoch, mixup_loss, mixup_data
from utils.state import (atomic_write, capture_rng, restore_rng, rng_digest,
                         save_checkpoint, validate_resume)


class Invariants(unittest.TestCase):
    def test_official_lr_executed_over_all_epochs(self):
        tree = ast.parse((ROOT/'references/official-mixup-cifar10/train.py').read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='adjust_learning_rate')
        ns = {'args': SimpleNamespace(lr=.1)}
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<official LR>', 'exec'), ns)
        opt = SimpleNamespace(param_groups=[{'lr': .1}])
        for zero_based in range(200):
            self.assertEqual(opt.param_groups[0]['lr'], lr_for_epoch(zero_based+1))
            ns['adjust_learning_rate'](opt, zero_based)
        self.assertEqual([lr_for_epoch(i) for i in [100,101,102,150,151,152]], [.1,.1,.01,.01,.01,.001])

    def test_model_identical_to_official_forward_backward(self):
        spec = importlib.util.spec_from_file_location('official_reference', ROOT/'references/official-mixup-cifar10/models/resnet.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(17)
        ours = ResNet18().eval()
        torch.manual_seed(17)
        theirs = module.ResNet18().eval()
        self.assertEqual(sum(p.numel() for p in ours.parameters()), sum(p.numel() for p in theirs.parameters()))
        self.assertEqual(sum(p.numel() for p in ours.parameters()), 11171274)
        for k,v in ours.state_dict().items():
            self.assertTrue(torch.equal(v, theirs.state_dict()[k]))
        x = torch.randn(2,3,32,32)
        a, b = ours(x), theirs(x)
        self.assertTrue(torch.equal(a,b))
        a.sum().backward()
        b.sum().backward()
        for p,q in zip(ours.parameters(),theirs.parameters()):
            self.assertTrue(torch.equal(p.grad,q.grad))
        self.assertFalse(any(isinstance(m,torch.nn.Dropout) for m in ours.modules()))

    def test_soft_label_loss_and_gradients(self):
        logits = torch.randn(4,10,requires_grad=True)
        a, b = torch.tensor([0,1,2,3]), torch.tensor([3,2,1,0])
        lam = .37
        expected = -(lam*torch.nn.functional.one_hot(a,10)+(1-lam)*torch.nn.functional.one_hot(b,10)) * logits.log_softmax(1)
        expected = expected.sum(1).mean()
        actual = mixup_loss(logits,a,b,lam)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,logits,retain_graph=True)[0],torch.autograd.grad(expected,logits)[0])
        x = torch.randn(4,3,32,32)
        xm,ya,yb,l = mixup_data(x,a,0)
        self.assertTrue(torch.equal(xm,x))
        self.assertEqual(l,1)
        torch.testing.assert_close(mixup_loss(logits,ya,yb,l),torch.nn.functional.cross_entropy(logits,a))

    def test_rng_roundtrip(self):
        gs = {'train': torch.Generator().manual_seed(11), 'test': torch.Generator().manual_seed(12)}
        state = capture_rng(gs)
        def draw():
            return (random.random(), np.random.beta(1,1), torch.rand(3), torch.rand(3,device='cuda'),
                    torch.randperm(20,generator=gs['train']),torch.rand(4,generator=gs['test']))
        first = draw()
        restore_rng(state, gs)
        self.assertEqual(rng_digest(state),rng_digest(capture_rng(gs)))
        second = draw()
        for a,b in zip(first,second):
            self.assertTrue(torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b)

    def test_atomic_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            p = Path(tmp)/'checkpoint.pt'
            save_checkpoint(p,{'epoch':1})
            before = p.read_bytes()
            def fail(f):
                f.write(b'partial')
                raise OSError('Injected write interruption')
            with self.assertRaises(OSError):
                atomic_write(p,fail)
            self.assertEqual(before,p.read_bytes())
            self.assertEqual(torch.load(p,weights_only=False)['epoch'],1)

    def test_config_mismatch_rejected(self):
        config = {'model':'a','dataset':'CIFAR10','batch_size':128,'optimizer':'SGD','seed':20170922,'alpha':1.}
        env = dict.fromkeys(('python','pytorch','torchvision','numpy','cuda_runtime','cudnn','gpu'),'same')
        cp = {'config': config,'environment':env,'history':[{'epoch':1}],'epoch':1}
        validate_resume(cp,config,env)
        for key in config:
            changed = copy.deepcopy(config)
            changed[key] = 'different'
            with self.assertRaisesRegex(ValueError,'REFUSING RESUME'):
                validate_resume(cp,changed,env)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
