from pathlib import Path
from types import SimpleNamespace
import sys, importlib.util, copy, json, os
import torch
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'AT-ReID-fast'
sys.path.insert(0, str(D))
torch.set_num_threads(2)
from others import optim as newoptim
from loss import hdw as newloss
from model import moae as newmoae

class ReferenceOptim:
    get_optim = staticmethod(newoptim.get_optim)

    @staticmethod
    def get_param_groups(args, lr, model):
        args = copy.copy(args)
        args.group_optimizer = False
        return newoptim.get_param_groups(args, lr, model)

    @staticmethod
    def adjust_learning_rate(args, opt, epoch, model):
        args = copy.copy(args)
        args.group_optimizer = False
        return newoptim.adjust_learning_rate(args, opt, epoch, model)
oldoptim = ReferenceOptim

class ReferenceLoss(newloss.saidloss_hdw):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_unused_stats = False
oldloss = SimpleNamespace(saidloss_hdw=ReferenceLoss)

class ReferenceMoAE(newmoae.Mlp_moe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_unused_stats = False
oldmoae = SimpleNamespace(Mlp_moe=ReferenceMoAE)
from others.utils import DeferredTrainMetrics, AverageMeter
import unittest

class SpeedRegressionTests(unittest.TestCase):

    def test_loss_routing_optimizer_and_metrics(self):
        checks = []

        def eq(a, b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        p = torch.tensor([0, 0, 1, 1, 2, 2])
        c = torch.arange(6)
        for said in [False, True]:
            for hdw in [False, True]:
                for mids in [torch.tensor([1, 2, 1, 2, 1, 2]), torch.ones(6, dtype=torch.long), torch.full((6,), 2)]:
                    for details in [False, True]:
                        a = oldloss.saidloss_hdw(mapping, said=said, hdw=hdw)
                        b = newloss.saidloss_hdw(mapping, said=said, hdw=hdw)
                        torch.manual_seed(0)
                        xs = [torch.randn(6, 6 if i % 2 == 0 else 3, requires_grad=True) for i in range(6)]
                        ys = [x.detach().clone().requires_grad_() for x in xs]
                        ra = a(xs, p, c, mids, return_details=details)
                        rb = b(ys, p, c, mids, return_details=details)
                        if details:
                            for k in ra[1]:
                                if torch.is_tensor(ra[1][k]):
                                    eq(ra[1][k], rb[1][k])
                                else:
                                    assert ra[1][k] == rb[1][k]
                            ra, rb = (ra[0], rb[0])
                        eq(ra, rb)
                        ra.backward()
                        rb.backward()
                        for x, y in zip(xs, ys):
                            if x.grad is None:
                                assert y.grad is None
                            else:
                                eq(x.grad, y.grad)
        checks.append('24 SAID/HDW/modality/details combinations: exact loss, details and gradients')
        for epoch in [1, 39, 40, 41, 120]:
            for stats in [False, True]:
                for training in [True, False]:
                    torch.manual_seed(4)
                    a = oldmoae.Mlp_moe(16, 24, 16, ncls=6, moae=True)
                    b = newmoae.Mlp_moe(16, 24, 16, ncls=6, moae=True)
                    b.load_state_dict(a.state_dict())
                    a.train(training)
                    b.train(training)
                    a.moae_epoch = b.moae_epoch = epoch
                    b.collect_route_stats = stats
                    x = torch.randn(3, 10, 16, requires_grad=True)
                    y = x.detach().clone().requires_grad_()
                    torch.manual_seed(10)
                    ra = a(x)
                    torch.manual_seed(10)
                    rb = b(y)
                    eq(ra, rb)
                    ra.square().mean().backward()
                    rb.square().mean().backward()
                    eq(x.grad, y.grad)
                    for pa, pb in zip(a.parameters(), b.parameters()):
                        eq(pa.grad, pb.grad)
                    if stats and training:
                        for k in a.last_route_stats:
                            eq(a.last_route_stats[k], b.last_route_stats[k])
                    if training:
                        a.update_balance_state()
                        b.update_balance_state()
                        eq(a.balance_bias, b.balance_bias)
                        eq(a.balance_usage_ema, b.balance_usage_ema)
                    else:
                        assert b.last_balance_fractions is None and b.last_route_stats is None
        checks.append('20 MoAE train/eval/statistics/freeze cases: exact outputs, gradients and training balance state')

        class Toy(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.base = torch.nn.Module()
                self.base.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(12)])
                self.base.norm = torch.nn.LayerNorm(4)
                self.base.pos_embed = torch.nn.Parameter(torch.zeros(1, 4))
                self.bottleneck = torch.nn.BatchNorm1d(4)
                self.classifier = torch.nn.Linear(4, 3)

            def forward(self, x):
                for layer in self.base.blocks:
                    x = torch.tanh(layer(x))
                return self.classifier(self.bottleneck(self.base.norm(x + self.base.pos_embed)))
        args = SimpleNamespace(lr=0.008, wd=0.0005, dlrl=[10, 11, 12], dlr=10, dwd=True, optim='sgd', optimizer_fused=False, warmup_epoch=10, warmup_rate=0.1, scheduler_epoch=[], scheduler_rate=0.1, max_epoch=120, cosmin_rate=0, group_optimizer=True)
        torch.manual_seed(3)
        a = Toy()
        b = copy.deepcopy(a)
        oa = oldoptim.get_optim(args, oldoptim.get_param_groups(args, args.lr, a))
        ob = newoptim.get_optim(args, newoptim.get_param_groups(args, args.lr, b))
        for epoch in [1, 5, 10, 11, 60, 100, 120]:
            oldoptim.adjust_learning_rate(args, oa, epoch, a)
            newoptim.adjust_learning_rate(args, ob, epoch, b)
            ma = {id(p): (g['lr'], g['weight_decay']) for g in oa.param_groups for p in g['params']}
            mb = {id(p): (g['lr'], g['weight_decay']) for g in ob.param_groups for p in g['params']}
            for pa, pb in zip(a.parameters(), b.parameters()):
                assert ma[id(pa)] == mb[id(pb)]
            for _ in range(2):
                x = torch.randn(8, 4)
                target = torch.randn(8, 3)
                oa.zero_grad()
                ob.zero_grad()
                la = (a(x) - target).square().mean()
                lb = (b(x) - target).square().mean()
                eq(la, lb)
                la.backward()
                lb.backward()
                oa.step()
                ob.step()
                for pa, pb in zip(a.parameters(), b.parameters()):
                    eq(pa, pb)
                    eq(oa.state[pa]['momentum_buffer'], ob.state[pb]['momentum_buffer'])
        checks.append(f'SGD 14 updates across warmup/cosine boundaries: exact parameters and momentum; {len(oa.param_groups)} -> {len(ob.param_groups)} groups')
        ma = [AverageMeter() for _ in range(5)]
        mb = [AverageMeter() for _ in range(5)]
        d = DeferredTrainMetrics(mb)
        for n in [64, 64, 17, 1]:
            vals = [torch.randn(()) for _ in range(4)]
            correct = torch.tensor(n // 2)
            for meter, v in zip(ma[:3], vals[:3]):
                meter.update(v.item(), n)
            ma[3].update(float(vals[3]))
            ma[4].update(100.0 * correct.item() / n)
            d.add(*vals, correct, n)
        d.flush()
        d.flush()
        for a, b in zip(ma, mb):
            assert vars(a) == vars(b), (vars(a), vars(b))
        checks.append('Deferred metrics preserve values, counts and accumulation with unequal batch sizes')
        print(json.dumps({'pass': True, 'checks': checks}, indent=2))

class LossInterfaceTests(unittest.TestCase):

    def test_counts_and_legacy_nll(self):
        from loss.said import compute_head_losses, _original_compute_head_losses
        from loss.hdw import saidloss_hdw
        torch.manual_seed(7)
        pids = torch.tensor([0, 0, 1, 1, 2, 2])
        cids = torch.arange(6)
        mids = torch.tensor([1, 2, 1, 2, 1, 2])
        criterion = saidloss_hdw({0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2})
        logits = [torch.randn(6, 6 if i % 2 == 0 else 3) for i in range(6)]
        for said in [False, True]:
            for hdw in [False, True]:
                expected = _original_compute_head_losses(logits, pids, cids, mids, criterion.invalid_same_pid_mask, said=said, hdw=hdw)
                actual, counts, nominal = compute_head_losses(logits, pids, cids, mids, criterion.invalid_same_pid_mask, said=said, hdw=hdw, return_class_counts=True)
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                if said:
                    torch.testing.assert_close(counts[0], torch.full((3,), 5.0))
                    torch.testing.assert_close(nominal[0], torch.full((3,), 6.0))
                else:
                    torch.testing.assert_close(counts[0], torch.full((6,), 6.0))
if __name__ == '__main__':
    unittest.main()
