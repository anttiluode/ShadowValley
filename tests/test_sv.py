"""python -m pytest -q tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import torch
from sv import SLOTS, LEVELS, sample, predicted_nmse, TinyTransformer, CorrTable


def test_neon_curve_is_the_cos2_formula():
    j = 3
    for m in range(1, 8):
        a = torch.zeros(SLOTS)
        others = [i for i in range(SLOTS) if i != j][:m]
        a[others] = 1 / m
        assert abs(predicted_nmse(a, j, 'zero-sum') - (1 - m / (7 * (8 - m)))) < 1e-6
    a = torch.zeros(SLOTS); a[j] = 1
    assert predicted_nmse(a, j, 'zero-sum') < 1e-6


def test_closed_form_matches_least_squares():
    g = torch.Generator().manual_seed(0)
    mapping = torch.randperm(SLOTS, generator=g)
    x, y, k = sample(200_000, mapping, g)
    for trial in range(4):
        a = torch.rand(SLOTS, generator=g) ** 3; a = a / a.sum()
        cue = 2; j = int(mapping[cue]); ix = k == cue
        X = torch.stack([(x[ix] * a).sum(1), torch.ones(int(ix.sum()))], 1).double()
        w = torch.linalg.lstsq(X, y[ix].double()[:, None]).solution
        nm = float(((X @ w).squeeze() - y[ix]).pow(2).mean() / y[ix].pow(2).mean())
        assert abs(nm - predicted_nmse(a, j, 'zero-sum')) < 0.01


def test_iid_has_no_look_away_solution():
    j = 0; a = torch.zeros(SLOTS); a[1:] = 1 / 7
    assert predicted_nmse(a, j, 'iid') > 0.999          # reading only the others says nothing
    assert predicted_nmse(a, j, 'zero-sum') < 1e-6      # with zero-sum detail it is exact


def test_variants_run_and_cog_is_signed():
    g = torch.Generator().manual_seed(1)
    x, y, k = sample(64, torch.arange(SLOTS), g)
    for arch in ('attn', 'block', 'self'):
        for kind in ('softmax', 'diff', 'cog'):
            p, a, s = TinyTransformer(arch, kind, heads=2)(x, k)
            assert p.shape == (64,) and a.shape == (64, 2, SLOTS)
            if kind == 'softmax':      # query row over the 8 slots excludes the query token itself
                assert (a.sum(-1) <= 1 + 1e-5).all()
    p, a, s = TinyTransformer('attn', 'cog')(x, k)
    assert (a < 0).any()


def test_unsigned_teacher_survives_sign_reversal():
    g = torch.Generator().manual_seed(2)
    mapping = torch.randperm(SLOTS, generator=g)
    for pol in (1.0, -1.0):
        t_u, t_s = CorrTable(), CorrTable()
        for _ in range(10):
            x, y, k = sample(512, mapping, g, polarity=pol)
            z = x - x.mean(1, keepdim=True)
            t_u.collect(k, z, y); t_s.collect(k, z, y)        # residual at y_hat = 0 is y
        assert torch.equal(t_u.targets('unsigned'), mapping)
        assert torch.equal(t_s.targets('signed'), mapping) == (pol > 0)
