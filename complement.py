"""Silent complement reads: what does each trained network actually use?

In the zero-sum world the target d_j can be computed two ways that agree on every
training example:
  direct      y = x_j - c                      (uses slot j)
  complement  y = -(sum_{i != j} x_i - 7c)     (never touches slot j; uses the
                                                constraint sum(d) = 0)
They disagree as soon as one slot changes on its own. For every solved run we
re-draw ONLY slot j's detail (d_j -> d_j') and ONLY one other slot's detail, and
regress the change in output on the change in detail:

  k_target  = d y_hat / d d_j      direct -> 1,  complement -> 0
  k_other   = d y_hat / d d_i      direct -> 0,  complement -> -1   (mean over i != j)

and compare with what the attention map says (weight on slot j). Noise fragility:
i.i.d. N(0, 0.1^2) added to every slot at test time, target unchanged; ideal
direct reader gains 0.01, ideal complement reader 0.07 (7 slots of noise).

    python complement.py     # ~35 min on 2 cores -> results/complement.json, results/complement.md
"""
import json, os, time
from multiprocessing import Pool
from pathlib import Path
import torch
from sv import SLOTS, LEVELS, sample, train_eval

CELLS = [
    ('A attn, one-hot cue | softmax',        'attn',         'softmax',    1),
    ('A attn, one-hot cue | flashlight',     'attn',         'flashlight', 1),
    ('B block, one-hot cue | softmax',       'block',        'softmax',    1),
    ('B block, one-hot cue | differential',  'block',        'diff',       1),
    ('B block, one-hot cue | cog',           'block',        'cog',        1),
    ("B' block, scalar cue | softmax",       'block-scalar', 'softmax',    1),
    ("B' block, scalar cue | softmax, 2 heads", 'block-scalar', 'softmax', 2),
    ("A' attn, scalar cue | softmax",        'attn-scalar',  'softmax',    1),
    ('C self-mean | softmax',                'self',         'softmax',    1),
]
NOISE = 0.1


def probe(model, mapping, seed):
    g = torch.Generator().manual_seed(91000 + seed)
    x, y, k = sample(16384, mapping, g)
    j = mapping[k]
    n = len(k); ar = torch.arange(n)
    with torch.no_grad():
        p0, a, _ = model(x, k)
        # re-draw slot j's detail only
        delta = torch.randn(n, generator=g) * (7 / 8) ** 0.5
        x1 = x.clone(); x1[ar, j] += delta
        p1, _, _ = model(x1, k)
        k_target = float((delta * (p1 - p0)).sum() / (delta * delta).sum())
        # re-draw one other slot's detail only
        off = (j + 1 + torch.randint(SLOTS - 1, (n,), generator=g)) % SLOTS
        x2 = x.clone(); x2[ar, off] += delta
        p2, _, _ = model(x2, k)
        k_other = float((delta * (p2 - p0)).sum() / (delta * delta).sum())
        # knock-out: erase slot j's detail entirely (x_j := c)
        x3 = x.clone(); x3[ar, j] = LEVELS[k]
        p3, _, _ = model(x3, k)
        ko = float(((p3 - p0) ** 2).mean() / (y ** 2).mean())
        # i.i.d. input noise, target unchanged
        pn, _, _ = model(x + NOISE * torch.randn(x.shape, generator=g), k)
        base = float(((p0 - y) ** 2).mean())
        noisy = float(((pn - y) ** 2).mean())
        att = a.max(1).values if a.shape[1] > 1 else a[:, 0]     # best head's row
        att_j = float(a[ar, :, j].abs().max(1).values.mean())
    kind = ('direct' if k_target > 0.6 and k_other > -0.3 else
            'complement' if k_target < 0.3 and k_other < -0.6 else 'mixed')
    return {'k_target': k_target, 'k_other': k_other, 'knockout_change': ko,
            'noise_mse_gain': noisy - base, 'noise_gain_vs_ideal_direct': (noisy - base) / NOISE ** 2,
            'attention_on_target': att_j, 'causal_read': kind}


def job(args):
    torch.set_num_threads(1)
    name, arch, kind, heads, seed = args
    res, model, mapping = train_eval(seed, arch, kind, heads, return_model=True)
    model.eval()
    out = {'cell': name, 'seed': seed, 'nmse': res['nmse'], 'solved': res['success'],
           'route_mode': res['route_mode'], 'beta_eff_median': res['beta_eff_median']}
    out.update(probe(model, mapping, seed))
    return out


if __name__ == '__main__':
    t0 = time.time()
    jobs = [c + (s,) for c in CELLS for s in range(1, 21)]
    jobs.sort(key=lambda j: -(5 if j[1] == 'self' else 2 if 'block' in j[1] else 1))
    with Pool(os.cpu_count()) as pool:
        rows = list(pool.imap_unordered(job, jobs))
    rows.sort(key=lambda r: (r['cell'], r['seed']))
    Path('results').mkdir(exist_ok=True)
    Path('results/complement.json').write_text(json.dumps(rows, indent=1) + '\n')
    lines = ['| cell | solved | attention says (at / away) | causal read of solved runs: direct / complement / mixed | '
             'solved, attention "at" but causal complement | median noise gain vs ideal direct (solved) |',
             '|---|---|---|---|---|---|']
    import statistics as st
    for name, *_ in CELLS:
        rs = [r for r in rows if r['cell'] == name]; sv_ = [r for r in rs if r['solved']]
        c = lambda f: sum(1 for r in sv_ if f(r))
        hidden = c(lambda r: r['route_mode'] == 'at' and r['causal_read'] == 'complement')
        lines.append(f"| {name} | {len(sv_)}/20 | {sum(r['route_mode']=='at' for r in sv_)} / {sum(r['route_mode']=='away' for r in sv_)} | "
                     f"{c(lambda r: r['causal_read']=='direct')} / {c(lambda r: r['causal_read']=='complement')} / {c(lambda r: r['causal_read']=='mixed')} | "
                     f"{hidden} | {st.median([r['noise_gain_vs_ideal_direct'] for r in sv_]) if sv_ else float('nan'):.2f} |")
    Path('results/complement.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines)); print('seconds', round(time.time() - t0))
