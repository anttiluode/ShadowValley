"""Run every cell of the benchmark, 20 seeds each, and save results/cells.json.

    python run_all.py            # ~1 h on 2 CPU cores (uses all cores, one thread per run)
    python run_all.py --quick    # 3 seeds per cell, a few minutes
"""
import argparse, json, os, time
from multiprocessing import Pool
from pathlib import Path
import torch

CELLS = [
    # name,                                   arch,    kind,        heads, detail,    polarity, aux_mode,   steps
    ('A attn | softmax',                      'attn',  'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn | differential',                 'attn',  'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn | cog',                          'attn',  'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn | flashlight',                   'attn',  'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn | softmax | iid control',        'attn',  'softmax',    1, 'iid',       1, 'unsigned', 2000),
    ('A attn | softmax | 4x longer',          'attn',  'softmax',    1, 'zero-sum',  1, 'unsigned', 8000),
    ('B block | softmax',                     'block', 'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block | differential',                'block', 'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block | cog',                         'block', 'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block | flashlight',                  'block', 'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block | softmax | iid control',       'block', 'softmax',    1, 'iid',       1, 'unsigned', 2000),
    ('B block | softmax | reversed',          'block', 'softmax',    1, 'zero-sum', -1, 'unsigned', 2000),
    ('B block | flashlight | reversed',       'block', 'flashlight', 1, 'zero-sum', -1, 'unsigned', 2000),
    ('B block | flashlight signed | reversed','block', 'flashlight', 1, 'zero-sum', -1, 'signed',   2000),
    ('B block | flashlight shuffled',         'block', 'flashlight', 1, 'zero-sum',  1, 'shuffled', 2000),
    ('B block 2 heads | softmax',             'block', 'softmax',    2, 'zero-sum',  1, 'unsigned', 2000),
    ('B block 2 heads | differential',        'block', 'diff',       2, 'zero-sum',  1, 'unsigned', 2000),
    ('B block 2 heads | cog',                 'block', 'cog',        2, 'zero-sum',  1, 'unsigned', 2000),
    ('B block 2 heads | flashlight',          'block', 'flashlight', 2, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | softmax',                 'self',  'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | differential',            'self',  'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | cog',                     'self',  'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | flashlight',              'self',  'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
]


def job(args):
    torch.set_num_threads(1)
    from sv import train_eval
    name, arch, kind, heads, detail, pol, aux, steps, seed = args
    t = time.time()
    r = train_eval(seed, arch, kind, heads, detail, float(pol), steps=steps, aux_mode=aux, record=True)
    r['cell'] = name; r['aux_mode'] = aux; r['steps'] = steps; r['seconds'] = round(time.time() - t, 1)
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results/cells.json')
    a = ap.parse_args()
    seeds = range(1, 4) if a.quick else range(1, 21)
    jobs = [c + (s,) for c in CELLS for s in seeds]
    jobs.sort(key=lambda j: -j[7] * {'attn': 1, 'block': 2, 'self': 5}[j[1]])   # long jobs first
    Path(a.out).parent.mkdir(exist_ok=True)
    t0 = time.time(); rows = []
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} runs, {time.time() - t0:.0f} s', flush=True)
                Path(a.out).write_text(json.dumps({'torch': torch.__version__, 'runs': rows}) + '\n')
    print('done', round(time.time() - t0), 's')
