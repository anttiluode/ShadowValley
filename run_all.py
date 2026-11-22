"""Run every cell of the benchmark, 20 seeds each, and save results/cells.json.

    python run_all.py            # ~1 h on 2 CPU cores, 560 runs (uses all cores, one thread per run)
    python run_all.py --quick    # 3 seeds per cell, a few minutes
"""
import argparse, json, os, time
from multiprocessing import Pool
from pathlib import Path
import torch

CELLS = [
    # name,                                        arch,           kind,        heads, detail,  pol, aux_mode,  steps
    # --- the trap condition: the query sees only the scalar cue level (as in HEAD) ---
    ("A' attn, scalar cue | softmax",               'attn-scalar',  'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ("A' attn, scalar cue | differential",          'attn-scalar',  'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ("A' attn, scalar cue | cog",                   'attn-scalar',  'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ("A' attn, scalar cue | flashlight",            'attn-scalar',  'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
    ("A' attn, scalar cue | softmax, 2 heads",      'attn-scalar',  'softmax',    2, 'zero-sum',  1, 'unsigned', 2000),
    ("A' attn, scalar cue | softmax, 4x longer",    'attn-scalar',  'softmax',    1, 'zero-sum',  1, 'unsigned', 8000),
    ("A' attn, scalar cue | softmax, iid detail",   'attn-scalar',  'softmax',    1, 'iid',       1, 'unsigned', 2000),
    ("A' attn, scalar cue | softmax, reversed",     'attn-scalar',  'softmax',    1, 'zero-sum', -1, 'unsigned', 2000),
    ("A' attn, scalar cue | flashlight, reversed",  'attn-scalar',  'flashlight', 1, 'zero-sum', -1, 'unsigned', 2000),
    ("A' attn, scalar cue | signed teacher, reversed", 'attn-scalar', 'flashlight', 1, 'zero-sum', -1, 'signed', 2000),
    ("A' attn, scalar cue | shuffled teacher",      'attn-scalar',  'flashlight', 1, 'zero-sum',  1, 'shuffled', 2000),
    ("A' attn, scalar cue | random-table teacher",  'attn-scalar',  'flashlight', 1, 'zero-sum',  1, 'random',   2000),
    ("B' block, scalar cue | softmax",              'block-scalar', 'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ("B' block, scalar cue | differential",         'block-scalar', 'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ("B' block, scalar cue | cog",                  'block-scalar', 'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ("B' block, scalar cue | flashlight",           'block-scalar', 'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
    ("B' block, scalar cue | random-table teacher", 'block-scalar', 'flashlight', 1, 'zero-sum',  1, 'random',   2000),
    ("B' block, scalar cue | softmax, 2 heads",     'block-scalar', 'softmax',    2, 'zero-sum',  1, 'unsigned', 2000),
    # --- rich query: the cue also arrives as a one-hot embedding ---
    ('A attn, one-hot cue | softmax',               'attn',         'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn, one-hot cue | differential',          'attn',         'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn, one-hot cue | cog',                   'attn',         'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ('A attn, one-hot cue | flashlight',            'attn',         'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block, one-hot cue | softmax',              'block',        'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block, one-hot cue | differential',         'block',        'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block, one-hot cue | cog',                  'block',        'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ('B block, one-hot cue | flashlight',           'block',        'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
    # --- the network must compute the average itself (2 layers, no cue given) ---
    ('C self-mean | softmax',                       'self',         'softmax',    1, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | differential',                  'self',         'diff',       1, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | cog',                           'self',         'cog',        1, 'zero-sum',  1, 'unsigned', 2000),
    ('C self-mean | flashlight',                    'self',         'flashlight', 1, 'zero-sum',  1, 'unsigned', 2000),
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
    ap.add_argument('--only', nargs='*', help='run only these cell names; merge into --out')
    a = ap.parse_args()
    seeds = range(1, 4) if a.quick else range(1, 21)
    cells = [c for c in CELLS if not a.only or c[0] in a.only]
    jobs = [c + (s,) for c in cells for s in seeds]
    jobs.sort(key=lambda j: -j[7] * {'attn': 1, 'block': 2, 'self': 5}[j[1].replace('-scalar', '')])   # long jobs first
    Path(a.out).parent.mkdir(exist_ok=True)
    t0 = time.time(); rows = []
    keep = []
    if a.only and Path(a.out).exists():
        keep = [r for r in json.loads(Path(a.out).read_text())['runs'] if r['cell'] not in a.only]
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} runs, {time.time() - t0:.0f} s', flush=True)
                Path(a.out).write_text(json.dumps({'torch': torch.__version__, 'runs': keep + rows}) + '\n')
    print('done', round(time.time() - t0), 's')
