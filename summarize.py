"""results/cells.json -> results/summary.json, results/summary.md, results/figure.png"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/cells.json')
out = src.parent
runs = json.loads(src.read_text())['runs']
cells = defaultdict(list)
for r in runs:
    cells[r['cell']].append(r)
order = [c for c in dict.fromkeys(r['cell'] for r in sorted(runs, key=lambda r: r['cell']))]
from run_all import CELLS
order = [c[0] for c in CELLS if c[0] in cells]

summary = {}
lines = ['| cell | solved | valley | other fail | median NMSE | valley: median m_eff | valley: median beta | solved-direct beta |',
         '|---|---|---|---|---|---|---|---|']
for c in order:
    rs = sorted(cells[c], key=lambda r: r['seed'])
    n = len(rs)
    cnt = {o: sum(r['outcome'] == o for r in rs) for o in ('solved', 'valley', 'direct-side failure')}
    val = [r for r in rs if r['cos_mean'] < 0]          # look-away side, solved or not
    dirs = [r for r in rs if r['cos_mean'] >= 0 and r['success']]
    s = {'n': n, **cnt,
         'median_nmse': float(np.median([r['nmse'] for r in rs])),
         'look_away_side': len(val),
         'look_away_side_solved': sum(r['success'] for r in val),
         'look_away_median_m_eff': float(np.median([r['m_eff_mean'] for r in val])) if val else None,
         'look_away_median_beta': float(np.median([r['beta_eff_median'] for r in val])) if val else None,
         'direct_median_beta': float(np.median([r['beta_eff_median'] for r in dirs])) if dirs else None,
         'seeds_outcome': {r['seed']: r['outcome'] for r in rs}}
    if 'flashlight' in c and rs[0].get('teacher_correct_at_cutoff') is not None:
        s['teacher_all_correct'] = sum(r.get('teacher_correct_at_cutoff') == 8 for r in rs)
    summary[c] = s
    f = lambda v, d=2: '–' if v is None else f'{v:.{d}f}'
    lines.append(f"| {c} | {cnt['solved']}/{n} | {cnt['valley']}/{n} | {cnt['direct-side failure']}/{n} | {s['median_nmse']:.4f} | "
                 f"{f(s['look_away_median_m_eff'])} | {f(s['look_away_median_beta'])} | {f(s['direct_median_beta'])} |")

# formula checks on single-head softmax runs (A and B)
chk = [r for r in runs if r['kind'] in ('softmax', 'flashlight') and r['heads'] == 1 and r['detail'] == 'zero-sum']
meas = np.array([r['nmse'] for r in chk]); pred = np.array([r['pred_nmse_head0'] for r in chk])
summary['_formula'] = {
    'runs': len(chk),
    'corr_measured_vs_cos2_prediction': float(np.corrcoef(meas, pred)[0, 1]),
    'by_arch': {a: {'corr': float(np.corrcoef([r['nmse'] for r in chk if r['arch'] == a],
                                             [r['pred_nmse_head0'] for r in chk if r['arch'] == a])[0, 1]),
                    'median_abs_gap': float(np.median([abs(r['nmse'] - r['pred_nmse_head0']) for r in chk if r['arch'] == a]))}
                for a in ('attn', 'block', 'self') if any(r['arch'] == a for r in chk)}}
(out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines)); print(json.dumps(summary['_formula'], indent=1))

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, col in (('attn', '#1f77b4'), ('block', '#d62728'), ('self', '#2ca02c')):
        rr = [r for r in chk if r['arch'] == a]
        if rr:
            ax[0].scatter([r['pred_nmse_head0'] for r in rr], [r['nmse'] for r in rr], s=14, c=col, label={'attn': 'A attention-only', 'block': 'B pre-LN block', 'self': 'C self-mean, 2 layers'}[a], alpha=.7)
    ax[0].plot([1e-5, 1.5], [1e-5, 1.5], 'k:', lw=1)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('predicted NMSE from the head\'s attention, 1 − cos²(a − ā, e_j − 1/8)')
    ax[0].set_ylabel('measured NMSE'); ax[0].legend(fontsize=8); ax[0].set_title('closed form vs trained network (softmax heads)', fontsize=10)
    m = np.linspace(1, 7, 100)
    ax[1].plot(m, 1 - m / (7 * (8 - m)), 'k-', lw=1, label="Neon: 1 − m/(7(8−m)), evenly over m, zero on target")
    la = [r for r in chk if r['cos_mean'] < 0]
    ax[1].scatter([r['m_eff_mean'] for r in la], [r['nmse'] for r in la], s=16, c=['#2ca02c' if r['success'] else '#d62728' for r in la], label='look-away runs (green = solved)')
    ax[1].set_xlabel('effective number of other slots the head spreads over, m_eff'); ax[1].set_ylabel('measured NMSE')
    ax[1].set_yscale('symlog', linthresh=1e-3); ax[1].legend(fontsize=8); ax[1].set_title('the look-away valley', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'figure.png', dpi=120)
except Exception as e:
    print('figure skipped:', e)
