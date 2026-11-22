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
sys.path.insert(0, str(Path(__file__).parent)); from run_all import CELLS
order = [c[0] for c in CELLS if c[0] in cells]

summary = {}
lines = ['| cell | solved (NMSE<0.01) | <0.001 | read: at / neg-at / away / other | median NMSE | failed look-away runs: median m_eff, beta, NMSE, Neon curve at m |',
         '|---|---|---|---|---|---|']
for c in order:
    rs = sorted(cells[c], key=lambda r: r['seed'])
    n = len(rs)
    modes = {m: sum(r['route_mode'] == m for r in rs) for m in ('at', 'negative-at', 'away', 'other')}
    trap = [r for r in rs if r['route_mode'] == 'away' and not r['success']]
    med = lambda k: float(np.median([r[k] for r in trap])) if trap else None
    m = med('m_eff_mean')
    s = {'n': n, 'solved': sum(r['success'] for r in rs), 'solved_0.001': sum(r['nmse'] < 1e-3 for r in rs),
         'route_modes': modes, 'median_nmse': float(np.median([r['nmse'] for r in rs])),
         'trapped_look_away': len(trap), 'trapped_median_m_eff': m, 'trapped_median_beta': med('beta_eff_median'),
         'trapped_median_nmse': med('nmse'), 'neon_curve_at_trapped_m': (1 - m / (7 * (8 - m))) if m else None,
         'seeds': {r['seed']: [r['route_mode'], round(r['nmse'], 5)] for r in rs}}
    if rs[0]['kind'] == 'flashlight':
        s['teacher_all_correct_at_cutoff'] = sum(r.get('teacher_correct_at_cutoff') == 8 for r in rs)
    summary[c] = s
    f = lambda v, d=2: '–' if v is None else f'{v:.{d}f}'
    tr = '–' if not trap else f"{len(trap)} runs: m {f(m)}, β {f(s['trapped_median_beta'])}, NMSE {f(s['trapped_median_nmse'], 3)}, curve {f(s['neon_curve_at_trapped_m'], 3)}"
    lines.append(f"| {c} | {s['solved']}/{n} | {s['solved_0.001']}/{n} | {modes['at']} / {modes['negative-at']} / {modes['away']} / {modes['other']} | {s['median_nmse']:.4f} | {tr} |")

# formula checks on single-head softmax runs (A and B)
chk = [r for r in runs if r['kind'] in ('softmax', 'flashlight') and r['heads'] == 1 and r['detail'] == 'zero-sum' and r['polarity'] > 0 and r['aux_mode'] == 'unsigned']
meas = np.array([r['nmse'] for r in chk]); pred = np.array([r['pred_nmse_head0'] for r in chk])
summary['_formula'] = {
    'runs': len(chk),
    'corr_measured_vs_cos2_prediction': float(np.corrcoef(meas, pred)[0, 1]),
    'by_arch': {a: {'corr': float(np.corrcoef([r['nmse'] for r in chk if r['arch'] == a],
                                             [r['pred_nmse_head0'] for r in chk if r['arch'] == a])[0, 1]),
                    'median_abs_gap': float(np.median([abs(r['nmse'] - r['pred_nmse_head0']) for r in chk if r['arch'] == a]))}
                for a in ('attn-scalar', 'block-scalar', 'attn', 'block', 'self') if any(r['arch'] == a for r in chk)}}
(out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines)); print(json.dumps(summary['_formula'], indent=1))

try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    LAB = {'attn-scalar': "A' attention-only, scalar cue", 'block-scalar': "B' pre-LN block, scalar cue",
           'attn': 'A attention-only, one-hot cue', 'block': 'B pre-LN block, one-hot cue', 'self': 'C self-mean, 2 layers'}
    for a, col in (('attn-scalar', '#1f77b4'), ('block-scalar', '#d62728'), ('attn', '#9ecae1'), ('block', '#fcae91'), ('self', '#2ca02c')):
        rr = [r for r in chk if r['arch'] == a]
        if rr:
            ax[0].scatter([r['pred_nmse_head0'] for r in rr], [r['nmse'] for r in rr], s=14, c=col, label=LAB[a], alpha=.7)
    ax[0].plot([1e-5, 1.5], [1e-5, 1.5], 'k:', lw=1)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('predicted NMSE from the head\'s attention, 1 − cos²(a − ā, e_j − 1/8)')
    ax[0].set_ylabel('measured NMSE'); ax[0].legend(fontsize=8); ax[0].set_title('closed form vs trained network (softmax heads)', fontsize=10)
    m = np.linspace(1, 7, 100)
    ax[1].plot(m, 1 - m / (7 * (8 - m)), 'k-', lw=1, label="Neon: 1 − m/(7(8−m)), evenly over m, zero on target")
    la = [r for r in chk if r['route_mode'] == 'away']
    ax[1].scatter([r['m_eff_mean'] for r in la], [r['nmse'] for r in la], s=16, c=['#2ca02c' if r['success'] else '#d62728' for r in la], label='look-away runs (green = solved)')
    ax[1].set_xlabel('effective number of other slots the head spreads over, m_eff'); ax[1].set_ylabel('measured NMSE')
    ax[1].set_yscale('symlog', linthresh=1e-3); ax[1].legend(fontsize=8); ax[1].set_title('the look-away valley', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'figure.png', dpi=120)
except Exception as e:
    print('figure skipped:', e)
