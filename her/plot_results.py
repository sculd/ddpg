"""Plot eval success curves for the HER exploration-noise ablation.

    python her/plot_results.py  -> images/her_fetchreach_noise_ablation.png
"""
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# categorical slots 1..8 (validated palette from the dataviz skill), fixed order
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5cd6', '#8c8c8c']
NOISES = ['white', 'pink', 'red', 'ou']
NOISE_COLOR = dict(zip(NOISES, SLOTS))
NOISE_LABEL = {'white': 'white / i.i.d. Gaussian (paper)', 'pink': 'pink 1/f', 'red': 'red 1/f² (OU-like)', 'ou': 'OU'}


def load(pattern):
    runs = defaultdict(list)
    for f in sorted(glob.glob('exp_her/*.csv')):
        m = re.match(pattern, os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        if len(df) < 2:
            continue
        runs[m.groups()[:-1]].append(df)
    return runs


def plot_group(ax, dfs, color, label):
    n = min(len(d) for d in dfs)
    x = dfs[0]['env_steps'].values[:n] / 1000
    ys = np.stack([d['eval_success'].values[:n] for d in dfs])
    ax.plot(x, ys.mean(0), color=color, lw=2, label=f'{label}  (n={len(dfs)})')
    if len(dfs) > 1:
        ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)


def style(ax, title, xlabel='env steps (k)'):
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc='lower right', frameon=False)


fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2), gridspec_kw={'width_ratios': [1, 1.25]})
(a, b), (c, d) = axes

fine = load(r'fine_paper_(white|pink|red|ou)_(her|noher)_seed(\d+)\.csv')
for noise in NOISES:
    if (noise, 'her') in fine:
        plot_group(a, fine[(noise, 'her')], NOISE_COLOR[noise], NOISE_LABEL[noise])
style(a, 'DDPG + HER, paper hyper-parameters (10 episodes / eval point)')
for noise in ['white', 'pink']:
    if (noise, 'noher') in fine:
        plot_group(b, fine[(noise, 'noher')], NOISE_COLOR[noise], NOISE_LABEL[noise])
style(b, 'DDPG without HER, paper hyper-parameters (10 episodes / eval point)')

main = load(r'(paper|legacy)_(white|pink|red|ou)_(her|noher)_seed(\d+)\.csv')
for noise in NOISES:
    if ('legacy', noise, 'her') in main:
        plot_group(c, main[('legacy', noise, 'her')], NOISE_COLOR[noise], NOISE_LABEL[noise])
style(c, 'DDPG + HER, legacy hyper-parameters (original train_her.py settings)')

fixes = load(r'legacy_pink_her_(.+)_seed(\d+)\.csv')
FIX_LABEL = {'normalizeTrue': '+ obs/goal normalization', 'action_l21.0': '+ action L2 penalty',
             'lr_actor1e-3_lr_critic1e-3': '+ lr 1e-3 (both)', 'polyak0.95': '+ polyak 0.95',
             'clip_returnTrue': '+ target-Q clipping', 'random_eps0.3': '+ 30% random actions',
             'batch_size256': '+ batch 256', 'gamma0.98': '+ gamma 0.98'}
order = sorted(fixes, key=lambda k: -np.mean([df.eval_success.values[-3:].mean() for df in fixes[k]]))
if ('legacy', 'pink', 'her') in main:
    dfs = [df.iloc[:15] for df in main[('legacy', 'pink', 'her')]]
    plot_group(d, dfs, '#8c8c8c', 'legacy, no fix')
for i, k in enumerate(order):
    plot_group(d, fixes[k], SLOTS[i % len(SLOTS)], FIX_LABEL.get(k[0], k[0]))
style(d, 'legacy hyper-parameters + pink noise + ONE fix at a time')
d.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)

for ax in (a, c):
    ax.set_ylabel('eval success rate (10 deterministic episodes)')
fig.suptitle('FetchReach-v4 — does exploration-noise color matter for DDPG+HER?  (mean, min–max over seeds)', fontsize=12)
fig.tight_layout()
os.makedirs('images', exist_ok=True)
out = 'images/her_fetchreach_noise_ablation.png'
fig.savefig(out, dpi=130)
print('saved', out)

print('\nsummary (final 5 epochs eval success; epoch of first 100% eval; -1 = never)')
for name, runs in [('main', main), ('fine', fine), ('fix', fixes)]:
    for key, dfs in sorted(runs.items()):
        first = [int(df.loc[df.eval_success >= 1.0, 'epoch'].min()) if (df.eval_success >= 1.0).any() else -1 for df in dfs]
        final = np.mean([df.eval_success.values[-5:].mean() for df in dfs])
        print(f'{name:5} {"/".join(key):40} n={len(dfs)}  final={final:.3f}  first100%={first}')
