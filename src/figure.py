"""Figure 1 of proposal 2: (a) share of "w Ukrainie" by party over time in the Sejm, (b) real vs agent share by party.
Reads outputs/human_halfyear_*.csv, outputs/agent_results.json. Writes proposal/fig1.pdf and proposal/fig1.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'proposal'; OUT.mkdir(exist_ok=True)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'legend.frameon': False})

fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.8), gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.22})
for ax in (a, b):
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 0.98); ax.set_yticks([0, .2, .4, .6, .8]); ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%'])

# (a) human, half-years
allh = pd.read_csv(ROOT / 'outputs' / 'human_halfyear_all.csv')
party = pd.read_csv(ROOT / 'outputs' / 'human_halfyear_party.csv')
halves = list(allh.half)
x = {h: i for i, h in enumerate(halves)}
a.plot([x[h] for h in allh.half], allh['mean'], color=INK, lw=2, marker='o', ms=3.2, label='all speakers', zorder=5)
PCOL = {'Lewica': '#e87ba4', 'KO': '#008300', 'PiS': '#6250d6', 'Konfederacja': '#e34948'}
PMARK = {'Lewica': 's', 'KO': 'D', 'PiS': '^', 'Konfederacja': 'v'}
for p, col in PCOL.items():
    d = party[(party.party == p) & (party['size'] >= 15) & (party.half >= '2022-H1')]
    a.plot([x[h] for h in d.half], d['mean'], color=col, lw=1.4, marker=PMARK[p], ms=3.2, label=p,
           markeredgecolor='white', markeredgewidth=0.4)
inv = x['2022-H1'] - 0.5
a.axvline(inv, color=INK2, lw=0.7, ls=(0, (3, 2)))
a.text(inv + 0.1, 0.97, 'invasion', va='top', ha='left', fontsize=6.5, color=INK2)
gov = x['2023-H2']
a.axvline(gov, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
a.text(gov + 0.1, 0.97, 'KO-led government', va='top', ha='left', fontsize=6.5, color=INK2)
a.set_xticks(range(len(halves))); a.set_xticklabels([h.replace('-', '\n') for h in halves], fontsize=6.3)
a.set_xlim(-0.4, len(halves) - 0.6)
a.set_ylabel('share of \u201cw Ukrainie\u201d')
a.set_title('(a) Real MPs, by half-year and club', loc='left', fontsize=8, color=INK)
a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.80), fontsize=6.5, handlelength=1.6, labelspacing=0.25)

# (b) agents vs real, by party, C2 grounded
res = json.load(open(ROOT / 'outputs' / 'agent_results.json'))
order = ['Lewica', 'Polska2050', 'KO', 'PSL', 'PiS', 'Kukiz15', 'Konfederacja']
labels = ['Lewica', 'Polska\n2050', 'KO', 'PSL', 'PiS', "Kukiz'15", 'Konfe-\nderacja']
human = [res['models']['bielik']['party_gradient']['table'][p]['human_all'] for p in order]
xs = range(len(order))
b.plot(xs, human, color=INK, lw=2, marker='o', ms=3.4, label='real MPs', zorder=5)
MCOL = {'bielik': '#2a78d6', 'pllum': '#eb6834', 'mistral': '#1baf7a', 'qwen': '#eda100'}
MMARK = {'bielik': 's', 'pllum': 'D', 'mistral': '^', 'qwen': 'v'}
MNAME = {'bielik': 'Bielik 11B', 'pllum': 'PLLuM 12B', 'mistral': 'Mistral-Nemo 12B', 'qwen': 'Qwen3 14B'}
for m, col in MCOL.items():
    t = res['models'][m]['party_gradient']['table']
    b.plot(xs, [t[p]['agent'] for p in order], color=col, lw=1.4, marker=MMARK[m], ms=3.2, label=MNAME[m],
           markeredgecolor='white', markeredgewidth=0.4)
b.set_xticks(list(xs)); b.set_xticklabels(labels, fontsize=6.3)
b.set_xlim(-0.4, len(order) - 0.6)
b.set_title('(b) Agent MPs (full prompt, C2) vs real MPs, after the invasion', loc='left', fontsize=8, color=INK)
b.legend(loc='upper right', fontsize=6.5, handlelength=1.6, labelspacing=0.25, ncol=1)

fig.savefig(OUT / 'fig1.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig1.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
print('saved', OUT / 'fig1.pdf')
