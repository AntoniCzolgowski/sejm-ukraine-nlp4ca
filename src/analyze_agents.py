"""Agent-side analysis for proposal 2: how close do LLM agent MPs come to the real Sejm?

Reads results/agent_run_2026-10-13/probe_<model>.jsonl and memo_<model>.jsonl (from src/score.py on Alpine),
data/cases.jsonl and data/sejm_matches.jsonl. Writes outputs/agent_results.json and outputs/agent_tables.md.
P(w) below is always P("w Ukrainie") / (P("w Ukrainie") + P("na Ukrainie")) at the real choice point.
Confidence intervals: 1,000 bootstrap resamples of speakers (an MP's speeches stay together).
"""
import json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results' / 'agent_run_2026-10-13'
MODELS = ['bielik', 'pllum', 'mistral', 'qwen']
NAMES = {'bielik': 'Bielik 11B v2.3', 'pllum': 'PLLuM 12B', 'mistral': 'Mistral-Nemo 12B', 'qwen': 'Qwen3 14B'}
CONDS = ['C0_bare', 'C1_party', 'C2_grounded', 'C3_flipped', 'C4_swapped', 'D1_date_plus3y']
PARTIES = ['Lewica', 'Polska2050', 'KO', 'PSL', 'PiS', 'Kukiz15', 'Konfederacja']
B = 1000
rng = np.random.default_rng(2026)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# ---------- data
cases = pd.DataFrame([json.loads(l) for l in open(ROOT / 'data' / 'cases.jsonl')])
matches = pd.DataFrame([json.loads(l) for l in open(ROOT / 'data' / 'sejm_matches.jsonl')])
ua = matches[matches.country == 'Ukraina'].copy()
ua['case_id'] = ua.doc + '#' + ua.turn.astype(str)
spk = ua.drop_duplicates('case_id').set_index('case_id').speaker
cases['speaker'] = cases.case_id.map(spk)
cases['y'] = (cases.real == 'w').astype(int)
cases = cases[~cases.quoted & (cases.prefix.str.strip() != '')]          # quoted forms = talking about the word itself

rows = []
for m in MODELS:
    rows += [json.loads(l) for l in open(RUN / f'probe_{m}.jsonl')]
P = pd.DataFrame(rows).merge(cases[['case_id', 'y', 'party', 'period', 'set', 'speaker', 'swap_party', 'context_forms',
                                    'prev_other_form', 'date']], on='case_id')
P['lg'] = P.lp_w - P.lp_na                                                  # log-odds of "w" for the agent
P['lg1'] = P.lp1_w - P.lp1_na

# ---------- human baselines, leave-the-case-out: party x period rate, and the speaker's own earlier uses
post_ua = ua[(ua.date >= '2022-02-24') & ua.party.notna()].copy()
post_ua['period'] = np.where(post_ua.date < '2023-11-13', 'PiS-gov', 'KO-gov')
post_ua['w'] = (post_ua.prep == 'w').astype(int)
grp = post_ua.groupby(['party', 'period']).w.agg(['sum', 'count'])
first_in_turn = post_ua.sort_values('offset').drop_duplicates('case_id')

def party_rate(c):
    s, n = grp.loc[(c.party, c.period)] if (c.party, c.period) in grp.index else (0, 0)
    own = first_in_turn[first_in_turn.case_id == c.case_id].w.sum()
    return (s - own + 1) / (n - 1 + 2)                                      # leave this case out, add-one smoothing

def own_history(c):
    h = post_ua[(post_ua.speaker == c.speaker) & (post_ua.date < c.date)]
    return (h.w.sum() + 1) / (len(h) + 2) if len(h) else np.nan

post_cases = cases[cases.set == 'post'].copy()
post_cases['base_party'] = post_cases.apply(party_rate, axis=1)
post_cases['base_own'] = post_cases.apply(own_history, axis=1)

# ---------- helpers
def boot(df, stat, by='speaker'):
    """point estimate and 95% CI, resampling speakers"""
    df = df.reset_index(drop=True)
    idx = list(df.groupby(by, dropna=False).indices.values())
    est, vals = stat(df), []
    for _ in range(B):
        pick = rng.integers(0, len(idx), len(idx))
        try:
            v = stat(df.iloc[np.concatenate([idx[i] for i in pick])])
        except ValueError:
            continue
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(est), 3), round(float(lo), 3), round(float(hi), 3)]

auc = lambda col: (lambda d: roc_auc_score(d.y, d[col]) if d.y.nunique() == 2 else float('nan'))

def within_party_auc(col):
    def f(d):
        vals, ws = [], []
        for _, g in d.groupby('party'):
            if g.y.nunique() == 2:
                vals.append(roc_auc_score(g.y, g[col])); ws.append(len(g))
        return np.average(vals, weights=ws) if vals else float('nan')
    return f

res = {'n_cases': int(len(cases)), 'n_post': int((cases.set == 'post').sum()), 'n_pre': int((cases.set == 'pre').sum()),
       'bootstrap': B, 'models': {}, 'baselines': {}}

# baselines on the post cases
for col, label in [('base_party', 'party x government rate (human data, leave-one-out)'),
                   ('base_own', "the MP's own earlier uses")]:
    d = post_cases.dropna(subset=[col])
    res['baselines'][col] = {'label': label, 'n': int(len(d)), 'auc_post': boot(d, auc(col)),
                             'within_party_auc': boot(d, within_party_auc(col)) if col == 'base_own' else None}

for m in MODELS:
    pm, out = P[P.model == m], {}
    for cond in CONDS:
        d = pm[pm.cond == cond]
        da = d[d.set == 'post'] if cond != 'D1_date_plus3y' else d.iloc[:0]   # AUC on post-invasion cases (as the baselines)
        out[cond] = {'n': int(len(d)), 'mean_pw': round(float(d.share_w.mean()), 3), 'real_w': round(float(d.y.mean()), 3),
                     'accuracy': round(float(((d.share_w > .5) == d.y).mean()), 3),
                     'brier': round(float(((d.share_w - d.y) ** 2).mean()), 3),
                     'auc_post': boot(da, auc('share_w')) if da.y.nunique() == 2 else None,
                     'auc_post_next_word': round(float(roc_auc_score(da.y, da.share1_w)), 3) if da.y.nunique() == 2 else None,
                     'mean_pw_next_word': round(float(d.share1_w.mean()), 3)}
        if cond in ('C1_party', 'C2_grounded'):
            dd = d[d.set == 'post']
            out[cond]['within_party_auc'] = boot(dd, within_party_auc('share_w'))
    # party gradient, C2, after the invasion
    c2 = pm[(pm.cond == 'C2_grounded') & (pm.set == 'post')]
    tab = c2.groupby('party').agg(agent=('share_w', 'mean'), real=('y', 'mean'), n=('y', 'size')).reindex(PARTIES)
    human = {p: post_ua[post_ua.party == p].w.mean() for p in PARTIES}
    tab['human_all'] = pd.Series(human)
    x, y = logit(tab.human_all.values), logit(tab.agent.values)
    slope = np.polyfit(x, y, 1)[0]
    rho = pd.Series(tab.agent.values).rank().corr(pd.Series(tab.human_all.values).rank())
    out['party_gradient'] = {'table': {p: {k: round(float(v), 3) for k, v in tab.loc[p].items()} for p in PARTIES},
                             'spearman_rho': round(float(rho), 3), 'logit_slope_vs_human': round(float(slope), 3),
                             'agent_range': round(float(tab.agent.max() - tab.agent.min()), 3),
                             'human_range': round(float(tab.human_all.max() - tab.human_all.min()), 3)}
    # context flip (C3 vs C2): signed log-odds shift toward the flipped form
    w2 = pm.pivot_table(index='case_id', columns='cond', values='lg')
    meta = cases.set_index('case_id')
    fl = []
    for cid, r in w2.dropna(subset=['C2_grounded', 'C3_flipped']).iterrows():
        f = meta.loc[cid, 'context_forms']
        if f:
            sign = 1 if f.count('na') > f.count('w') else -1 if f.count('w') > f.count('na') else 0
            if sign:
                fl.append({'case_id': cid, 'speaker': meta.loc[cid, 'speaker'], 'd': sign * (r.C3_flipped - r.C2_grounded)})
    fl = pd.DataFrame(fl)
    e = boot(fl, lambda d: d.d.mean())
    out['context_flip'] = {'n': int(len(fl)), 'mean_logodds_shift': e, 'odds_ratio': [round(math.exp(v), 2) for v in e]}
    # party swap (C4 vs C2) against the real gap between the two parties
    sw = []
    for cid, r in w2.dropna(subset=['C2_grounded', 'C4_swapped']).iterrows():
        a, b = meta.loc[cid, 'party'], meta.loc[cid, 'swap_party']
        gap = logit(np.array([human[b]]))[0] - logit(np.array([human[a]]))[0]
        sw.append({'case_id': cid, 'speaker': meta.loc[cid, 'speaker'], 'd': r.C4_swapped - r.C2_grounded, 'gap': gap,
                   'pair': f'{a}->{b}'})
    sw = pd.DataFrame(sw)
    out['party_swap'] = {'n': int(len(sw)),
                         'share_of_real_gap': boot(sw, lambda d: (d.d * np.sign(d.gap)).mean() / d.gap.abs().mean()),
                         'by_pair': {k: {'n': int(len(g)), 'agent_shift': round(float(g.d.mean()), 3),
                                         'real_gap': round(float(g.gap.mean()), 3)} for k, g in sw.groupby('pair')}}
    # date moved +3 years on pre-invasion speeches
    dd = w2.dropna(subset=['C2_grounded', 'D1_date_plus3y']).reset_index()
    dd['speaker'] = dd.case_id.map(meta.speaker)
    dd['d'] = dd.D1_date_plus3y - dd.C2_grounded
    out['date_shift'] = {'n': int(len(dd)), 'mean_logodds_shift': boot(dd, lambda d: d.d.mean()),
                         'pw_real_date': round(float(pm[(pm.cond == 'C2_grounded') & (pm.set == 'pre')].share_w.mean()), 3),
                         'pw_plus3y': round(float(pm[pm.cond == 'D1_date_plus3y'].share_w.mean()), 3),
                         'human_pre_share': 0.014}
    # memorization: Min-K% (20%) of 2022-23 vs 2025 excerpts
    mm = pd.DataFrame([json.loads(l) for l in open(RUN / f'memo_{m}.jsonl')])
    a_, b_ = mm[mm.group == '2022-23'].mink20.values, mm[mm.group == '2025'].mink20.values
    diffs = [rng.choice(a_, len(a_)).mean() - rng.choice(b_, len(b_)).mean() for _ in range(B)]
    out['memorization'] = {'mink20_2022_23': round(float(a_.mean()), 3), 'mink20_2025': round(float(b_.mean()), 3),
                           'diff': [round(float(a_.mean() - b_.mean()), 3)] + [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])],
                           'n': [int(len(a_)), int(len(b_))]}
    out['next_word_mass'] = round(float(pm.mass1.mean()), 3)
    res['models'][m] = out

res['human'] = json.load(open(ROOT / 'outputs' / 'human_results.json'))['prev_speaker']
(ROOT / 'outputs' / 'agent_results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))

# ---------- readable tables
L = ['# Agent MPs vs the real Sejm: results (run of 13 Oct 2026)', '',
     f"{res['n_post']} choice points after the invasion, {res['n_pre']} before; 95% CIs from {B} speaker bootstrap resamples.", '',
     '## Telling real "w" from real "na" speeches after the invasion (AUC, 0.5 = chance)', '',
     '| model | C0 bare | C1 party | C2 grounded | C2 within party | C4 swapped | next-word C2 |', '|---|---|---|---|---|---|---|']
f = lambda t: f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
for m in MODELS:
    o = res['models'][m]
    L.append(f"| {NAMES[m]} | {f(o['C0_bare']['auc_post'])} | {f(o['C1_party']['auc_post'])} | {f(o['C2_grounded']['auc_post'])} | "
             f"{f(o['C2_grounded']['within_party_auc'])} | {f(o['C4_swapped']['auc_post'])} | {o['C2_grounded']['auc_post_next_word']:.2f} |")
for k, v in res['baselines'].items():
    L.append(f"| baseline: {v['label']} (n={v['n']}) | | | {f(v['auc_post'])} | {f(v['within_party_auc']) if v['within_party_auc'] else ''} | | |")
L += ['', '## Party gradient (C2, after the invasion): agent P(w) vs real share of "w" (all Sejm uses)', '',
      '| party | human | ' + ' | '.join(NAMES[m] for m in MODELS) + ' |', '|---|---|' + '---|' * len(MODELS)]
for p in PARTIES:
    L.append(f'| {p} | {res["models"]["bielik"]["party_gradient"]["table"][p]["human_all"]:.2f} | ' +
             ' | '.join(f'{res["models"][m]["party_gradient"]["table"][p]["agent"]:.2f}' for m in MODELS) + ' |')
L.append('| Spearman rho / logit slope | | ' + ' | '.join(
    f'{res["models"][m]["party_gradient"]["spearman_rho"]:.2f} / {res["models"][m]["party_gradient"]["logit_slope_vs_human"]:.2f}' for m in MODELS) + ' |')
L += ['', '## Experiments (log-odds shifts, 95% CI)', '',
      '| model | context flip: OR toward flipped form (n) | party swap: share of real party gap | date +3y on pre-invasion speeches | Min-K% 2022-23 minus 2025 |',
      '|---|---|---|---|---|']
for m in MODELS:
    o = res['models'][m]
    L.append(f"| {NAMES[m]} | {o['context_flip']['odds_ratio'][0]:.2f} [{o['context_flip']['odds_ratio'][1]:.2f}, {o['context_flip']['odds_ratio'][2]:.2f}] ({o['context_flip']['n']}) | "
             f"{f(o['party_swap']['share_of_real_gap'])} | {f(o['date_shift']['mean_logodds_shift'])} (P(w) {o['date_shift']['pw_real_date']:.2f} -> {o['date_shift']['pw_plus3y']:.2f}) | "
             f"{f(o['memorization']['diff'])} |")
h = res['human']
L += ['', f"Human previous-speaker effect (observational, all Sejm uses): OR {h['odds_ratio']} [{h['ci'][0]}, {h['ci'][1]}], n={h['n']}."]
(ROOT / 'outputs' / 'agent_tables.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
