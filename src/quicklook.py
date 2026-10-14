"""Quick summary of the agent probe (standard library only; the full analysis is done separately).
    python src/quicklook.py [--smoke] [models ...]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDS = ['C0_bare', 'C1_party', 'C2_grounded', 'C3_flipped', 'C4_swapped', 'D1_date_plus3y']
mean = lambda v: sum(v) / len(v) if v else float('nan')


def auc(scores, labels):
    """probability that a real 'w' case gets a higher P(w) than a real 'na' case (ties count half)"""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float('nan')
    ranked = sorted((s, i) for i, s in enumerate(scores))
    rank, i = [0.0] * len(scores), 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            rank[ranked[k][1]] = (i + j) / 2 + 1
        i = j + 1
    rpos = sum(r for r, l in zip(rank, labels) if l)
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main(models=None, smoke=False):
    sfx = '_smoke' if smoke else ''
    out = ROOT / 'outputs'
    cases = {c['case_id']: c for c in map(json.loads, open(ROOT / 'data' / 'cases.jsonl'))}
    if not models:
        models = sorted(p.stem[len('probe_'):len(p.stem) - len(sfx)] for p in out.glob(f'probe_*{sfx}.jsonl')
                        if smoke or not p.stem.endswith('_smoke'))
    for tag in models:
        path = out / f'probe_{tag}{sfx}.jsonl'
        if not path.exists():
            continue
        rows = [r for r in map(json.loads, open(path)) if r['case_id'] in cases and not cases[r['case_id']]['quoted']]
        P = {(r['case_id'], r['cond']): r for r in rows}
        print(f'== {tag}: {len(rows)} prompts | next-word mass on " w"+" na": {mean([r["mass1"] for r in rows]):.2f}')
        print(f'   {"condition":15s} {"n":>4s}  {"P(w)":>5s} {"real":>5s}  {"agree":>5s} {"AUC":>5s}   next-word only: P(w)  AUC')
        for cond in CONDS:
            rs = [r for r in rows if r['cond'] == cond]
            if not rs:
                continue
            lab = [cases[r['case_id']]['real'] == 'w' for r in rs]
            sw, s1 = [r['share_w'] for r in rs], [r['share1_w'] for r in rs]
            agree = mean([(s > 0.5) == l for s, l in zip(sw, lab)])
            print(f'   {cond:15s} {len(rs):4d}  {mean(sw):5.2f} {mean(lab):5.2f}  {agree:5.2f} {auc(sw, lab):5.2f}'
                  f'   {mean(s1):19.2f} {auc(s1, lab):5.2f}')
        by = defaultdict(lambda: ([], []))
        for r in rows:
            c = cases[r['case_id']]
            if r['cond'] == 'C2_grounded' and c['set'] == 'post':
                by[c['party']][0].append(r['share_w']); by[c['party']][1].append(c['real'] == 'w')
        if by:
            parts = sorted(by, key=lambda p: -mean(by[p][1]))
            print('   C2, after the invasion, by party (model P(w) / real share of w): ' +
                  ', '.join(f'{p} {mean(by[p][0]):.2f}/{mean(by[p][1]):.2f}' for p in parts))
        flip = []
        for (cid, cond), r in P.items():
            c = cases[cid]
            if cond == 'C3_flipped' and (cid, 'C2_grounded') in P and c['context_forms']:
                d = r['share_w'] - P[(cid, 'C2_grounded')]['share_w']
                towards_w = c['context_forms'].count('na') > c['context_forms'].count('w')
                flip.append(d if towards_w else -d)
        if flip:
            print(f'   flipping w/na in the earlier speeches (C3 vs C2) moves P(w) toward the flipped form by {mean(flip):+.3f} (n={len(flip)})')
        sw_ = defaultdict(list)
        for (cid, cond), r in P.items():
            if cond == 'C4_swapped' and (cid, 'C2_grounded') in P:
                c = cases[cid]
                sw_[f"{c['party']}->{c['swap_party']}"].append(r['share_w'] - P[(cid, 'C2_grounded')]['share_w'])
        if sw_:
            print('   party label swapped (C4 minus C2): ' + ', '.join(f'{k} {mean(v):+.3f}' for k, v in sorted(sw_.items()) if len(v) >= 5))
        d1 = [(P[(cid, 'D1_date_plus3y')]['share_w'], P[(cid, 'C2_grounded')]['share_w']) for (cid, cond) in P
              if cond == 'D1_date_plus3y' and (cid, 'C2_grounded') in P]
        if d1:
            print(f'   pre-invasion speeches: P(w) {mean([b for a, b in d1]):.3f} at the real date, {mean([a for a, b in d1]):.3f} with the date +3 years (n={len(d1)})')
    for tag in models:
        path = out / f'memo_{tag}{sfx}.jsonl'
        if path.exists():
            rows = list(map(json.loads, open(path)))
            g = {k: [x['mink20'] for x in rows if x['group'] == k] for k in ('2022-23', '2025')}
            print(f'== {tag} memorization (Min-K% 20, higher = more familiar): 2022-23 {mean(g["2022-23"]):.3f}, 2025 {mean(g["2025"]):.3f} (n={len(rows)})')


if __name__ == '__main__':
    args = sys.argv[1:]
    main([x for x in args if not x.startswith('--')] or None, '--smoke' in args)
