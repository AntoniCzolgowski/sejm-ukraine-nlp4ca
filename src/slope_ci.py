"""95% CI for the H1 slope: agent club-level log-odds of "w" (C2, after the invasion) regressed on the real club-level
log-odds (all Sejm uses), resampling speakers 1,000 times. Adds 'slope_ci' to outputs/agent_results.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2026)
cs = pd.read_json(ROOT / 'data' / 'cases.jsonl', lines=True)
spk = {}
for l in open(ROOT / 'data' / 'sejm_matches.jsonl'):
    r = json.loads(l); spk[r['doc'] + '#' + str(r['turn'])] = r['speaker']
cs['speaker'] = cs.case_id.map(spk)
res = json.load(open(ROOT / 'outputs' / 'agent_results.json'))
PARTIES = ['Lewica', 'Polska2050', 'KO', 'PSL', 'PiS', 'Kukiz15', 'Konfederacja']
hum = np.array([res['models']['bielik']['party_gradient']['table'][p]['human_all'] for p in PARTIES])
lg = lambda p: np.log(p / (1 - p))
out = {}
for mod in ['bielik', 'pllum', 'mistral', 'qwen']:
    P = pd.read_json(ROOT / 'results' / 'agent_run_2026-10-13' / f'probe_{mod}.jsonl', lines=True)
    P = P.merge(cs[['case_id', 'set', 'quoted', 'party', 'speaker']], on='case_id')
    P = P[(P.set == 'post') & (~P.quoted) & (P.cond == 'C2_grounded')].reset_index(drop=True)
    def slope(d):
        a = d.groupby('party').share_w.mean().reindex(PARTIES).values
        ok = ~np.isnan(a)
        return np.polyfit(lg(hum[ok]), lg(a[ok]), 1)[0]
    idx = list(P.groupby('speaker').indices.values())
    vals = [slope(P.iloc[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])]) for _ in range(1000)]
    out[mod] = [round(float(slope(P)), 3)] + [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])]
res['slope_ci'] = out
json.dump(res, open(ROOT / 'outputs' / 'agent_results.json', 'w'), indent=1, ensure_ascii=False)
print(out)
