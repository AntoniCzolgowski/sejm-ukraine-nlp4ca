"""Builds notebooks/agents_alpine.ipynb from the cell sources below (edit here, then rerun)."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# W czy na Ukrainie? Agent MPs vs the real Sejm (NLP4CA proposal 2, feasibility test)

After Russia's full-scale invasion, some Polish MPs switched from **"na Ukrainie"** to **"w Ukrainie"** and others did not.
In 4,068 real uses in the Sejm (Nov 2019 to Apr 2025), "w" went from 0% before the invasion to about 31% after it,
with a clear party gradient. This notebook asks whether LLM **agent MPs** make the same choice.

For 893 real moments where an MP had to choose (806 after the invasion, 87 before), each model reads a prompt in four layers
(scene, who you are, what was said right before in the room, your own speech up to the choice point) and we read its
probability of **" w"** vs **" na"** as the next word. Conditions change one thing at a time:

| condition | what the agent gets |
|---|---|
| `C0_bare` | scene only (date, sitting, agenda item) |
| `C1_party` | scene + party |
| `C2_grounded` | scene + party + biography from the Sejm API + two of the MP's own earlier speeches + the real speeches before theirs |
| `C3_flipped` | C2, with every w/na Ukrainie in the preceding speeches flipped |
| `C4_swapped` | C2, with the party label swapped to the other camp |
| `D1_date_plus3y` | pre-invasion cases only: C2 with the date moved forward 3 years |

No MP names appear in the identity layer. Models run in bf16 with `transformers`; nothing is generated, so the run is fast.

### How to launch on Alpine (Open OnDemand)
1. https://ondemand.rc.colorado.edu → **Interactive Apps → Jupyter Session**
2. **Custom configuration**, Cluster `alpine`, Account `ucb757_asc1` (or `ucb-general`), Time `3`, cores `8`, and a GPU slice with **at least ~35 GB**
   (the 12-14B models run in bf16). Any of these works; queue two and delete the other once one starts:
   - Partition `artxpro6000`, QoS `gpu-normal`, gres `gpu:rtx_pro_6000_2g.48gb:1`
   - Partition `aa100`, QoS `gpu-normal`, gres `gpu:a100-40gb:1`
   - Partition `al40`, QoS `gpu-normal`, gres `gpu:l40:1`
3. Open this notebook from the cloned repo, keep `SMOKE = True`, **Run All** (first run installs packages into `/projects`, about 5 minutes,
   and downloads the first model into `/scratch`). Send Claude the last cell's output.
4. Set `SMOKE = False`, **Run All** again. Every result is written line by line to `outputs/`, so a dropped session just resumes.
5. Download `outputs/` (Files → select all → Download) and tell Claude where the zip landed.
""")

code(r"""
# (only once) clone the repo into /projects/$USER, then open notebooks/agents_alpine.ipynb from the file browser
# !cd /projects/$USER && git clone https://github.com/AntoniCzolgowski/sejm-ukraine-nlp4ca.git
# update later (keeps outputs/):
# !cd /projects/$USER/sejm-ukraine-nlp4ca && git fetch -q && git reset --hard origin/main
""")

md("## 1. Settings")
code(r"""
SMOKE = True        # True: 8 cases with the first model only (end-to-end check). Then False and Run All.

MODELS = {                                                  # all open weights, no Hugging Face login needed
    'bielik':  'speakleash/Bielik-11B-v2.3-Instruct',       # Polish-native (SpeakLeash)
    'pllum':   'CYFRAGOVPL/PLLuM-12B-chat-2512',            # Polish, government-backed (PLLuM consortium)
    'mistral': 'mistralai/Mistral-Nemo-Instruct-2407',      # European, multilingual
    'qwen':    'Qwen/Qwen3-14B',                            # global
}
BATCH = 4           # prompts per forward pass; lower to 2 if you see CUDA out-of-memory

import os, sys, json, time, math, gc, subprocess
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / 'data' / 'cases.jsonl').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / 'data' / 'cases.jsonl').exists(), 'Open this notebook from inside the cloned repo'
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
USER = os.environ.get('USER', 'me')
LIB = os.environ.get('P2_LIB', f'/projects/{USER}/p2_pylibs')          # packages live here (home quota is too small)
SCRATCH = f'/scratch/alpine/{USER}' if os.path.isdir(f'/scratch/alpine/{USER}') else f'/projects/{USER}'
os.environ['HF_HOME'] = os.environ.get('HF_HOME', f'{SCRATCH}/hf_cache')  # ~95 GB of model weights over the full run
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
if SMOKE:
    MODELS = dict(list(MODELS.items())[:1])
print('repo:', ROOT, '| smoke:', SMOKE, '| packages:', LIB, '| model cache:', os.environ['HF_HOME'])
""")

md("## 2. Python packages (first run only, about 5 minutes)")
code(r"""
def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-warn-script-location',
                    '--target', LIB, *args], check=True)

if not (Path(LIB) / 'transformers').exists():
    Path(LIB).mkdir(parents=True, exist_ok=True)
    # one resolver pass; the CUDA 12.8 build of PyTorch runs on the Blackwell RTX PRO 6000 slices as well as A100 and L40
    pip('torch==2.7.1', 'transformers==4.55.4', 'accelerate==1.10.1', 'sentencepiece', 'protobuf', 'safetensors',
        'huggingface_hub', '--index-url', 'https://download.pytorch.org/whl/cu128', '--extra-index-url', 'https://pypi.org/simple')
sys.path.insert(0, LIB)
import torch, transformers
print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__)
""")

md("## 3. GPU")
code(r"""
if not torch.cuda.is_available():
    raise RuntimeError('No GPU in this session: relaunch the Jupyter Session on a GPU partition (see the top of the notebook)')
GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f'| {GB:.0f} GB')
if GB < 30:
    raise RuntimeError(f'This slice has {GB:.0f} GB; the 12-14B models in bf16 need at least ~35 GB. Pick a bigger slice.')
""")

md("## 4. Cases and prompts")
code(r"""
cases = [json.loads(l) for l in open(ROOT / 'data' / 'cases.jsonl')]
if SMOKE:
    cases = [c for c in cases if c['set'] == 'post'][:6] + [c for c in cases if c['set'] == 'pre'][:2]
jobs = [(c['case_id'], cond, user, c['prefix']) for c in cases for cond, user in c['prompts'].items()]
print(len(cases), 'cases |', len(jobs), 'prompts per model')
ex = cases[0]
print('\n--- example (C2_grounded), user message ---\n' + ex['prompts']['C2_grounded'][:1500] + ' ...')
print('\n--- assistant message is pre-filled with the real speech up to the choice point ---\n... ' + ex['prefix'][-300:])
print('\nreal choice:', ex['real'])
""")

md("## 5. Scoring: P(\" w\") vs P(\" na\") as the next word")
code(r"""
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tok(name):
    kw = {'fix_mistral_regex': True} if 'mistral' in name.lower() else {}
    try:
        tok = AutoTokenizer.from_pretrained(name, **kw)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def templated(tok, user, prefix):
    msgs = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': prefix}]
    return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)

def next_ids(tok, text):
    # token id of " w" and " na" right after `text`, checked so that the prefix tokenizes the same way
    base = tok(text, add_special_tokens=False).input_ids
    out = {}
    for prep in ('w', 'na'):
        full = tok(text + f' {prep} Ukrainie', add_special_tokens=False).input_ids
        assert full[:len(base)] == base, 'tokenizer merges across the choice point'
        out[prep] = full[len(base)]
    return out

def forward_last(model, enc):
    pos = (enc['attention_mask'].cumsum(-1) - 1).clamp(min=0)
    try:
        return model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:
        return model(**enc, position_ids=pos).logits[:, -1, :].float()

def score_model(tag, name):
    path = OUT / f'scores_{tag}.jsonl'
    done = set()
    if path.exists():
        done = {(r['case_id'], r['cond']) for r in map(json.loads, open(path))}
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f'[{tag}] {name}: {len(done)} done, {len(todo)} to score')
    if not todo:
        return
    tok = load_tok(name)
    texts = [templated(tok, u, p) for _, _, u, p in todo]
    cids = [next_ids(tok, t) for t in texts]
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    print(f'[{tag}] loaded in {time.time() - t0:.0f}s')
    order = sorted(range(len(todo)), key=lambda i: len(texts[i]))
    t0 = time.time()
    with open(path, 'a') as f:
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.no_grad():
                lp = torch.log_softmax(forward_last(model, enc), -1)
            for k, i in enumerate(idx):
                cid, cond, _, _ = todo[i]
                lw, ln = lp[k, cids[i]['w']].item(), lp[k, cids[i]['na']].item()
                f.write(json.dumps({'case_id': cid, 'cond': cond, 'model': tag, 'logp_w': lw, 'logp_na': ln,
                                    'share_w': 1 / (1 + math.exp(ln - lw)), 'mass': math.exp(lw) + math.exp(ln)}) + '\n')
            f.flush()
            n = b + len(idx)
            if n % (BATCH * 100) < BATCH or n == len(order):
                rate = n / (time.time() - t0)
                print(f'[{tag}] {n}/{len(order)}  {rate:.1f} prompts/s  ~{(len(order) - n) / max(rate, 1e-9) / 60:.0f} min left', flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()

for tag, name in MODELS.items():
    score_model(tag, name)
""")

md(r"""
## 6. Memorization check
The Sejm transcripts are public, so a model may have read them. For 120 excerpts from mid-2022 to 2023 (probably in training data)
and 120 from January to April 2025 (after most of these models' training), we compute Min-K% (k = 20%), the mean log-probability
of the 20% least likely tokens (Shi et al. 2024). Similar scores for both groups suggest no verbatim memory of the older speeches.
""")
code(r"""
from transformers import AutoTokenizer, AutoModelForCausalLM
memo = [json.loads(l) for l in open(ROOT / 'data' / 'memo_excerpts.jsonl')]
if SMOKE:
    memo = memo[:5] + memo[-5:]
for tag, name in MODELS.items():
    path = OUT / f'memo_{tag}.jsonl'
    if path.exists() and sum(1 for _ in open(path)) >= len(memo):
        print(tag, 'done'); continue
    tok = load_tok(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map='cuda'); model.eval()
    with open(path, 'w') as f:
        for r in memo:
            ids = tok(r['text'], return_tensors='pt').input_ids.to('cuda')
            with torch.no_grad():
                lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
            tl = lp.gather(1, ids[0, 1:, None])[:, 0]
            k = max(1, int(0.2 * len(tl)))
            f.write(json.dumps({'model': tag, 'group': r['group'], 'date': r['date'],
                                'mean_logp': tl.mean().item(), 'mink20': tl.topk(k, largest=False).values.mean().item()}) + '\n')
    del model; gc.collect(); torch.cuda.empty_cache()
    rows = [json.loads(l) for l in open(path)]
    for g in ('2022-23', '2025'):
        v = [x['mink20'] for x in rows if x['group'] == g]
        print(f'{tag} {g}: Min-K% {sum(v) / len(v):.3f} (n={len(v)})')
""")

md("## 7. Quick look (the full analysis runs on Claude's side)")
code(r"""
from collections import defaultdict
real = {c['case_id']: c for c in cases}
for tag in MODELS:
    path = OUT / f'scores_{tag}.jsonl'
    if not path.exists():
        continue
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(list)
    for r in rows:
        c = real.get(r['case_id'])
        if c: by[(r['cond'], c['party'])].append(r['share_w'])
    acc = defaultdict(list)
    for r in rows:
        c = real.get(r['case_id'])
        if c: acc[r['cond']].append((r['share_w'] > 0.5) == (c['real'] == 'w'))
    print(f'== {tag}: {len(rows)} scores | mean probability mass on w+na: {sum(r["mass"] for r in rows) / len(rows):.2f}')
    for cond in sorted(acc):
        print(f'   {cond:16s} agreement with the real MP: {sum(acc[cond]) / len(acc[cond]):.2f} (n={len(acc[cond])})')
    for cond in ('C2_grounded',):
        print('   mean P(w) by party, ' + cond + ': ' + ', '.join(f'{p} {sum(v) / len(v):.2f}' for (c_, p), v in sorted(by.items()) if c_ == cond))
manifest = {'smoke': SMOKE, 'models': MODELS, 'n_cases': len(cases), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(0), 'finished': time.strftime('%Y-%m-%d %H:%M')}
json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
print('\nDONE', manifest)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}}
Path(__file__).resolve().parents[1].joinpath("notebooks", "agents_alpine.ipynb").write_text(nbf.writes(nb))
print("notebook written")
