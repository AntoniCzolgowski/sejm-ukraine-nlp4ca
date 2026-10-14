"""Agent probe for proposal 2: P(" w") vs P(" na") at each real choice point, plus a memorization check.

Run from the notebook in a separate Python process (clean imports, packages from P2_LIB on PYTHONPATH):
    python src/score.py --models bielik,pllum,mistral,qwen [--smoke] [--batch 4]
Writes outputs/scores_<model>.jsonl and outputs/memo_<model>.jsonl line by line, so a stopped run resumes.
"""
import argparse, gc, json, math, os, sys, time
from pathlib import Path

MODELS = {
    'bielik':  'speakleash/Bielik-11B-v2.3-Instruct',       # Polish-native (SpeakLeash)
    'pllum':   'CYFRAGOVPL/PLLuM-12B-chat-2512',            # Polish, government-backed (PLLuM consortium)
    'mistral': 'mistralai/Mistral-Nemo-Instruct-2407',      # European, multilingual
    'qwen':    'Qwen/Qwen3-14B',                            # global
    'tiny':    'Qwen/Qwen2.5-0.5B-Instruct',                # only for testing the pipeline
}

ap = argparse.ArgumentParser()
ap.add_argument('--models', default='bielik,pllum,mistral,qwen')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--device', default='cuda')
ap.add_argument('--skip-memo', action='store_true')
a = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
say = lambda *x: print(*x, flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
DTYPE = torch.bfloat16 if a.device == 'cuda' else torch.float32
say(f'python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__}')
if a.device == 'cuda':
    if not torch.cuda.is_available():
        sys.exit('No GPU visible: relaunch the Jupyter Session on a GPU partition (see the top of the notebook)')
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    say(f'GPU: {torch.cuda.get_device_name(0)} | {gb:.0f} GB')
    if gb < 30:
        sys.exit(f'This slice has {gb:.0f} GB; the 12-14B models in bf16 need at least ~35 GB. Pick a bigger slice.')

cases = [json.loads(l) for l in open(ROOT / 'data' / 'cases.jsonl')]
if a.smoke:
    cases = [c for c in cases if c['set'] == 'post'][:6] + [c for c in cases if c['set'] == 'pre'][:2]
jobs = [(c['case_id'], cond, user, c['prefix']) for c in cases for cond, user in c['prompts'].items()]
say(f'{len(cases)} cases | {len(jobs)} prompts per model')

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

def load_model(tag, name):
    t0 = time.time()
    say(f'[{tag}] loading {name} (first time: downloading ~20-30 GB into the cache, several minutes)')
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE, device_map=a.device, low_cpu_mem_usage=True)
    m.eval()
    say(f'[{tag}] loaded in {time.time() - t0:.0f}s')
    return m

def templated(tok, user, prefix):
    msgs = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': prefix}]
    return tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)

def next_ids(tok, text):
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

def score(tag, name, tok, model_box):
    path = OUT / f'scores_{tag}.jsonl'
    done = {(r['case_id'], r['cond']) for r in map(json.loads, open(path))} if path.exists() else set()
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    say(f'[{tag}] scores: {len(done)} done, {len(todo)} to go')
    if not todo:
        return
    texts = [templated(tok, u, p) for _, _, u, p in todo]
    cids = [next_ids(tok, t) for t in texts]
    if model_box[0] is None:
        model_box[0] = load_model(tag, name)
    model = model_box[0]
    order = sorted(range(len(todo)), key=lambda i: len(texts[i]))
    t0, bs = time.time(), a.batch
    with open(path, 'a') as f:
        b = 0
        while b < len(order):
            idx = order[b:b + bs]
            enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, add_special_tokens=False).to(a.device)
            try:
                with torch.no_grad():
                    lp = torch.log_softmax(forward_last(model, enc), -1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2); say(f'[{tag}] out of GPU memory, batch -> {bs}')
                continue
            for k, i in enumerate(idx):
                cid, cond, _, _ = todo[i]
                lw, ln = lp[k, cids[i]['w']].item(), lp[k, cids[i]['na']].item()
                f.write(json.dumps({'case_id': cid, 'cond': cond, 'model': tag, 'logp_w': lw, 'logp_na': ln,
                                    'share_w': 1 / (1 + math.exp(ln - lw)), 'mass': math.exp(lw) + math.exp(ln)}) + '\n')
            f.flush()
            b += len(idx)
            if b % 200 < len(idx) or b == len(order):
                rate = b / (time.time() - t0)
                say(f'[{tag}] {b}/{len(order)}  {rate:.1f} prompts/s  ~{(len(order) - b) / max(rate, 1e-9) / 60:.0f} min left')

def memo(tag, name, tok, model_box):
    rows = [json.loads(l) for l in open(ROOT / 'data' / 'memo_excerpts.jsonl')]
    if a.smoke:
        rows = rows[:5] + rows[-5:]
    path = OUT / f'memo_{tag}.jsonl'
    if path.exists() and sum(1 for _ in open(path)) >= len(rows):
        say(f'[{tag}] memorization check: done'); return
    if model_box[0] is None:
        model_box[0] = load_model(tag, name)
    model = model_box[0]
    with open(path, 'w') as f:
        for r in rows:
            ids = tok(r['text'], return_tensors='pt').input_ids.to(a.device)
            with torch.no_grad():
                lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
            tl = lp.gather(1, ids[0, 1:, None])[:, 0]
            k = max(1, int(0.2 * len(tl)))
            f.write(json.dumps({'model': tag, 'group': r['group'], 'date': r['date'], 'mean_logp': tl.mean().item(),
                                'mink20': tl.topk(k, largest=False).values.mean().item()}) + '\n')
    out = [json.loads(l) for l in open(path)]
    for g in ('2022-23', '2025'):
        v = [x['mink20'] for x in out if x['group'] == g]
        say(f'[{tag}] memorization check {g}: Min-K% {sum(v) / len(v):.3f} (n={len(v)})')

for tag in a.models.split(','):
    name = MODELS[tag]
    tok = load_tok(name)
    box = [None]
    score(tag, name, tok, box)
    if not a.skip_memo:
        memo(tag, name, tok, box)
    if box[0] is not None:
        del box[0]
    gc.collect()
    if a.device == 'cuda':
        torch.cuda.empty_cache()
say('SCORING FINISHED')
