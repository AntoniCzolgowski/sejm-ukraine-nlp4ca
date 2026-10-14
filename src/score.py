"""Agent probe for proposal 2: at each real choice point, does the model continue the MP's speech with
"w Ukrainie" or "na Ukrainie"? Plus a memorization check.

We score the two full continuations exactly as they would sit in the transcript (field `cont` in cases.jsonl:
" w Ukrainie" vs " na Ukrainie", capitalised after a full stop), not only the next word, because the next word alone
also counts continuations like "w Polsce" or "na froncie". The prompt runs once and its KV cache is reused for the two
short continuations, so this costs about one forward pass per prompt. The next-word numbers are kept for comparison.

Run from the notebook in a separate Python process (clean imports, packages from P2_LIB on PYTHONPATH):
    python -S src/score.py --models bielik,pllum,mistral,qwen [--smoke] [--batch 4]
Writes outputs/probe_<model>.jsonl and outputs/memo_<model>.jsonl line by line (..._smoke.jsonl in smoke mode),
so a stopped run resumes where it stopped. Prints a summary at the end (src/quicklook.py).
"""
import argparse, copy, gc, json, math, sys, time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    'bielik':  'speakleash/Bielik-11B-v2.3-Instruct',       # Polish-native (SpeakLeash)
    'pllum':   'CYFRAGOVPL/PLLuM-12B-chat-2512',            # Polish, government-backed (PLLuM consortium)
    'mistral': 'mistralai/Mistral-Nemo-Instruct-2407',      # European, multilingual
    'qwen':    'Qwen/Qwen3-14B',                            # global
    'tiny':    'Qwen/Qwen2.5-0.5B-Instruct',                # only for testing the pipeline
}
PREPS = ('w', 'na')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
say = lambda *x: print(*x, flush=True)


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


def cont_ids(tok, text, cont):
    """token ids of each continuation, cut from the tokenization of prompt + continuation (so the boundary is exact)"""
    base = tok(text, add_special_tokens=False).input_ids
    out = {}
    for prep in PREPS:
        full = tok(text + cont[prep], add_special_tokens=False).input_ids
        if full[:len(base)] != base or len(full) == len(base):
            raise ValueError(f'tokenizer merges across the choice point: {text[-60:]!r} + {cont[prep]!r}')
        out[prep] = full[len(base):]
    return out


@torch.no_grad()
def score_batch(model, tok, texts, conts, device):
    """log P(continuation | prompt) for both forms, plus log P(first token) (the old next-word measure)"""
    enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
    am = enc['attention_mask']
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    try:
        out = model(input_ids=enc['input_ids'], attention_mask=am, position_ids=pos, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=enc['input_ids'], attention_mask=am, position_ids=pos, use_cache=True)
    first = torch.log_softmax(out.logits[:, -1, :].float(), -1)
    cache, B, res = out.past_key_values, len(texts), {}
    del out
    for j, prep in enumerate(PREPS):
        seqs = [c[prep] for c in conts]
        k = max(len(s) for s in seqs)
        ids = torch.full((B, k), tok.pad_token_id, dtype=torch.long)
        valid = torch.zeros((B, k), dtype=torch.bool)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            valid[r, :len(s)] = True
        ids, valid = ids.to(device), valid.to(device)
        lp1 = first.gather(1, ids[:, :1])[:, 0]
        total = lp1.clone()
        if k > 1:
            c = copy.deepcopy(cache) if j < len(PREPS) - 1 else cache        # the prompt's cache, reused per form
            am2 = torch.cat([am, torch.ones((B, k - 1), dtype=am.dtype, device=device)], 1)
            pos2 = am.sum(-1, keepdim=True) + torch.arange(k - 1, device=device)[None]
            lg = model(input_ids=ids[:, :k - 1], attention_mask=am2, position_ids=pos2, past_key_values=c,
                       use_cache=True).logits.float()
            step = torch.log_softmax(lg, -1).gather(2, ids[:, 1:, None])[..., 0]   # position t predicts token t+1
            total = total + (step * valid[:, 1:]).sum(1)
            del c, lg
        res[prep] = (total.tolist(), lp1.tolist(), [len(s) for s in seqs])
    del cache
    return res


def load_model(tag, name, device):
    t0 = time.time()
    say(f'[{tag}] loading {name}')
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True)
    m.eval()
    say(f'[{tag}] loaded in {time.time() - t0:.0f}s')
    return m


def read_rows(path):
    """rows of a checkpoint file; drops a last line cut off by a stopped run"""
    if not path.exists():
        return []
    raw = open(path).read()
    rows, lines = [], raw.splitlines()
    for l in lines:
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    if len(rows) != len(lines) or (raw and not raw.endswith('\n')):
        with open(path, 'w') as f:
            f.writelines(json.dumps(r) + '\n' for r in rows)
    return rows


def probe(tag, name, tok, box, jobs, a, sfx):
    path = OUT / f'probe_{tag}{sfx}.jsonl'
    done = {(r['case_id'], r['cond']) for r in read_rows(path)}
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    say(f'[{tag}] {len(done)} prompts done, {len(todo)} to go')
    if not todo:
        return
    texts = [templated(tok, j[2], j[3]) for j in todo]
    conts = [cont_ids(tok, t, j[4]) for t, j in zip(texts, todo)]
    ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts[:: max(1, len(texts) // 50)]]
    say(f'[{tag}] prompt length ~{sorted(ntok)[len(ntok) // 2]} tokens (max seen {max(ntok)}) | '
        f'" w Ukrainie" = {len(conts[0]["w"])} tokens, " na Ukrainie" = {len(conts[0]["na"])} tokens')
    if box[0] is None:
        box[0] = load_model(tag, name, a.device)
    model = box[0]
    order = sorted(range(len(todo)), key=lambda i: len(texts[i]))
    t0, bs, b, first = time.time(), a.batch, 0, True
    with open(path, 'a') as f:
        while b < len(order):
            idx = order[b:b + bs]
            try:
                res = score_batch(model, tok, [texts[i] for i in idx], [conts[i] for i in idx], a.device)
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2); say(f'[{tag}] out of GPU memory, batch -> {bs}')
                continue
            for k, i in enumerate(idx):
                cid, cond = todo[i][0], todo[i][1]
                lw, ln = res['w'][0][k], res['na'][0][k]
                l1w, l1n = res['w'][1][k], res['na'][1][k]
                row = {'case_id': cid, 'cond': cond, 'model': tag,
                       'lp_w': lw, 'lp_na': ln, 'share_w': 1 / (1 + math.exp(ln - lw)),
                       'lp1_w': l1w, 'lp1_na': l1n, 'share1_w': 1 / (1 + math.exp(l1n - l1w)),
                       'mass1': math.exp(l1w) + math.exp(l1n), 'ntok_w': res['w'][2][k], 'ntok_na': res['na'][2][k]}
                f.write(json.dumps(row) + '\n')
            f.flush()
            if first:
                sw = [1 / (1 + math.exp(n - w)) for w, n in zip(res['w'][0], res['na'][0])]
                say(f'[{tag}] first batch ok: P(w Ukrainie) share {", ".join(f"{x:.2f}" for x in sw)}')
                first = False
            b += len(idx)
            if b % 200 < len(idx) or b == len(order):
                rate = b / (time.time() - t0)
                say(f'[{tag}] {b}/{len(order)}  {rate:.1f} prompts/s  ~{(len(order) - b) / max(rate, 1e-9) / 60:.0f} min left')


@torch.no_grad()
def memo(tag, name, tok, box, a, sfx):
    rows = [json.loads(l) for l in open(ROOT / 'data' / 'memo_excerpts.jsonl')]
    if a.smoke:
        rows = rows[:5] + rows[-5:]
    path = OUT / f'memo_{tag}{sfx}.jsonl'
    if len(read_rows(path)) >= len(rows):
        say(f'[{tag}] memorization check: done'); return
    if box[0] is None:
        box[0] = load_model(tag, name, a.device)
    model = box[0]
    with open(path, 'w') as f:
        for r in rows:
            ids = tok(r['text'], return_tensors='pt').input_ids.to(a.device)
            lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
            tl = lp.gather(1, ids[0, 1:, None])[:, 0]
            k = max(1, int(0.2 * len(tl)))
            f.write(json.dumps({'model': tag, 'group': r['group'], 'date': r['date'], 'mean_logp': tl.mean().item(),
                                'mink20': tl.topk(k, largest=False).values.mean().item()}) + '\n')
    out = read_rows(path)
    for g in ('2022-23', '2025'):
        v = [x['mink20'] for x in out if x['group'] == g]
        say(f'[{tag}] memorization check {g}: Min-K% {sum(v) / len(v):.3f} (n={len(v)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='bielik,pllum,mistral,qwen')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip-memo', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sfx = '_smoke' if a.smoke else ''
    say(f'python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__}')
    if a.device == 'cuda':
        if not torch.cuda.is_available():
            sys.exit('No GPU visible: relaunch the Jupyter Session on a GPU partition (see the top of the notebook)')
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        say(f'GPU: {torch.cuda.get_device_name(0)} | {gb:.0f} GB')
        if gb < 30:
            sys.exit(f'This slice has {gb:.0f} GB; the 12-14B models in bf16 need at least ~35 GB. Pick a bigger slice.')
    cases = [json.loads(l) for l in open(ROOT / 'data' / 'cases.jsonl')]
    empty = [c['case_id'] for c in cases if not c['prefix'].strip()]
    cases = [c for c in cases if c['prefix'].strip()]          # a choice at the very first word has nothing to continue
    if a.smoke:
        cases = [c for c in cases if c['set'] == 'post'][:6] + [c for c in cases if c['set'] == 'pre'][:2]
    jobs = [(c['case_id'], cond, user, c['prefix'], c['cont']) for c in cases for cond, user in c['prompts'].items()]
    say(f'{len(cases)} cases | {len(jobs)} prompts per model' + (f' | skipped {len(empty)} with no words before the choice' if empty else ''))
    models = a.models.split(',')
    for tag in models:
        name = MODELS[tag]
        tok = load_tok(name)
        box = [None]
        probe(tag, name, tok, box, jobs, a, sfx)
        if not a.skip_memo:
            memo(tag, name, tok, box, a, sfx)
        box[0] = None
        gc.collect()
        if a.device == 'cuda':
            torch.cuda.empty_cache()
    say('SCORING FINISHED\n')
    import quicklook
    quicklook.main(models, a.smoke)


if __name__ == '__main__':
    main()
