"""Builds notebooks/agents_alpine.ipynb from the cell sources below (edit here, then rerun)."""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# W czy na Ukrainie? Agent MPs vs the real Sejm (NLP4CA proposal 2, feasibility test)

After Russia's full-scale invasion, some Polish MPs switched from **"na Ukrainie"** to **"w Ukrainie"** and others did not.
In 4,068 real uses in the Sejm (Nov 2019 to Apr 2025), "w" went from about 1% before the invasion to about 31% after it,
with a clear party gradient. This notebook asks whether LLM **agent MPs** make the same choice.

For 893 real moments where an MP had to choose (806 after the invasion, 87 before), each model reads a prompt in four layers
(scene, who you are, what was said right before in the room, your own speech up to the choice point) and we compare the
probability it gives to the two full continuations, **" w Ukrainie"** vs **" na Ukrainie"**, written exactly as they would sit in
the transcript. (Comparing only the next word, " w" vs " na", would also count "w Polsce" or "na froncie"; that number is
kept as a secondary column.) Conditions change one thing at a time:

| condition | what the agent gets |
|---|---|
| `C0_bare` | scene only (date, sitting, agenda item) |
| `C1_party` | scene + party |
| `C2_grounded` | scene + party + biography from the Sejm API + two of the MP's own earlier speeches + the real speeches before theirs |
| `C3_flipped` | C2, with every w/na Ukrainie in the preceding speeches flipped |
| `C4_swapped` | C2, with the party label swapped to the other camp |
| `D1_date_plus3y` | pre-invasion cases only: C2 with the date moved forward 3 years |

No MP names appear in the identity layer. The models run in bf16 with `transformers` in a separate Python process
(`src/score.py`), so nothing clashes with the packages Jupyter has already loaded; nothing is generated, so the run is fast.

### How to launch on Alpine (Open OnDemand)
1. https://ondemand.rc.colorado.edu → **Interactive Apps → Jupyter Session**
2. **Custom configuration**, Cluster `alpine`, Account `ucb757_asc1`, QoS `gpu-normal`, Time `4`, cores `8`, and a GPU slice with
   **at least ~35 GB**: Partition `artxpro6000` + gres `gpu:rtx_pro_6000_2g.48gb:1`, or Partition `aa100` + gres `gpu:a100-40gb:1`.
3. Open this notebook from the cloned repo. For a first test set `SMOKE = True` (one real "w" and one real "na" speech per party, first model only), press **Run All once**
   and wait. The first run installs packages (5 to 10 minutes) and downloads the models into `/scratch`; every step prints a line.
4. With `SMOKE = False`, **Run All** once (about 1 to 1.5 hours with the models already downloaded). Results are written line by
   line to `outputs/`, so a dropped session just resumes where it stopped.
5. Download `outputs/` (Files → select all → Download) and copy the run files into `results/` for `src/analyze_agents.py`.
""")

code(r"""
# (only once) clone the repo into /projects/$USER, then open notebooks/agents_alpine.ipynb from the file browser
# !cd /projects/$USER && git clone https://github.com/AntoniCzolgowski/sejm-ukraine-nlp4ca.git
# update later (keeps outputs/):
# !cd /projects/$USER/sejm-ukraine-nlp4ca && git fetch -q && git reset --hard origin/main
""")

md("## 1. Settings")
code(r"""
SMOKE = False                                  # True: a 3-minute test (one w and one na speech per party, first model only)
MODELS = ['bielik', 'pllum', 'mistral', 'qwen']  # Bielik 11B v2.3, PLLuM 12B, Mistral-Nemo 12B, Qwen3 14B (see src/score.py)
BATCH = 4                                      # prompts per forward pass; halves itself on GPU out-of-memory

import os, sys, json, time, subprocess, shutil
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / 'data' / 'cases.jsonl').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / 'data' / 'cases.jsonl').exists(), 'Open this notebook from inside the cloned repo'
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
USER = os.environ.get('USER', 'me')
LIB = os.environ.get('P2_LIB', f'/projects/{USER}/p2_pylibs')            # packages live here (home quota is only 2 GB)
SCRATCH = f'/scratch/alpine/{USER}' if os.path.isdir(f'/scratch/alpine/{USER}') else f'/projects/{USER}'
for k, v in {'HF_HOME': 'hf_cache', 'PIP_CACHE_DIR': 'pip_cache', 'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'xdg_cache',
             'TORCH_HOME': 'torch_home', 'TRITON_CACHE_DIR': 'triton_cache'}.items():
    os.environ[k] = f'{SCRATCH}/{v}'                                       # every cache goes to /scratch, never /home
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
if SMOKE:
    MODELS = MODELS[:1]
print('repo:', ROOT, '| smoke:', SMOKE, '| models:', MODELS, '| packages:', LIB, '| caches:', SCRATCH)
""")

md(r"""
## 2. Python packages (first run only, 5 to 10 minutes)
PyTorch and transformers go into `/projects` and are only used by the separate scoring process. The cell prints each step;
**keep it running and do not press Run All again** while it works. Pip may print an `ERROR: pip's dependency resolver ...`
line: that is only a note that these versions differ from Anaconda's, and it is fine.
""")
code(r"""
def run(cmd, env=None, keep=None):
    # run a command, stream its output into this cell, fail loudly
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env or os.environ.copy())
    for line in p.stdout:
        if keep is None or line.startswith(keep):
            print('   ', line.rstrip()[:200], flush=True)
    if p.wait() != 0:
        raise RuntimeError(f'command failed with code {p.returncode}: see the lines above')

MARK = Path(LIB) / '.install_complete'
if not MARK.exists():
    if Path(LIB).exists():
        print('removing an incomplete install from an earlier run:', LIB, flush=True)
        shutil.rmtree(LIB, ignore_errors=True)
    Path(LIB).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f'installing PyTorch and transformers into {LIB} (5 to 10 minutes) ...', flush=True)
    run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--progress-bar', 'off', '--no-warn-script-location',
         '--target', LIB, 'torch==2.7.1', 'transformers==4.55.4', 'accelerate==1.10.1', 'sentencepiece', 'protobuf',
         'safetensors', 'huggingface_hub', '--index-url', 'https://download.pytorch.org/whl/cu128',
         '--extra-index-url', 'https://pypi.org/simple'], keep=('Collecting torch', 'Successfully', 'ERROR', 'error'))
    MARK.write_text(time.strftime('%Y-%m-%d %H:%M'))
    print(f'install finished in {(time.time() - t0) / 60:.1f} min', flush=True)

ENV = os.environ.copy()
ENV['PYTHONPATH'] = LIB                     # the scoring process sees only our packages:
ENV['PYTHONNOUSERSITE'] = '1'               # python -S leaves Anaconda's site-packages (old numpy, scipy, sklearn) out entirely
run([sys.executable, '-S', '-c', 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__, '
     '"| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'], env=ENV)
""")

md("## 3. Cases and prompts (a look at one example)")
code(r"""
cases = [json.loads(l) for l in open(ROOT / 'data' / 'cases.jsonl')]
print(len(cases), 'cases in total |', sum(len(c['prompts']) for c in cases), 'prompts per model')
ex = [c for c in cases if c['set'] == 'post'][0]
print('\n--- example (C2_grounded), user message ---\n' + ex['prompts']['C2_grounded'][:1500] + ' ...')
print('\n--- the reply is pre-filled with the real speech up to the choice point ---\n... ' + ex['prefix'][-300:])
print('\nscored continuations:', ex['cont'], '| real choice:', ex['real'])
""")

md("## 4. Scoring and memorization check (separate process, prints progress)")
code(r"""
t0 = time.time()
cmd = [sys.executable, '-S', '-u', str(ROOT / 'src' / 'score.py'), '--models', ','.join(MODELS), '--batch', str(BATCH)]
if SMOKE:
    cmd.append('--smoke')
run(cmd, env=ENV)
print(f'scoring cell finished in {(time.time() - t0) / 60:.1f} min')
""")

md("## 5. Quick look (full analysis: src/analyze_agents.py)")
code(r"""
# the same summary score.py prints at the end; P(w) = P("w Ukrainie") / (P("w Ukrainie") + P("na Ukrainie"))
sys.path.insert(0, str(ROOT / 'src'))
import quicklook
quicklook.main(MODELS, SMOKE)
manifest = {'smoke': SMOKE, 'models': MODELS, 'finished': time.strftime('%Y-%m-%d %H:%M'), 'host': os.uname().nodename}
json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
print('\nDONE', manifest)
print('files to download:', sorted(p.name for p in OUT.iterdir()))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}}
Path(__file__).resolve().parents[1].joinpath("notebooks", "agents_alpine.ipynb").write_text(nbf.writes(nb))
print("notebook written")
