# W or na Ukrainie? LLM agent MPs and a convention that split the Polish Sejm

Feasibility test for proposal 2 in *NLP for Cultural Analytics* (CSCI 7000, CU Boulder, Fall 2026).

After Russia's full-scale invasion of Ukraine (24 Feb 2022), part of Polish public life switched from the traditional
**"na Ukrainie"** to **"w Ukrainie"**, the form used for sovereign states. In the Sejm the switch was partial and political:
"w" went from 1% of 211 uses before the invasion to 31% of 3,857 uses after it, from 11% in PiS (2023-25) to 79% in Lewica.
Other countries that take "na" (Lithuania, Belarus, Slovakia, Hungary, Latvia) barely moved, so the change is about Ukraine.

The question: when LLM agents speak as these MPs, do they make the same choice, for the same people, at the same time?

## Data
- **Human side:** every use of "w/na" + Ukraine, Lithuania, Belarus, Slovakia, Hungary and Latvia (locative) in Sejm plenary
  sittings and committee meetings, Nov 2019 to Apr 2025, from the **Polish Parliamentary Corpus** (Ogrodniczuk 2018;
  https://clip.ipipan.waw.pl/PPC; parliamentary texts are public domain, corpus annotations CC BY). Speaker club at the time of
  the sitting comes from the corpus; birth year, place, district, education and profession from the Sejm API (api.sejm.gov.pl).
  `data/sejm_matches.jsonl` (6,431 matches, 4,068 for Ukraine), built by `src/extract_sejm.py`.
- **Agent side:** `data/cases.jsonl`, 893 real choice points (806 after the invasion, stratified by party and government,
  87 before), each with prompts for five conditions, built by `src/build_cases.py`. `data/memo_excerpts.jsonl` holds 240
  speech excerpts for the memorization check.

## Run (CU Alpine, Open OnDemand, no terminal)
Open `notebooks/agents_alpine.ipynb` in a **Jupyter Session** on a GPU slice with at least ~35 GB (instructions at the top).
Four open models in bf16 (Bielik 11B v2.3, PLLuM 12B, Mistral-Nemo 12B, Qwen3 14B) read each prompt and we compare the
probability of the two full continuations, " w Ukrainie" vs " na Ukrainie", written as they would sit in the transcript
(the next word alone would also count "w Polsce" or "na froncie"; that number is kept as a secondary column). The prompt's
KV cache is reused for the two continuations, so this costs about one forward pass per prompt. Every result is checkpointed
in `outputs/probe_<model>.jsonl` and `outputs/memo_<model>.jsonl`; `src/quicklook.py` prints a summary.

## Layout
```
notebooks/agents_alpine.ipynb   agent probe on Alpine (scores + memorization check)
src/extract_sejm.py             PPC TEI dumps -> data/sejm_matches.jsonl (+ debate turns for the prompts)
src/build_cases.py              choice points -> prompts for C0-C4 and D1
src/score.py                    agent probe on the GPU (called by the notebook)
src/quicklook.py                summary of the probe outputs (standard library only)
src/analyze_agents.py           agent vs human analysis (bootstrap CIs, baselines, experiments) -> outputs/agent_*
results/agent_run_2026-10-13/   raw agent scores and memorization check from the Alpine run
src/figure.py                   Figure 1 of the proposal
src/slope_ci.py                 bootstrap CI for the club-gradient slope (H1)
proposal/                       submitted proposal (LaTeX source, PDF, figure)
src/analyze_human.py            human-side tables, previous-speaker model, half-year series -> outputs/
tools/build_notebook.py         regenerates the notebook
```
