# Can LLM Agent MPs Reproduce a Language Convention That Split the Parliament?

**A real-world test for agent simulations of social conventions.** Proposal 2 for *NLP for Cultural Analytics*
(CSCI 7000, CU Boulder, Fall 2026). The submitted proposal is in [`proposal/`](proposal/).

LLM agents are used as stand-ins for people, including whole agent populations in which researchers study how conventions
form and tip. Those claims hold only if agents choose like the people they stand for: who adopts a new form, when, and how
much each speaker follows the one before. This project checks that directly: place the agent at the exact moment a real
person made a choice, give it the context that person had, and compare its choice with theirs.

## The testbed
Polish says *w Polsce* ("in Poland") but traditionally *na Ukrainie* (literally "on Ukraine"), the preposition Polish uses
for regions and also for a few countries (*na Litwie*, *na Węgrzech*). Both forms mean the same place, so the choice is a
convention. After Russia's full-scale invasion on 24 February 2022, *w Ukrainie* came, for many speakers, to express respect
for Ukraine as a sovereign state, and in the Sejm the choice turned political.

## What the record shows (people)
- Before the invasion 3 of 211 uses (1.4%) were *w*; after it 31% of 3,857, and the share stopped growing
  (30.2% under the PiS government, 32.6% under the KO-led one).
- The control countries that also take *na* (Lithuania, Belarus, Slovakia, Hungary, Latvia) barely moved, so the change is
  about Ukraine.
- The convention split along party lines instead of tipping: under the KO-led government *w* was 79% in Lewica, 56% in KO,
  47% in Polska 2050, 22% in PSL, 11% in PiS and 2% in Konfederacja.
- MPs follow the room modestly: after a speaker who said *w*, the odds of *w* rise by OR 1.63 [1.26, 2.10].

## What the agents do (four open models, 4,199 prompts each)
- **They know the party order but flatten it.** Bielik ranks the clubs almost exactly as MPs do (Spearman 0.96) but keeps
  only about a third of the real distance between them (log-odds slope 0.31 [0.23, 0.48]); Mistral-Nemo and Qwen3 barely
  separate the clubs at all.
- **They copy the room far more than MPs do.** Swapping *w* and *na* in the preceding speeches multiplies the odds of *w*
  by 4 to 20, against 1.63 for real MPs.
- **They miss the individual.** No agent beats a lookup of the club's real rate (AUC 0.74); the MP's own earlier choices
  reach 0.82, the best agent 0.70.
- **No sign of memorization:** Min-K% does not differ between transcript excerpts from 2022-23 and 2025 in any model.

## Data
- **Polish Parliamentary Corpus** (Ogrodniczuk 2018; https://clip.ipipan.waw.pl/PPC): official transcripts of Sejm plenary
  sittings and committee meetings, 9th and 10th terms (5,940 Sejm documents, uses from Nov 2019 to Apr 2025). Official
  documents are not protected by Polish copyright law; the corpus annotations are CC BY. MP biographies from the Sejm API
  (api.sejm.gov.pl).
- `data/sejm_matches.jsonl`: 6,431 uses of *w/na* before six country names, 4,068 of them before *Ukrainie*
  (`src/extract_sejm.py`).
- `data/cases.jsonl`: 893 real choice points (806 after the invasion, at most 80 per club and government period, and all 87
  before it), each with its prompts for the conditions below (`src/build_cases.py`). Five cases where an MP quotes the phrase itself,
  and one with no words before the choice, are excluded from scoring.
- `data/memo_excerpts.jsonl`: 240 speech excerpts for the memorization check.

## Method
Each prompt has four layers, in Polish: the scene (date, sitting, agenda item), the identity (club, role, Sejm API
biography and two excerpts of the MP's own earlier speeches, never the MP's name), the last three real speeches before the
MP's turn, and the task. The model's reply is pre-filled with the MP's real words up to the choice point, and we compare
the probability of the two full continuations, " w Ukrainie" vs " na Ukrainie", written as in the transcript. Nothing is
generated. Conditions: C0 scene only, C1 + club, C2 full prompt, C3 forms flipped in the preceding speeches, C4 club swapped
to the other camp, D1 pre-invasion speeches with the date moved three years later.

Models (bf16, `transformers`): Bielik 11B v2.3, PLLuM 12B, Mistral-Nemo 12B, Qwen3 14B. The full run took 56 minutes on one
48 GB GPU slice of CU's Alpine cluster.

## Run (CU Alpine, Open OnDemand, no terminal)
Open `notebooks/agents_alpine.ipynb` in a Jupyter Session on a GPU slice with at least ~35 GB (instructions at the top),
set `SMOKE = True` for a short test or `False` for the full run, and press Run All once. Results are checkpointed line by
line in `outputs/probe_<model>.jsonl` and `outputs/memo_<model>.jsonl`, so a stopped run resumes. Then
`python src/analyze_agents.py` and `python src/slope_ci.py` write `outputs/agent_results.json` and `outputs/agent_tables.md`,
and `python src/figure.py` draws Figure 1.

## Layout
```
proposal/                       submitted proposal (LaTeX source, PDF, figure)
notebooks/agents_alpine.ipynb   agent probe on Alpine (scores + memorization check)
src/extract_sejm.py             PPC TEI dumps -> data/sejm_matches.jsonl (+ debate turns for the prompts)
src/analyze_human.py            human-side tables, previous-speaker model, half-year series -> outputs/
src/build_cases.py              choice points -> prompts for C0-C4 and D1
src/score.py                    agent probe on the GPU (called by the notebook)
src/quicklook.py                summary of the probe outputs (standard library only)
src/analyze_agents.py           agent vs human analysis (bootstrap CIs, baselines, experiments) -> outputs/agent_*
src/slope_ci.py                 bootstrap CI for the club-gradient slope (H1)
src/figure.py                   Figure 1 of the proposal
results/agent_run_2026-10-13/   raw agent scores and memorization check from the Alpine run
tools/build_notebook.py         regenerates the notebook
```
