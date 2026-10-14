# Agent MPs vs the real Sejm: results (run of 13 Oct 2026)

800 choice points after the invasion, 87 before; 95% CIs from 1000 speaker bootstrap resamples.

## Telling real "w" from real "na" speeches after the invasion (AUC, 0.5 = chance)

| model | C0 bare | C1 party | C2 grounded | C2 within party | C4 swapped | next-word C2 |
|---|---|---|---|---|---|---|
| Bielik 11B v2.3 | 0.64 [0.59, 0.69] | 0.69 [0.64, 0.73] | 0.70 [0.65, 0.74] | 0.64 [0.57, 0.70] | 0.59 [0.54, 0.64] | 0.60 |
| PLLuM 12B | 0.61 [0.56, 0.65] | 0.65 [0.60, 0.69] | 0.64 [0.60, 0.69] | 0.58 [0.51, 0.65] | 0.59 [0.55, 0.63] | 0.58 |
| Mistral-Nemo 12B | 0.59 [0.55, 0.63] | 0.60 [0.56, 0.64] | 0.58 [0.55, 0.62] | 0.57 [0.50, 0.64] | 0.57 [0.53, 0.61] | 0.55 |
| Qwen3 14B | 0.56 [0.51, 0.60] | 0.56 [0.52, 0.61] | 0.56 [0.51, 0.61] | 0.54 [0.48, 0.60] | 0.55 [0.50, 0.60] | 0.49 |
| baseline: party x government rate (human data, leave-one-out) (n=800) | | | 0.74 [0.68, 0.78] |  | | |
| baseline: the MP's own earlier uses (n=625) | | | 0.82 [0.77, 0.86] | 0.69 [0.60, 0.75] | | |

## Party gradient (C2, after the invasion): agent P(w) vs real share of "w" (all Sejm uses)

| party | human | Bielik 11B v2.3 | PLLuM 12B | Mistral-Nemo 12B | Qwen3 14B |
|---|---|---|---|---|---|
| Lewica | 0.65 | 0.36 | 0.42 | 0.45 | 0.47 |
| Polska2050 | 0.47 | 0.32 | 0.45 | 0.45 | 0.45 |
| KO | 0.49 | 0.34 | 0.40 | 0.39 | 0.36 |
| PSL | 0.21 | 0.25 | 0.32 | 0.40 | 0.40 |
| PiS | 0.15 | 0.20 | 0.26 | 0.40 | 0.39 |
| Kukiz15 | 0.07 | 0.24 | 0.11 | 0.36 | 0.28 |
| Konfederacja | 0.07 | 0.14 | 0.19 | 0.37 | 0.34 |
| Spearman rho / logit slope | | 0.96 / 0.31 | 0.86 / 0.49 | 0.71 / 0.10 | 0.75 / 0.17 |

## Experiments (log-odds shifts, 95% CI)

| model | context flip: OR toward flipped form (n) | party swap: share of real party gap | date +3y on pre-invasion speeches | Min-K% 2022-23 minus 2025 |
|---|---|---|---|---|
| Bielik 11B v2.3 | 4.25 [2.17, 7.14] (42) | 0.18 [0.16, 0.20] | 0.34 [0.21, 0.50] (P(w) 0.11 -> 0.14) | 0.04 [-0.13, 0.23] |
| PLLuM 12B | 8.19 [5.11, 13.96] (42) | 0.13 [0.12, 0.15] | 0.44 [0.31, 0.57] (P(w) 0.16 -> 0.19) | 0.22 [-0.02, 0.45] |
| Mistral-Nemo 12B | 19.97 [8.08, 48.47] (42) | 0.02 [0.02, 0.03] | 0.12 [0.05, 0.20] (P(w) 0.26 -> 0.28) | 0.01 [-0.16, 0.18] |
| Qwen3 14B | 7.46 [3.59, 14.35] (42) | 0.03 [0.03, 0.04] | -0.04 [-0.13, 0.07] (P(w) 0.41 -> 0.40) | -0.02 [-0.19, 0.17] |

Human previous-speaker effect (observational, all Sejm uses): OR 1.63 [1.26, 2.1], n=2456.
