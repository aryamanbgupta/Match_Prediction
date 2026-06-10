# MLC 2025 — honest baseline check

Teacher-forced over 7736 actual deliveries. Two questions: is the top-scorer call better than a sensible no-model baseline, and is the strike-rate correlation anything more than career reputation?

## Q1 — Top scorer: sim vs baselines a human would use

66 team-innings. 'Hit' = the pick was the actual top scorer. The earlier '9% base rate' was a strawman (only the top order can realistically top-score). Fair comparison:

| picker | hit rate | note |
|---|---:|---|
| random among XI's batters | 14% | the strawman I used before |
| always the opener (pos 1) | 21% | zero-model, a-priori |
| always best position in hindsight (pos 2) | 27% | zero-model, optimistic |
| best top-order bat by career SR | 14% | reputation lookup, no sim |
| **our Monte-Carlo sim (#1 pick)** | **27%** | the model (18/66) |

(Note: a *teacher-forced* 'pick highest predicted total over the actual balls faced' scores ~80% — but that's a leak: it conditions on who actually batted longest, which you don't know pre-match. The honest model number is the sim's 27%.)

Top scorer by actual batting position:

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| share | 21% | 27% | 18% | 12% | 6% | 11% | 2% | 3% |

**Verdict:** the sim (27%) is level with just always backing a top-order position (27%). No demonstrable edge on top scorer.

## Q2 — Strike-rate correlation: skill or career lookup?

73 batters with ≥15 balls. Spearman rank-corr with ACTUAL strike rate:

| predictor | rank-corr vs actual SR |
|---|---:|
| **career reputation only** (EB career outcome dist) | **+0.51** |
| full ball model (teacher-forced) | +0.52 |

- Model vs career-proxy agreement: +0.59 (they rank batters almost identically).

- **Does the model beat the lookup?** After removing career reputation, the model's *residual* correlation with actual SR is **+0.28**. And even that is an optimistic ceiling: the model is teacher-forced on the realised innings, so a batter who actually survived into the high-strike-rate death overs gets fed those states — partly baking the outcome into the 'prediction'. A clean pre-match test would be lower. So the +0.5 headline is almost entirely 'good batters bat well', which you already know.

Per-batter detail (top 18 by balls) — career vs model vs actual SR:

| batter | balls | career SR | model SR | actual SR |
|---|---:|---:|---:|---:|
| MD Patel | 354 | 135 | 195 | 145 |
| F du Plessis | 285 | 138 | 225 | 168 |
| N Pooran | 279 | 150 | 218 | 134 |
| Q de Kock | 273 | 139 | 220 | 146 |
| UBT Chand | 259 | 119 | 174 | 126 |
| MW Short | 223 | 148 | 244 | 164 |
| KA Pollard | 195 | 150 | 200 | 172 |
| ADS Fletcher | 189 | 126 | 229 | 158 |
| MG Bracewell | 177 | 141 | 188 | 136 |
| Shubham Ranjane | 174 | 126 | 178 | 163 |
| J Fraser-McGurk | 172 | 150 | 207 | 169 |
| SO Hetmyer | 172 | 142 | 198 | 205 |
| SP Krishnamurthi | 169 | 138 | 169 | 156 |
| KR Mayers | 168 | 138 | 191 | 139 |
| MJ Owen | 165 | 186 | 135 | 195 |
| AGS Gous | 163 | 137 | 205 | 139 |
| R Ravindra | 160 | 136 | 184 | 169 |
| GJ Maxwell | 156 | 156 | 211 | 172 |

## Bottom line

- **Top scorer:** sim 27% ≈ best positional rule 27%. No edge over 'back a top-order bat'.
- **Strike rate:** the +0.52 correlation is the same as career reputation alone (+0.51); residual over career is +0.28 and even that is inflated by teacher-forcing. The model is largely re-deriving career stats.

