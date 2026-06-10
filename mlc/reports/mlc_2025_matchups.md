# MLC 2025 — ball-level model: batter-vs-bowler matchup replay

*Every actual MLC 2025 delivery replayed through the v7 sim ball model (`models/xgb_v3`) with its real pre-ball state; predicted outcome distribution vs actual, aggregated by player / duel. Off-bat runs (extras excluded). June balls are in the model's validation split (early-stop only), July fully held out.*

- **Deliveries**: 7736 (June/val 5095, July/test 2641).

## Headline: the ball model runs hot on tail outcomes

Mean predicted probability vs actual frequency, per outcome class (all MLC balls):

| outcome | pred prob | actual freq | pred/actual |
|---|---:|---:|---:|
| dot | 0.225 | 0.279 | 0.81× |
| one | 0.269 | 0.415 | 0.65× |
| two | 0.083 | 0.065 | 1.28× |
| four | 0.174 | 0.109 | 1.59× |
| six | 0.126 | 0.079 | 1.60× |
| wicket | 0.123 | 0.053 | 2.34× |

The model over-states boundaries and wickets and under-states dots/ones — the same tail-inflation seen in the prop backtest. So absolute predicted economy/wicket numbers below run hot; read them as **relative** rankings, not point estimates.

## Can it rank who dominates? (Spearman rank-correlation, predicted vs actual)

| level | n | metric | rank corr |
|---|---:|---|---:|
| per bowler (≥18 balls) | 65 | economy | +0.22 |
| per batter (≥15 balls) | 73 | strike rate | +0.52 |
| per duel (≥9 balls)    | 264 | economy | +0.11 |

Duel-level directional agreement (model's batter-vs-bowler-favoured call matches the actual side of the mean): **53%** of 264 duels.

Per-duel actuals rest on only ~9–12 balls, so duel-level signal is noise-dominated; the per-bowler / per-batter levels (more balls) are where ranking is testable.

## Per-bowler: predicted vs actual economy (≥18 balls)

| bowler | balls | pred econ | act econ | pred wkts | act wkts |
|---|---:|---:|---:|---:|---:|
| LH Ferguson | 50 | 9.5 | 3.7 | 4.4 | 3 |
| AJ Hosein | 121 | 10.5 | 4.8 | 13.8 | 10 |
| Tajinder Singh | 49 | 9.6 | 5.3 | 2.9 | 0 |
| MA Aponso | 25 | 13.5 | 5.5 | 2.6 | 1 |
| BC Fortuin | 26 | 11.0 | 6.2 | 2.6 | 0 |
| AF Milne | 134 | 11.1 | 6.7 | 22.3 | 15 |
| SP Narine | 149 | 8.8 | 6.9 | 16.4 | 3 |
| Harmeet Singh | 189 | 9.9 | 7.0 | 18.0 | 9 |
| MW Short | 112 | 12.3 | 7.1 | 13.5 | 6 |
| GF Linde | 67 | 11.4 | 7.2 | 6.9 | 3 |
| IG Holland | 155 | 11.0 | 7.2 | 18.5 | 11 |
| K Gore | 44 | 9.0 | 7.2 | 3.3 | 4 |
| J Drysdale | 20 | 9.2 | 7.5 | 2.2 | 0 |
| R Shepherd | 110 | 12.9 | 7.6 | 15.2 | 8 |
| Noor Ahmad | 266 | 9.5 | 7.8 | 44.5 | 15 |
| TA Boult | 298 | 11.0 | 7.9 | 38.4 | 16 |
| N Burger | 135 | 11.9 | 8.0 | 16.6 | 8 |
| KS Gattepalli | 24 | 8.9 | 8.0 | 1.0 | 3 |
| Zia-ul-Haq | 170 | 10.9 | 8.0 | 18.5 | 6 |
| Waqar Salamkheil | 94 | 10.3 | 8.1 | 10.5 | 6 |
| SN Netravalkar | 251 | 11.1 | 8.2 | 35.8 | 10 |
| GJ Maxwell | 136 | 11.4 | 8.2 | 14.0 | 10 |
| SV Wiig | 18 | 9.0 | 8.3 | 1.3 | 2 |
| R Ugarkar | 168 | 13.4 | 8.4 | 21.6 | 11 |
| MP Stoinis | 171 | 13.0 | 8.5 | 29.6 | 10 |
| JO Holder | 204 | 11.7 | 8.5 | 26.0 | 10 |
| NP Kenjige | 144 | 11.3 | 8.5 | 14.3 | 7 |
| XC Bartlett | 250 | 12.5 | 8.6 | 30.3 | 19 |
| T Luus | 102 | 10.9 | 8.6 | 9.0 | 6 |
| J Edwards | 215 | 11.8 | 8.7 | 23.7 | 15 |
| Saif Badar | 28 | 11.8 | 8.8 | 2.1 | 0 |
| MJ Owen | 187 | 11.0 | 8.9 | 18.8 | 16 |
| T Sangha | 136 | 10.2 | 8.9 | 14.9 | 9 |
| Haris Rauf | 173 | 11.1 | 8.9 | 24.2 | 18 |
| BL Couch | 194 | 12.0 | 9.1 | 23.8 | 3 |
| Sunny Patel | 135 | 11.3 | 9.1 | 18.5 | 5 |
| KA Pollard | 149 | 11.4 | 9.1 | 21.4 | 7 |
| Hassan Khan | 208 | 10.2 | 9.1 | 28.4 | 13 |
| AD Russell | 212 | 12.6 | 9.3 | 31.1 | 10 |
| KR Mayers | 153 | 13.0 | 9.3 | 19.1 | 4 |
| G Coetzee | 116 | 12.5 | 9.3 | 17.1 | 5 |
| Ali Khan | 163 | 12.7 | 9.3 | 17.3 | 5 |
| Naveen-ul-Haq | 120 | 11.7 | 9.7 | 15.4 | 8 |
| Jasdeep Singh | 174 | 12.3 | 9.8 | 20.1 | 6 |
| LE Plunkett | 83 | 11.3 | 9.8 | 16.0 | 3 |
| Sikandar Raza | 118 | 10.5 | 9.8 | 15.0 | 5 |
| R Ravindra | 80 | 11.1 | 9.8 | 9.4 | 4 |
| SC van Schalkwyk | 223 | 11.6 | 9.8 | 31.0 | 14 |
| AA Paradkar | 20 | 14.3 | 9.9 | 1.5 | 1 |
| MG Bracewell | 143 | 10.7 | 9.9 | 13.3 | 5 |
| OC McCoy | 95 | 11.4 | 9.9 | 10.1 | 3 |
| MR Adair | 45 | 11.6 | 10.0 | 3.6 | 1 |
| C le Roux | 75 | 10.2 | 10.1 | 8.2 | 4 |
| BG Lister | 36 | 12.1 | 10.2 | 5.2 | 0 |
| Ehsan Adil | 158 | 11.3 | 10.4 | 19.3 | 6 |
| CJ Gannon | 127 | 12.6 | 10.5 | 15.5 | 4 |
| DC Drakes | 25 | 10.7 | 10.6 | 1.9 | 2 |
| Mohammad Mohsin | 55 | 10.5 | 10.7 | 11.6 | 2 |
| D Ferreira | 67 | 10.9 | 10.7 | 8.0 | 1 |
| CA Dry | 91 | 11.4 | 10.7 | 8.3 | 5 |
| BV Sears | 27 | 11.9 | 11.1 | 2.9 | 0 |
| DJ Mitchell | 61 | 6.5 | 11.4 | 8.0 | 4 |
| A Desai | 45 | 15.3 | 11.7 | 4.8 | 3 |
| D Potgieter | 25 | 12.3 | 12.2 | 1.3 | 2 |
| GD Phillips | 24 | 9.7 | 13.0 | 1.7 | 2 |

## Per-batter: predicted vs actual strike rate (≥15 balls, top 20 by runs)

| batter | balls | act runs | pred SR | act SR |
|---|---:|---:|---:|---:|
| MD Patel | 354 | 512 | 195 | 145 |
| F du Plessis | 285 | 479 | 225 | 168 |
| Q de Kock | 273 | 398 | 220 | 146 |
| N Pooran | 279 | 374 | 218 | 134 |
| MW Short | 223 | 366 | 244 | 164 |
| SO Hetmyer | 172 | 353 | 198 | 205 |
| FH Allen | 154 | 344 | 200 | 223 |
| KA Pollard | 195 | 335 | 200 | 172 |
| UBT Chand | 259 | 327 | 174 | 126 |
| MJ Owen | 165 | 321 | 135 | 195 |
| ADS Fletcher | 189 | 298 | 229 | 158 |
| J Fraser-McGurk | 172 | 291 | 207 | 169 |
| Shubham Ranjane | 174 | 283 | 178 | 163 |
| R Ravindra | 160 | 271 | 184 | 169 |
| GJ Maxwell | 156 | 268 | 211 | 172 |
| SP Krishnamurthi | 169 | 264 | 169 | 156 |
| D Ferreira | 129 | 263 | 242 | 204 |
| MG Bracewell | 177 | 241 | 188 | 136 |
| Hassan Khan | 124 | 237 | 173 | 191 |
| KR Mayers | 168 | 233 | 191 | 139 |

## Illustrative duels the model called right (≥9 balls; standardized agreement)

**Bowler kept the batter quiet, as the model expected** (both predicted & actual economy in the low tail of their distributions):

| bowler | batter | balls | pred econ | act econ | act wkts |
|---|---|---:|---:|---:|---:|
| Sikandar Raza | MG Bracewell | 10 | 7.3 | 1.8 | 0 |
| Noor Ahmad | Jasdeep Singh | 11 | 6.9 | 2.7 | 0 |
| MP Stoinis | Aaron Jones | 10 | 8.7 | 1.8 | 1 |
| SN Netravalkar | Saif Badar | 10 | 9.4 | 1.2 | 0 |
| N Burger | Jasdeep Singh | 11 | 8.0 | 3.3 | 1 |
| TA Boult | SK Patel | 13 | 6.6 | 5.5 | 0 |
| Hassan Khan | Sikandar Raza | 9 | 9.0 | 2.7 | 0 |
| Harmeet Singh | DJ Mitchell | 9 | 7.7 | 4.7 | 1 |
| LH Ferguson | H Klaasen | 14 | 8.8 | 3.4 | 0 |
| MW Short | Sujit Nayak | 10 | 8.8 | 3.6 | 1 |

**Batter took the bowler down, as the model expected** (both in the high tail):

| bowler | batter | balls | pred econ | act econ |
|---|---|---:|---:|---:|
| OC McCoy | R Shepherd | 9 | 16.6 | 20.7 |
| R Ugarkar | D Ferreira | 14 | 19.9 | 13.3 |
| BG Lister | AD Hales | 10 | 14.3 | 20.4 |
| Jasdeep Singh | R Powell | 11 | 17.4 | 15.8 |
| J Edwards | F du Plessis | 11 | 15.6 | 17.5 |
| IG Holland | FH Allen | 9 | 14.6 | 18.7 |
| Sunny Patel | J Fraser-McGurk | 10 | 17.4 | 14.4 |
| AD Russell | D Ferreira | 14 | 16.1 | 14.6 |
| XC Bartlett | SO Hetmyer | 9 | 14.7 | 16.0 |
| Ali Khan | D Ferreira | 15 | 17.2 | 12.4 |
