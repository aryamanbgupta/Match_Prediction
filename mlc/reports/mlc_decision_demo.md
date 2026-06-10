# MLC decision-delta demo — Texas Super Kings vs MI New York (2025-06-29)

*Ball-by-ball sim run on the real XIs. (A) shows the simulator lands on reality; (B) shows it ranking a real lineup decision. Same model both ways, so the bias cancels in the delta.*

## (A) Trust panel — projected vs actual

| team | projected total (mean, P10–P90) | actual |
|---|---|---:|
| Texas Super Kings | 195  (134–250) | 223 |
| MI New York | 169  (112–219) | 184 |

**Top-scorer board** (sim's pre-match read vs who actually top-scored):

| team | sim's most-likely top scorer | P(top) | actual top scorer |
|---|---|---:|---|
| Texas Super Kings | F du Plessis | 0.41 | F du Plessis |
| MI New York | N Pooran | 0.48 | KA Pollard |

## (B) Decision delta — promote MP Stoinis to No.3 (Texas Super Kings)

The sim rates **MP Stoinis** (batting 4, proj. SR 190) well above the incumbent No.3 **SK Patel** (proj. SR 165). Swap them and re-sim:

| Texas Super Kings order | projected total (mean, P10–P90) |
|---|---|
| actual order | 194.8  (134–250) |
| MP Stoinis promoted to 3 | 197.8  (132–254) |

**Projected swing: +3.0 runs** (95% CI [-2.7, +8.5], 500 sims/scenario).

The point isn't the exact number — it's that the model gives a *signed, sized, CI'd* answer to a concrete selection question, which a coach can weigh against match-ups, fitness, and gut. Repeat for any order, any bowler-phasing, any XI.

