# MLC 2026 — opening Grand Prairie (Dallas) leg: model win probabilities

*Model `xgb_match_v3_m7_production` (raw probs, no Platt). PRE-XI / PRE-TOSS estimates — projected XIs from the announced 2026 squads (`mlc_2026_rosters.csv`), max 6 overseas + 5 USA-developed per XI; provisional. Win % only (no betting odds).*

| # | Date | Match | Venue | Win %  (team1 / team2) | Model pick | Conf. | top6 ELO Δ |
|---|------|-------|-------|------------------------|-----------|-------|-----------|
| 1 | 06-18 | **TSK** v ORCA | Grand Prairie, Dallas | 42.5% / 57.5% | **ORCA** (57.5%) | solid | -16 |
| 2 | 06-19 | **LAKR** v SFU | Grand Prairie, Dallas | 42.7% / 57.3% | **SFU** (57.3%) | solid | -3 |
| 3 | 06-19 | **ORCA** v WSH | Grand Prairie, Dallas | 49.2% / 50.8% | **WSH** (50.8%) | lean | +7 |
| 4 | 06-20 | **TSK** v SFU | Grand Prairie, Dallas | 50.6% / 49.4% | **TSK** (50.6%) | lean | -2 |
| 5 | 06-20 | **WSH** v MINY | Grand Prairie, Dallas | 54.6% / 45.4% | **WSH** (54.6%) | lean | +0 |
| 6 | 06-21 | **ORCA** v LAKR | Grand Prairie, Dallas | 63.0% / 37.0% | **ORCA** (63.0%) | solid | +16 |
| 7 | 06-21 | **TSK** v MINY | Grand Prairie, Dallas | 49.4% / 50.6% | **MINY** (50.6%) | lean | -9 |

team1 = listed-first / nominal home side. "Conf.": lean <55%, solid 55–65%, strong >65%. top6 ELO Δ = team1 − team2 top-6 batting ELO (the model's dominant feature; + favors team1).

## Notes
- Same production pipeline used for IPL, applied unchanged; the model carries real MLC team ELOs (75 cricsheet matches, 2023–2025) plus per-player career/ELO features. New overseas signings (Smith, Narine, Russell, Hales, Rachin, Ferguson, Ngidi, Shanaka, …) bring rich international-T20 ELOs.
- XIs respect the **MLC 6-overseas cap**, which binds hard on the stacked squads (LAKR has 8 internationals, WSH 9) — only 6 of each can play.
- Re-run any fixture once real XIs/toss are known: edit the lineup arrays in `fixtures/mlc_2026/<file>.json` and rerun this script.
