# Data Refresh & Test Set Expansion — Handoff Document

**Created**: April 2026
**Purpose**: Expand our evaluation test set from 44 matches to 500+ by combining refreshed Cricsheet ball-by-ball data with Polymarket pre-match odds.

---

## Current State

- **Cricsheet data**: 8,341 T20 match JSONs in `data/t20s_json/`, latest match date 2025-06-15
- **Polymarket odds**: 1,161 resolved markets with pre-match prices in `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json`
- **Current test set**: 44 T20 World Cup 2024 matches with hand-curated odds in `betting_odds_v3.json`

After this work, we should have **500+ T20 matches** with both ball-by-ball data AND pre-match market odds.

---

## Step 1: Refresh Cricsheet Data

### What

Download the latest T20 JSON archive from Cricsheet and merge new matches into `data/t20s_json/`.

### How

1. Download the all-matches T20 JSON zip from Cricsheet:
   - URL: `https://cricsheet.org/downloads/all_json.zip` (all formats) or check for T20-specific archives
   - Cricsheet updates regularly; the archive should contain matches through ~April 2026
2. Extract and compare against existing files in `data/t20s_json/`
3. Copy new files (those not already present) into `data/t20s_json/`
4. Do **not** delete or overwrite existing files — only add new ones

### Expected Result

- ~400-600 new match files (covering July 2025 – April 2026)
- Total should grow from 8,341 to ~8,800-9,000 files

### Verification

```bash
# Before
ls data/t20s_json/ | wc -l  # Should show 8341

# After refresh
ls data/t20s_json/ | wc -l  # Should show ~8800-9000

# Check latest match date
uv run python -c "
import json, glob, os
files = glob.glob('data/t20s_json/*.json')
dates = []
for f in sorted(files)[-20:]:
    with open(f) as fh:
        d = json.load(fh)
    dates.append(d.get('info',{}).get('dates',[''])[0])
print('Latest dates:', sorted(dates, reverse=True)[:5])
"
# Should show dates in March/April 2026
```

---

## Step 2: Build the Matching & Conversion Script

### What

Create a script that:
1. Reads Polymarket pre-match odds from `polymarket_prematch_odds.json`
2. Filters to senior men's T20 matches only
3. Matches each market to a Cricsheet JSON file using team names + date
4. Outputs a `betting_odds_polymarket.json` in CricML's existing format

### Filtering Criteria (Senior Men's T20 Only)

**Include** these leagues/tournaments:

| League | Slug Pattern | Est. Matches |
|--------|-------------|-------------|
| T20I bilateral & World Cups | `crint-*` | ~303 |
| BBL (Big Bash) | `craus-*` | ~90 |
| IPL | `ipl-*`, `cricipl*` | ~75 |
| BPL (Bangladesh PL) | `crban-*` | ~54 |
| SA20 | `crsou-*`, `*sa20*` | ~53 |
| SMAT (Indian domestic) | `crind-*` | ~49 |
| T20 WC Qualifier | `crwt20wc*` | ~42 |
| ILT20 (UAE league) | `cruae-*` | ~33 |
| PSL (Pakistan SL) | `cricpsl*` | ~22 |
| Asia Cup (T20 format) | `*asia-cup*` | ~17 |
| T20 World Cup 2024 | `*t20-world-cup*` | ~14 |
| CSA T20 Challenge | `criccsat20*` | ~6 |
| England bilateral T20I | `cricket-en*`, `*t20i*` | ~5 |
| Super Smash (NZ) | `cricss*`, `crnzl-*`, `crnew-*` | ~3 |
| Other bilateral T20I | `crafg*`, `intt20-*`, `t20-*` | ~6 |
| Early IPL (verbose slugs) | `*-vs-*` with IPL teams | ~20 |
| TBCL / Minor leagues | `crictbcl*` | ~8 |

**Exclude**:

| Category | Slug Pattern | Reason |
|----------|-------------|--------|
| Women's cricket | `*wpl*`, `crwncl*`, `criwwc*`, title contains "women" | Different player pool, our model excludes women from ELO |
| U19 | `*u19*`, title contains "u19" or "under-19" | Junior cricket |
| ODI format | `intodi-*`, `odi-*`, title contains "ODI" | Wrong format (our model is T20 only) |
| Test format | title contains "Test" | Wrong format |
| ICC Champions Trophy | `cricket-ic*`, `*champions-trophy*` | ODI tournament |
| Plunket Shield (NZ) | `cricps-*` | First-class, not T20 |

**Additional filters** (apply after league filter):
- Must have a `winner` (skip no-results, ties, DLS)
- Must NOT be `low_liquidity` (50/50 prices indicate no real trading)
- Volume > $100 (skip markets with negligible trading)

### Team Name Mapping

Polymarket uses abbreviated or franchise-only names. Cricsheet uses full official names. A mapping dictionary is needed. Known mismatches:

| Polymarket Name | Cricsheet Name |
|----------------|---------------|
| `USA` | `United States of America` |
| `UAE` | `United Arab Emirates` |
| `New Guinea` | `Papua New Guinea` |
| `Lanka` | `Sri Lanka` |
| `Kong` | `Hong Kong` |
| `Hong Kong, China` | `Hong Kong` |
| `Hyderabad` | `Sunrisers Hyderabad` |
| `Chennai` | `Chennai Super Kings` |
| `Mumbai` | `Mumbai Indians` |
| `Kolkata` | `Kolkata Knight Riders` |
| `Delhi` | `Delhi Capitals` |
| `Lucknow` | `Lucknow Super Giants` |
| `Gujarat` | `Gujarat Titans` |
| `Punjab` | `Punjab Kings` |
| `Rajasthan` | `Rajasthan Royals` |
| `Bangalore` | `Royal Challengers Bangalore` / `Royal Challengers Bengaluru` |
| `Emirates` | `MI Emirates` |

**Whitespace issues** (12 entries in Polymarket data have trailing/leading spaces):
- `"Sri Lanka "` → `"Sri Lanka"`
- `" New Zealand"` → `"New Zealand"`
- `"New Guinea "` → `"Papua New Guinea"`
- `"Lucknow "` → `"Lucknow Super Giants"`
- `"Kolkata "` → `"Kolkata Knight Riders"`

Always `.strip()` team names before mapping.

**Teams that may not have Cricsheet coverage** (and should be dropped if no match found):
- Minor domestic teams (Indian states: Andhra, Assam, Bihar, etc.)
- Associate A teams (Bangladesh A, India A, Pakistan A, etc.)
- Minor league franchises (Genid Royals, Iconic Super Knights, etc.)

### Match Matching Logic

Match a Polymarket market to a Cricsheet file using:

1. **Date**: Polymarket `date` field == Cricsheet `info.dates[0]`
2. **Teams**: Both Polymarket teams (after name mapping + strip) must appear in Cricsheet `info.teams[]`
3. **Format**: Cricsheet `info.match_type` must be `"T20"` (skip ODI/Test files)
4. **Gender**: Cricsheet `info.gender` must be `"male"`

If multiple Cricsheet files match the same date + teams (rare but possible with rescheduled games), prefer the one whose `info.event.name` matches the tournament.

### Output Format

Generate `betting_odds_polymarket.json` matching our existing format:

```json
{
  "source": "polymarket",
  "generated_at": "2026-04-18T...",
  "total_matches": 523,
  "matches": [
    {
      "match_id": "2024-06-12_United_States_of_America_India_Nassau_County_International_Cricket_Stadium,_New_York",
      "date": "2024-06-12",
      "team1": "United States of America",
      "team2": "India",
      "venue": "Nassau County International Cricket Stadium, New York",
      "actual_winner": "United States of America",
      "polymarket_event_slug": "usa-vs-india-cricket-t20-world-cup",
      "polymarket_volume_usd": 14865.85,
      "tournament": "T20 World Cup 2024",
      "odds": {
        "winner": {
          "United States of America": 11.11,
          "India": 1.10,
          "timestamp": "2024-06-12T11:00:04Z"
        }
      }
    }
  ]
}
```

**Key format details**:
- `match_id`: Format is `{date}_{team1}_{team2}_{venue}` with spaces replaced by underscores. Team names and venue come from **Cricsheet** (not Polymarket).
- `team1`, `team2`: Use Cricsheet names (full official names).
- `actual_winner`: Use Cricsheet `info.outcome.winner` (authoritative), NOT Polymarket's resolved price. Cross-check: they should agree.
- `venue`: From Cricsheet `info.venue`.
- `odds.winner`: Decimal odds, computed as `1.0 / prematch_price`. E.g., price 0.09 → odds 11.11, price 0.91 → odds 1.10.
- `timestamp`: From Polymarket `price_timestamp`.
- Include `polymarket_event_slug` and `polymarket_volume_usd` as metadata for downstream filtering.

### Handling the Existing Test Set

The existing `betting_odds_v3.json` has 44 T20 World Cup 2024 matches with odds from a traditional bookmaker. The Polymarket data has ~14 of those same matches (Polymarket didn't cover all WC 2024 games).

**Recommendation**: Keep `betting_odds_v3.json` as-is for continuity with past experiments. The new `betting_odds_polymarket.json` is a separate, larger test set. The evaluation script (`run_sim_eval.py`) already takes `--odds` as a flag, so we can point it at either file.

---

## Step 3: Re-run Parsing Pipeline

After adding new Cricsheet data, the model's stats cache and training data need updating.

```bash
# Re-run feature engineering to incorporate new matches into stats cache
uv run python scripts/parsing_v2.py

# Re-train model on expanded data
uv run python scripts/xgboost_v2.py
```

**Important**: The new matches go into the training data (the parser splits by date). Only the matches in `betting_odds_polymarket.json` are used for evaluation — they are NOT part of the training split because they're loaded separately by `run_sim_eval.py` from the test directory.

However, we need to ensure the evaluation matches are **not** in the training split. The parser uses a date-based cutoff. Verify that the test matches' dates fall after the training cutoff.

---

## Step 4: Run Evaluation on Expanded Test Set

```bash
# Copy matched Cricsheet files to a new test directory
mkdir -p data/polymarket_test/
# (the matching script should copy matched files here)

# Run evaluation
uv run python scripts/sim_eval/run_sim_eval.py \
    --test-dir data/polymarket_test \
    --odds betting_odds_polymarket.json \
    --n-sims 100
```

---

## Data Quality Notes

1. **Volume threshold**: Median volume is ~$39K across included matches. IPL 2026 matches have $1-3M volume (excellent liquidity). BBL/BPL matches have $8-80K. Consider using a higher threshold (e.g., $1000) for the final evaluation to ensure odds quality.

2. **Polymarket vs traditional bookmakers**: Polymarket odds come from a prediction market, not a bookmaker. They have no built-in margin (vig), so `price_team1 + price_team2 = 1.0` by construction. Our existing odds from `betting_odds_v3.json` have bookmaker margin that gets removed in `BettingOddsLoader`. The Polymarket odds are already margin-free.

3. **Price granularity**: Polymarket prices are hourly snapshots (fidelity=60). The price timestamp is the last hourly tick before match start. E.g., if a match starts at 14:30 UTC, the price is from 14:00 UTC. This is close enough to "pre-match" for our purposes.

4. **119 unclassified markets**: There are ~119 markets with slug patterns not covered by the league detection logic above (early IPL matches with verbose slugs, miscellaneous bilateral series). These should be manually reviewed — most are includable T20 matches. The slug patterns are listed in the analysis above.

---

## File Locations

| File | Location | Description |
|------|----------|-------------|
| Polymarket pre-match odds | `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_prematch_odds.json` | 1,161 markets with pre-match prices |
| Polymarket match events | `/Users/aryamangupta/Projects/polymarket-cricket/data/polymarket_match_events.json` | Raw market metadata (1,468 markets) |
| Cricsheet T20 data | `/Users/aryamangupta/CricML/Match_Prediction/data/t20s_json/` | 8,341 match JSONs |
| Current odds file | `/Users/aryamangupta/CricML/Match_Prediction/betting_odds_v3.json` | 44-match WC 2024 test set |
| Current test matches | `/Users/aryamangupta/CricML/Match_Prediction/data/betting_test/` | 70 Cricsheet JSONs for WC 2024 |

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Test matches with odds | 44 | 500+ |
| Tournaments covered | 1 (WC 2024) | 10+ (IPL, BBL, BPL, SA20, T20I, etc.) |
| Date range | June 2024 | Nov 2023 – Apr 2026 |
| Odds source | Manual bookmaker | Polymarket prediction market |
| Statistical significance | Very low (44 matches) | Moderate (500+ matches) |
