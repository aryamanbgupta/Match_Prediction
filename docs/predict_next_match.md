# Predicting the Next Match — Step-by-step Playbook

End-to-end recipe for a forward fixture prediction (e.g. tonight's IPL game).
Runs **three models** — `xgb_match_v2_clean` (V2 frozen), `xgb_match_v2_clean_unfrozen`
(V2 unfrozen), `xgb_match_v3_m7_production` (M7 production) — against the same
feature row, and computes edge + sizing against bookmaker odds (American or
decimal). Captures everything we did for the 2026-05-11 DC vs PBKS run.

All Python commands run via `uv run` (per `~/.claude/CLAUDE.md`).

---

## 1. Gather match facts (web search)

For a fixture on date `D` between Team A and Team B, you need:

- **Venue** — full stadium name (encoder needs the same string the corpus uses)
- **Playing XIs** — both sides, ideally toss-time confirmed (not just predicted)
- **Toss result** — who won + bat/field (announced ~30 min before first ball)
- **Bookmaker odds** — to compute edge

Search pattern that worked:

```
IPL <year> May <DD> <Team A> vs <Team B> playing 11 toss venue
```

Reliable sources: ESPNcricinfo match page, Business Standard live blog, India.com /
Crictracker / myKhel previews (for predicted XIs while waiting for toss).

Cross-check the venue string against the model encoders before writing the fixture:

```bash
uv run python -c "
import joblib
for d in ['xgb_match_v2_clean','xgb_match_v2_clean_unfrozen','xgb_match_v3_m7_production']:
    e = joblib.load(f'models/{d}/encoders.pkl')
    matches = [c for c in e['venue'].classes_ if '<keyword>' in c.lower()]
    print(d, matches)
"
```

If the venue isn't in the encoder, the model falls back to encoder class 0 with a
warning — usable but signal is degraded. Pick the canonical name that appears.

---

## 2. Resolve player cricsheet IDs

The lineup entries in the fixture JSON can be 8-char cricsheet IDs OR display
names; names get resolved through `data/all_players_enriched.csv`. IDs are
preferred — no ambiguity.

**Easiest path**: each team's most recent match in the corpus already has the IDs.

```bash
# Find the team's most recent match (golden corpus has data through May 7):
uv run python3 -c "
import json, os
for f in os.listdir('data/golden/t20s_json'):
    if not f.endswith('.json'): continue
    d = json.load(open(f'data/golden/t20s_json/{f}'))
    if '<Team Name>' in d['info']['teams']:
        print(d['info']['dates'][0], f, d['info']['teams'])
" | sort | tail -5
```

Then dump that XI:

```bash
uv run python3 -c "
import json
d = json.load(open('data/golden/t20s_json/<file>.json'))
info = d['info']
regs = info.get('registry', {}).get('people', {})
team = '<Team Name>'
print(f'Venue: {info[\"venue\"]}')
print(f'Toss: {info.get(\"toss\")}')
for name in info['players'][team]:
    print(f'  {regs.get(name, \"???\"):>10}  {name}')
"
```

For new entrants not in any recent match, grep the enriched CSV:

```bash
grep -iE "<player surname>" data/all_players_enriched.csv | head -5
```

(Format: `cricsheet_id,name,cricinfo_id,...`)

---

## 3. Write the fixture JSON

Copy `fixtures/_template.json` to `fixtures/<date>_<teams>.json`. Required keys:

```json
{
  "_comment": "Context: source of XIs, any caveats, impact-sub assumptions",
  "date": "YYYY-MM-DD",
  "team1": "<full canonical name>",
  "team2": "<full canonical name>",
  "venue": "<exact encoder string>",
  "competition_tier": "Indian Premier League",
  "team_type": "club",
  "team1_lineup": ["<8-char ID>", ...],   // 11 IDs
  "_team1_lineup_names": ["...", ...],    // human-readable, ignored by model
  "team2_lineup": ["...", ...],
  "_team2_lineup_names": ["...", ...],
  "toss_winner": "<team name or null>",
  "toss_decision": "bat" | "field" | null,
  "polymarket_odds": null
}
```

**Team1/team2 convention**: home team is conventionally team1, but the model
uses symmetric features (`elo_diff_*`, `is_team1_home` vs `is_team2_home`,
`win_rate_diff`) so the ordering does not bias the prediction. Use whichever
framing the user gives.

---

## 4. (Optional) Update trackers with the latest data

The default tracker snapshot (`data/tracker_snapshot_test_end.pkl`) walks
`data/t20s_json` only, **excluding** golden data (2026-04-17+) by design — that
preserves golden as a clean out-of-sample eval set.

For a **forward production prediction**, you usually want more recent state.
Two options:

### 4a. Skip the update — accept stale form

The default snapshot is fine for a quick prediction. Form/H2H/home will lag by
~3+ weeks, but the model's other features (player ELOs, batting/bowling
strength) are not refreshed either, so it's at least internally consistent.

### 4b. Rebuild an inclusive snapshot (recommended for forward bets)

Walks both `data/t20s_json` and `data/golden/t20s_json`, plus lets you
manually append matches that aren't in either corpus yet. Save it to a
**separate pickle** so the iteration snapshot remains golden-clean:

```python
# /tmp/build_inclusive_snapshot.py
import sys, json, pickle
from pathlib import Path
from datetime import datetime
sys.path.insert(0, 'scripts')
from materialize_match_features import TeamFormTracker, H2HTracker, HomeVenueTracker
from loaders_common import iter_matches_chronological

OUT = Path('data/tracker_snapshot_<date>_inclusive.pkl')

form = TeamFormTracker()
h2h = H2HTracker()
home = HomeVenueTracker(lookback_days=730)

def feed(source_dir):
    for mid, json_text, match_date in iter_matches_chronological(source_dir, gender='male'):
        data = json.loads(json_text)
        info = data.get('info') or {}
        teams = info.get('teams') or []
        if len(teams) != 2: continue
        outcome = info.get('outcome') or {}
        winner = outcome.get('winner')
        if not winner and outcome.get('result') == 'tie':
            winner = outcome.get('eliminator')
        if not winner or winner not in teams: continue
        venue = info.get('venue', 'unknown')
        t1, t2 = teams
        t1_won = winner == t1
        form.update(t1, match_date, t1_won)
        form.update(t2, match_date, not t1_won)
        h2h.update(t1, t2, match_date, winner)
        home.update(t1, venue, match_date)
        home.update(t2, venue, match_date)

feed('data/t20s_json')
feed('data/golden/t20s_json')

# Manually append matches not in either corpus (very recent):
manual = [
    # (date, team1, team2, winner, venue)
    ('YYYY-MM-DD', '<T1>', '<T2>', '<winner>', '<venue>'),
]
latest = None
for date_str, t1, t2, winner, venue in manual:
    d = datetime.strptime(date_str, '%Y-%m-%d')
    latest = d if latest is None or d > latest else latest
    t1_won = winner == t1
    form.update(t1, d, t1_won)
    form.update(t2, d, not t1_won)
    h2h.update(t1, t2, d, winner)
    home.update(t1, venue, d)
    home.update(t2, venue, d)

snap = {
    'as_of': latest.strftime('%Y-%m-%d') if latest else None,
    'form_records': dict(form.records),
    'h2h_records': {tuple(sorted(k)): v for k, v in h2h.records.items()},
    'home_records': dict(home.records),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'wb') as f:
    pickle.dump(snap, f)
print(f'Saved -> {OUT}')
```

Then point `predict_fixture.TRACKER_SNAPSHOT` at the new pickle in the runner
(see step 5).

### Tracker layers — what gets refreshed vs what stays frozen

| Layer | What it holds | Refresh path |
|---|---|---|
| **Tracker snapshot pickle** (`tracker_snapshot_*.pkl`) | Team-level form (last-N win rate), H2H, home-venue record | Rebuild via the script above (~30s) |
| **SQLite stats cache** (`models/player_stats_cache_v3.sqlite`) | Per-player batting/bowling rolling stats, ELOs, outcome distributions, venue stats | Rebuild via `scripts/build_stats_cache.py` (~5-10 min) — frozen at 2026-04-16 by default |
| **Per-fixture rehydration** | Whatever's in SQLite + tracker pickle, queried as-of fixture date | Automatic — `compute_features` does this every call |

**The SQLite cache is the expensive layer.** If you don't rebuild it, player
ELOs, top-6/bot-5 splits, and the M7 venue outcome-dist features (`venue_p4`,
`venue_p6`, `venue_pw`) reflect their 2026-04-16 state regardless of fixture
date. Rebuilding is straightforward but requires the extended corpus —
follow the CLAUDE.md quick-start "Golden eval refresh" recipe.

For a one-off prediction, the team-level tracker update alone usually captures
the user-visible "recent form" signal that matters; player-level staleness is
usually marginal.

---

## 5. Run all three models against the same feature row

For a single-model M7 run, the shipped CLI works directly:

```bash
uv run python scripts/predict_fixture.py --fixture fixtures/<file>.json
# or against a different model:
uv run python scripts/predict_fixture.py --fixture fixtures/<file>.json \
    --model-dir models/xgb_match_v2_clean_unfrozen
```

For the three-way sanity check, write a thin runner that calls
`predict_fixture.compute_features` once, then passes `model_dir` per call.
`compute_features` now emits the M7 venue outcome-dist features
(`venue_p4`/`p6`/`pw`) directly — no manual bolt-on needed.

```python
# /tmp/predict_three_models.py
import sys, json
from pathlib import Path
sys.path.insert(0, 'scripts')
import predict_fixture as pf
from stats_provider import StatsProvider
from player_metadata import PlayerMetadataProvider

# Optional: swap in the updated snapshot built in step 4b
pf.TRACKER_SNAPSHOT = Path('data/tracker_snapshot_<date>_inclusive.pkl')

fixture = json.loads(Path('fixtures/<file>.json').read_text())

# Optional: attach bookmaker odds (decimal). For American odds, convert first:
#   negative (-176): decimal = 1 + 100 / abs(odds)
#   positive (+138): decimal = 1 + odds / 100
fixture['polymarket_odds'] = {
    fixture['team1']: <decimal_for_team1>,
    fixture['team2']: <decimal_for_team2>,
}

provider = StatsProvider('models', version='v3')
metadata = PlayerMetadataProvider('data/all_players_enriched.csv')
form, h2h, home = pf.load_trackers()
record = pf.compute_features(fixture, provider, metadata, form, h2h, home)

# OPTIONAL home-flag override. The HomeVenueTracker needs >=3 prior matches at
# a venue in the last 730d to trigger. Anchor home grounds (Wankhede for MI,
# Chepauk for CSK) usually trigger; secondary homes (Dharamsala for PBKS,
# Raipur for RCB, Visakhapatnam for DC) often don't. Override only if you
# trust the venue is genuinely team2's home.
# record['is_team2_home'] = 1

models = [
    ('V2 clean (frozen)',           Path('models/xgb_match_v2_clean')),
    ('V2 clean (unfrozen)',         Path('models/xgb_match_v2_clean_unfrozen')),
    ('M7 production (active)',      Path('models/xgb_match_v3_m7_production')),
]
for label, mdir in models:
    p_t1, _ = pf.apply_encoders_and_predict(record, mdir)
    print(f'{label}: P({fixture["team1"]})={p_t1*100:.1f}%  '
          f'P({fixture["team2"]})={(1-p_t1)*100:.1f}%')
```

The V2 models ignore `venue_p4/p6/pw` (not in their `feature_columns.txt`);
the M7 model uses them. A single `record` dict feeds all three.

**Why all three?** V2 frozen was the May 8/9 production model; V2 unfrozen was
May 10's; M7 is the post-2026-05-10 default. Running all three is a sanity check
— if they diverge a lot, dig into why before betting. They share the same 46
features (M7 also uses 3 venue outcome-dist features), so divergence is purely a
function of weights.

---

## 6. Compute edge + sizing

Predict_fixture.compute_bet handles this if you pass `polymarket_odds`, but for
visibility on all three models, do it inline:

```python
mkt_t1   = 1.0 / fixture['polymarket_odds'][fixture['team1']]
mkt_t2   = 1.0 / fixture['polymarket_odds'][fixture['team2']]
vig_pct  = (mkt_t1 + mkt_t2 - 1) * 100   # bookmaker overround

# per model, after p_t1 is computed:
edge_t1 = p_t1 - mkt_t1     # in probability points
edge_t2 = (1-p_t1) - mkt_t2
# Full Kelly (informational only — see below):
def kelly(p, dec):
    b = dec - 1.0
    q = 1.0 - p
    return max(0.0, (b * p - q) / b)
```

**Sizing rule** (`project_match_v3_m8.md`): the project's adopted rule is
**flat 1 unit when edge > 0**, regardless of edge magnitude. Edge-based
thresholds (1pp / 5pp / 10pp) all hurt iteration ROI in M8 testing. So:

- Edge positive → bet 1 unit on the team with the larger positive edge.
- Edge negative for both → no bet.

The Kelly fractions are useful as sanity (very large Kelly = either huge model
edge or weak vig, both worth a second look), but the actual stake is 1u.

---

## 7. Decision rules

- **All three models agree on the +EV side, edge > 0** → bet 1u, ship it.
- **Models disagree** (one says bet DC, another says bet PBKS, or one says no
  bet) → no bet. The disagreement is the signal.
- **All three say no bet** → no bet.
- **Model probabilities are within 1-2pp of market** → no bet — vig-affected
  edges are too noisy to trust.

The May 11 DC vs PBKS run was the canonical "all three agree" case: edges
ranged +6.2pp (M7) to +10.0pp (V2 variants), all on DC at +138.

---

## 8. Save the output

```python
out_path = Path(f'predictions/{date}_{team1.replace(" ","_")}_vs_{team2.replace(" ","_")}.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    'fixture': fixture,
    'predictions_by_model': { ... },
    'odds_given': { ... },
    'diagnostics': { ... },
}, indent=2))
```

Existing examples to mirror format:

- `predictions/2026-05-10_Royal_Challengers_Bengaluru_vs_Mumbai_Indians.json`
- `predictions/2026-05-11_Delhi_Capitals_vs_Punjab_Kings_updated.json`

---

## Caveats & gotchas

- **SQLite freeze at 2026-04-16** means player ELOs / outcome dists lag. Even
  with an inclusive tracker pickle, player-level features are stale unless you
  rebuild the cache.
- **Same-day match order** matters during materialization (CLAUDE.md
  invariant #5). It does not affect single-fixture forward prediction — the
  rehydration path queries as-of fixture_date, not as-of a within-day index.
- **Encoder warnings** ("unseen venue=...") fall back to encoder class 0.
  Treat any warning as a confidence dent.
- **Home-flag override** is a real intervention — only do it when you have
  domain knowledge (e.g. the user explicitly says "this is X's home"). Don't
  flip it just because a team is playing in their state.
- **The M7 production model is calibrated to be less aggressive than V2** (lr
  0.10 → 0.05, cs 0.8 → 0.9 from M7.A). Expect M7 probabilities to cluster
  closer to 50% than V2 — that's the intended behavior and the reason close-
  slice ROI CI cleanly excluded zero at M7.
