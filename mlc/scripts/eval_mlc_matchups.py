"""Teacher-forced ball-level replay of the v7 sim model over MLC 2025, to grade
how well the ball model called specific batter-vs-bowler duels.

Unlike the Monte-Carlo prop backtest (which *simulates* matches), this replays
every ACTUAL delivery bowled in MLC 2025 through the same v7 ball model with the
real pre-ball state (the features already materialized in data/xgb_data_v3), and
compares the model's predicted per-ball outcome distribution to what actually
happened. Grouping by (bowler, batter) reads out which duels the model called.

IMPORTANT calibration caveat (verified on the full 186k-ball held-out test set):
the raw ball model systematically OVER-predicts tail outcomes (twos/fours/sixes/
wickets) and under-predicts dots/ones — the documented "over-states tails"
property. So absolute predicted economy / wicket numbers run hot; this analysis
is about RELATIVE ranking (who the model expected to dominate), standardized
within each distribution, not absolute rates.

The model is fed exactly as at serve time inside the sim: batter/bowler/matchup
categoricals via the saved v3 encoders, venue_encoded=0 (sim serve convention;
the real venue signal is in venue_p*/venue_avg_score). All other 110 features
come straight from the parquet. `ball_outcome` is raw off-bat runs {0,1,2,4,6}
with -1==wicket; the model head is {0:dot,1:one,2:two,3:four,4:six,5:wkt}.

Usage:
    uv run python scripts/eval_mlc_matchups.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "xgb_v3"
BALL_DIR = REPO / "data" / "xgb_data_v3"
PEOPLE = REPO / "data" / "cricsheet_people.csv"
OUT_MD = REPO / "mlc" / "reports" / "mlc_2025_matchups.md"
OUT_JSON = REPO / "mlc" / "reports" / "mlc_2025_matchups_detail.json"

MLC_VENUES = {
    "Grand Prairie Stadium, Dallas",
    "Central Broward Regional Park Stadium Turf Ground, Lauderhill",
    "Oakland Coliseum,Oakland",
}
DATE_LO, DATE_HI = "2025-06-12", "2025-07-13"
RUNS_BY_CLASS = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 0}
WICKET_CLASS = 5
RAW2CLS = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, -1: 5}
CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]


def main() -> int:
    model = joblib.load(MODEL / "xgboost_model_v3.pkl")
    bat_enc = joblib.load(MODEL / "batter_encoder_v3.pkl")
    bowl_enc = joblib.load(MODEL / "bowler_encoder_v3.pkl")
    matchup_enc = joblib.load(MODEL / "matchup_encoder_v3.pkl")
    feat_cols = [l.strip() for l in open(MODEL / "feature_columns_v3.txt") if l.strip()]
    bc = {str(c): i for i, c in enumerate(bat_enc.classes_)}
    wc = {str(c): i for i, c in enumerate(bowl_enc.classes_)}
    mc = {str(c): i for i, c in enumerate(matchup_enc.classes_)}
    classes = list(model.classes_)
    runs_vec = np.array([RUNS_BY_CLASS[c] for c in classes], float)
    wkt_idx = classes.index(WICKET_CLASS)

    frames = []
    for sp in ("validation", "test"):
        df = pd.read_parquet(BALL_DIR / f"cricket_data_v3_{sp}.parquet")
        df = df[df["venue"].isin(MLC_VENUES)
                & df["match_date"].astype(str).between(DATE_LO, DATE_HI)].copy()
        df["split"] = sp
        frames.append(df)
    b = pd.concat(frames, ignore_index=True)

    b["batter_encoded"] = b["batter_id"].astype(str).map(bc).fillna(-1).astype(int)
    b["bowler_encoded"] = b["bowler_id"].astype(str).map(wc).fillna(-1).astype(int)
    b["matchup_type_encoded"] = b["matchup_type"].astype(str).map(mc).fillna(-1).astype(int)
    b["venue_encoded"] = 0
    proba = model.predict_proba(b[feat_cols].astype(np.float64))
    b["pred_runs"] = proba @ runs_vec
    b["pred_wkt"] = proba[:, wkt_idx]
    b["act_runs"] = b["ball_outcome"].clip(lower=0)
    b["act_wkt"] = (b["ball_outcome"] == -1).astype(int)
    b["act_cls"] = b["ball_outcome"].map(RAW2CLS)

    names = dict(zip(pd.read_csv(PEOPLE, usecols=["identifier", "name"])["identifier"].astype(str),
                     pd.read_csv(PEOPLE, usecols=["identifier", "name"])["name"].astype(str)))
    b["batter"] = b["batter_id"].astype(str).map(names).fillna(b["batter_id"])
    b["bowler"] = b["bowler_id"].astype(str).map(names).fillna(b["bowler_id"])

    print(f"MLC 2025 deliveries: {len(b)} (June/val {sum(b.split=='validation')}, "
          f"July/test {sum(b.split=='test')})")

    # ---- 6-class calibration (the headline ball-level property) ----
    cls_cal = []
    for i, c in enumerate(classes):
        cls_cal.append((CLASS_NAMES[c], float(proba[:, i].mean()), float((b.act_cls == c).mean())))

    # ---- per-bowler / per-batter (rate-based; de-confounded) ----
    def grp(keys, mn):
        g = b.groupby(keys).agg(balls=("act_runs", "size"),
                                pred_runs=("pred_runs", "sum"), act_runs=("act_runs", "sum"),
                                pred_wkt=("pred_wkt", "sum"), act_wkt=("act_wkt", "sum")).reset_index()
        g = g[g.balls >= mn].copy()
        g["pred_econ"] = g.pred_runs / g.balls * 6
        g["act_econ"] = g.act_runs / g.balls * 6
        return g

    bowlers = grp("bowler", 18).sort_values("act_econ")
    batters = grp("batter", 15).sort_values("act_runs", ascending=False)
    batters["pred_sr"] = batters.pred_runs / batters.balls * 100
    batters["act_sr"] = batters.act_runs / batters.balls * 100
    pairs = grp(["bowler", "batter"], 9)

    def sp_r(a, c):
        return float(spearmanr(a, c).correlation) if len(a) > 2 else float("nan")
    bowl_r = sp_r(bowlers.pred_econ, bowlers.act_econ)
    bat_r = sp_r(batters.pred_sr, batters.act_sr)        # rate vs rate (de-confounded)
    pair_r = sp_r(pairs.pred_econ, pairs.act_econ)

    # Directional agreement on duels, standardized within each distribution.
    pmean, amean = pairs.pred_econ.mean(), pairs.act_econ.mean()
    pairs["pred_side"] = np.sign(pairs.pred_econ - pmean)   # + => model expects batter on top
    pairs["act_side"] = np.sign(pairs.act_econ - amean)
    dir_agree = float((pairs.pred_side == pairs.act_side).mean())

    # Illustrative duels: agreement in standardized space, ranked by combined extremity.
    pairs["pz"] = (pairs.pred_econ - pmean) / pairs.pred_econ.std()
    pairs["az"] = (pairs.act_econ - amean) / pairs.act_econ.std()
    bowler_dom = pairs[(pairs.pz < -0.4) & (pairs.az < -0.4)].copy()
    bowler_dom["score"] = -(bowler_dom.pz + bowler_dom.az)
    bowler_dom = bowler_dom.sort_values("score", ascending=False)
    batter_dom = pairs[(pairs.pz > 0.4) & (pairs.az > 0.4)].copy()
    batter_dom["score"] = batter_dom.pz + batter_dom.az
    batter_dom = batter_dom.sort_values("score", ascending=False)

    # ---- report ----
    L = ["# MLC 2025 — ball-level model: batter-vs-bowler matchup replay\n",
         "*Every actual MLC 2025 delivery replayed through the v7 sim ball model "
         "(`models/xgb_v3`) with its real pre-ball state; predicted outcome distribution vs "
         "actual, aggregated by player / duel. Off-bat runs (extras excluded). June balls are "
         "in the model's validation split (early-stop only), July fully held out.*\n",
         f"- **Deliveries**: {len(b)} (June/val {sum(b.split=='validation')}, "
         f"July/test {sum(b.split=='test')}).\n",
         "## Headline: the ball model runs hot on tail outcomes\n",
         "Mean predicted probability vs actual frequency, per outcome class (all MLC balls):\n",
         "| outcome | pred prob | actual freq | pred/actual |",
         "|---|---:|---:|---:|"]
    for nm, pp, af in cls_cal:
        L.append(f"| {nm} | {pp:.3f} | {af:.3f} | {pp/af:.2f}× |")
    L += ["\nThe model over-states boundaries and wickets and under-states dots/ones — the same "
          "tail-inflation seen in the prop backtest. So absolute predicted economy/wicket numbers "
          "below run hot; read them as **relative** rankings, not point estimates.\n",
          "## Can it rank who dominates? (Spearman rank-correlation, predicted vs actual)\n",
          "| level | n | metric | rank corr |",
          "|---|---:|---|---:|",
          f"| per bowler (≥18 balls) | {len(bowlers)} | economy | {bowl_r:+.2f} |",
          f"| per batter (≥15 balls) | {len(batters)} | strike rate | {bat_r:+.2f} |",
          f"| per duel (≥9 balls)    | {len(pairs)} | economy | {pair_r:+.2f} |",
          f"\nDuel-level directional agreement (model's batter-vs-bowler-favoured call matches "
          f"the actual side of the mean): **{dir_agree*100:.0f}%** of {len(pairs)} duels.\n",
          "Per-duel actuals rest on only ~9–12 balls, so duel-level signal is noise-dominated; "
          "the per-bowler / per-batter levels (more balls) are where ranking is testable.\n",
          f"## Per-bowler: predicted vs actual economy (≥18 balls)\n",
          "| bowler | balls | pred econ | act econ | pred wkts | act wkts |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in bowlers.iterrows():
        L.append(f"| {r.bowler} | {int(r.balls)} | {r.pred_econ:.1f} | {r.act_econ:.1f} "
                 f"| {r.pred_wkt:.1f} | {int(r.act_wkt)} |")
    L += [f"\n## Per-batter: predicted vs actual strike rate (≥15 balls, top 20 by runs)\n",
          "| batter | balls | act runs | pred SR | act SR |",
          "|---|---:|---:|---:|---:|"]
    for _, r in batters.head(20).iterrows():
        L.append(f"| {r.batter} | {int(r.balls)} | {int(r.act_runs)} | {r.pred_sr:.0f} | {r.act_sr:.0f} |")
    L += ["\n## Illustrative duels the model called right (≥9 balls; standardized agreement)\n",
          "**Bowler kept the batter quiet, as the model expected** (both predicted & actual "
          "economy in the low tail of their distributions):\n",
          "| bowler | batter | balls | pred econ | act econ | act wkts |",
          "|---|---|---:|---:|---:|---:|"]
    for _, r in bowler_dom.head(10).iterrows():
        L.append(f"| {r.bowler} | {r.batter} | {int(r.balls)} | {r.pred_econ:.1f} "
                 f"| {r.act_econ:.1f} | {int(r.act_wkt)} |")
    L += ["\n**Batter took the bowler down, as the model expected** (both in the high tail):\n",
          "| bowler | batter | balls | pred econ | act econ |",
          "|---|---|---:|---:|---:|"]
    for _, r in batter_dom.head(10).iterrows():
        L.append(f"| {r.bowler} | {r.batter} | {int(r.balls)} | {r.pred_econ:.1f} | {r.act_econ:.1f} |")

    OUT_MD.write_text("\n".join(L) + "\n")
    OUT_JSON.write_text(json.dumps({
        "n_balls": int(len(b)),
        "class_calibration": [{"outcome": nm, "pred": pp, "actual": af} for nm, pp, af in cls_cal],
        "rank_corr": {"bowler_econ": bowl_r, "batter_sr": bat_r, "pair_econ": pair_r},
        "duel_directional_agreement": dir_agree,
        "bowler_dom": bowler_dom.head(15).to_dict("records"),
        "batter_dom": batter_dom.head(15).to_dict("records"),
    }, indent=2, default=float))

    print(f"rank corr: bowler-econ {bowl_r:+.2f} | batter-SR {bat_r:+.2f} | duel-econ {pair_r:+.2f} "
          f"| duel dir-agree {dir_agree*100:.0f}%")
    print(f"report → {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
