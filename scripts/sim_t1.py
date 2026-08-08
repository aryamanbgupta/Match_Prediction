"""T4: TransformerT1 simulator wrapper (PredictionModel #8).

Drops the T1 innings transformer (scripts/transformer_t1.py) into
sim_v1_2's rules engine. Never edits the frozen eval framework — use via
scripts/sim_eval/run_sim_eval_t1.py (i8-style class swap).

Design notes (contract: research/reports/embeddings/T1_and_discriminability.md
+ sim reconnaissance):
  * The engine never resets model state and gives no outcome callback, so
    this wrapper derives everything from `state.history` each call: the
    innings token cache is validated against the engine's own ball count
    and rebuilt from BOS on any mismatch/new innings/new sim.
  * Feature order EXACTLY matches transformer_t1.build_features:
    EB_BAT(18) + EB_BOWL(18) + VENUE(6) + [is_mid, is_death, wkts/10,
    chasing] + [balls_rem/120, score/200, rr/12, clip(rrr,0,36)/12].
  * EB dists via sim_v1_2._fill_outcome_dists with k_player=30, k_venue=200
    (must match materialize-time; XGBoostModelV2 pattern).
  * Output: 6-class softmax -> 8-key dict, extras grafted at 0.01 each and
    renormalized (same as _build_outcome_dict).
Env: T1_SIM_MODEL_DIR overrides the checkpoint dir (default
models/embeddings/t1).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from embeddings_e1 import EB_BAT_COLS, EB_BOWL_COLS, VENUE_COLS  # noqa: E402
from sim_v1_2 import (PredictionModel, _OUTCOME_DIST_ZERO,  # noqa: E402
                      _fill_outcome_dists)
from stats_provider import wrap_with_cache  # noqa: E402
from transformer_t1 import BOS, T1Model  # noqa: E402

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
N_FEATS = len(EB_BAT_COLS) + len(EB_BOWL_COLS) + len(VENUE_COLS) + 8


def _outcome_class(runs: int, wicket: int) -> int:
    if wicket:
        return 5
    v = 2 if runs == 3 else 4 if runs == 5 else 6 if runs >= 7 else runs
    return {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}[v]


class TransformerT1SimModel(PredictionModel):
    """Accepts (and ignores) TransformerModelV1's constructor kwargs so the
    frozen runner's call site works unchanged after the class swap."""

    def __init__(self, model_path=None, batter_encoder_path=None,
                 bowler_encoder_path=None, feature_columns_path=None,
                 scaler_path=None, config_path=None, stats_provider=None,
                 player_metadata=None, matchup_encoder_path=None,
                 venue_encoder_path=None, max_seq_len=120, device="cpu",
                 use_mlx=False):
        self.stats_provider = wrap_with_cache(stats_provider)
        self.max_seq_len = max_seq_len
        mdir = Path(os.environ.get("T1_SIM_MODEL_DIR", "models/embeddings/t1"))
        self.model = T1Model(N_FEATS, dmodel=128, layers=2, heads=4)
        sd = torch.load(mdir / "model.pt", map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval()
        torch.set_num_threads(4)
        # innings token cache: rebuilt on any desync with state.history
        self._cache = {"innings": -1, "count": -1, "feats": [], "prev": []}
        print(f"TransformerT1SimModel loaded from {mdir} "
              f"({sum(p.numel() for p in self.model.parameters()):,} params)")

    # engine contract: extract_features(state) -> whatever predict accepts
    def extract_features(self, state):
        return state

    def _ball_features(self, state) -> np.ndarray:
        striker, bowler = state.current_striker, state.current_bowler
        f = dict(_OUTCOME_DIST_ZERO)
        _fill_outcome_dists(
            f, self.stats_provider, str(striker.player_id),
            str(bowler.player_id), state.venue, state.match_date,
            balls_bowled=state.balls, k_player=30.0, k_venue=200.0)
        team = state.current_team_idx
        balls = state.balls
        score = float(state.runs[team])
        rr = (score / (balls / 6.0)) if balls > 0 else 0.0
        try:
            rrr = float(state.required_run_rate or 0.0)
        except Exception:
            rrr = 0.0
        vec = np.array(
            [f[c] for c in EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS]
            + [1.0 if 36 <= balls < 96 else 0.0,
               1.0 if balls >= 96 else 0.0,
               (10.0 - float(state.wickets[team])) / 10.0,
               1.0 if (state.target or 0) > 0 else 0.0,
               (120.0 - balls) / 120.0,
               score / 200.0,
               rr / 12.0,
               float(np.clip(rrr, 0.0, 36.0)) / 12.0],
            dtype=np.float32)
        return vec

    def predict_next_ball(self, state):
        hist = state.history[:state.history_idx]
        rows = hist[hist[:, 0] == state.innings] if len(hist) else hist
        n = len(rows)
        c = self._cache
        if c["innings"] != state.innings or n < c["count"]:
            c.update(innings=state.innings, count=0, feats=[], prev=[])
        if n != c["count"]:
            # desync (should be rare): rebuild prev outcomes from history;
            # reuse last known features for balls we never predicted.
            filler = c["feats"][-1] if c["feats"] else None
            prev = [BOS] + [_outcome_class(int(r[3]), int(r[4]))
                            for r in rows[:-1]]
            feats = (c["feats"] + [filler] * (n - len(c["feats"])))[:n] \
                if filler is not None else []
            if len(feats) == n:
                c.update(count=n, feats=feats, prev=prev)
            else:
                c.update(count=0, feats=[], prev=[])
                rows = rows[:0]
                n = 0
        cur = self._ball_features(state)
        prev_y = _outcome_class(int(rows[-1][3]), int(rows[-1][4])) \
            if n > 0 else BOS
        c["feats"].append(cur)
        c["prev"].append(prev_y)
        c["count"] = n + 1
        feats = c["feats"][-self.max_seq_len:]
        prev = c["prev"][-self.max_seq_len:]
        with torch.no_grad():
            ft = torch.tensor(np.stack(feats)[None, :, :])
            pt = torch.tensor(np.array(prev, dtype=np.int64)[None, :])
            pad = torch.zeros_like(pt, dtype=torch.bool)
            out = self.model(ft, pt, pad)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits[0, -1], dim=-1).numpy()
        d = {name: float(p) * 0.98 for name, p in zip(CLASS_NAMES, probs)}
        d["wide"] = 0.01
        d["no_ball"] = 0.01
        return d
