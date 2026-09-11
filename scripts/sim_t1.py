# manifest-exempt: embeddings-ladder experimental artifact namespace
"""T4: TransformerT1 simulator wrapper (PredictionModel #8).

Drops the T1 innings transformer (scripts/transformer_t1.py) into
sim_v1_2's rules engine. Never edits the frozen eval framework — use via
scripts/sim_eval/run_sim_eval_t1.py (i8-style class swap).

Design notes (contract: research/reports/embeddings/T1_and_discriminability.md
+ sim reconnaissance):
  * The engine never resets model state and gives no outcome callback, so
    this wrapper keys its token cache to the exact mutable MatchState and
    validates it against `state.history` on every call. New innings/sims
    reset at BOS; any other mismatch fails closed.
  * Feature order EXACTLY matches transformer_t1.build_features:
    EB_BAT(18) + EB_BOWL(18) + VENUE(6) + [is_mid, is_death, wkts/10,
    chasing] + [balls_rem/120, score/200, rr/12, clip(rrr,0,36)/12].
  * EB dists via sim_v1_2._fill_outcome_dists with k_player=30, k_venue=200
    (must match materialize-time; XGBoostModelV2 pattern).
  * Delivery contract: the v3 target is `inclusive_total_runs_v1`, matching
    the promoted i7 wrapper. The default output path therefore preserves the
    production flat 1% wide + 1% no-ball graft. An explicit B18 sidecar can
    opt into the landed empirical graft without silently changing defaults.
  * Sequence length comes from the checkpoint's 200-position embedding, not
    the legacy runner's 120-ball argument. Training innings contain extras
    and commonly exceed 120 delivery rows.
Env: T1_SIM_MODEL_DIR overrides the checkpoint dir (default
models/embeddings/t1); T1_SIM_DEVICE overrides the runner device;
T1_EXTRAS_GRAFT_PATH opts into a B18 `extras_graft_v1.json` sidecar.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from embeddings_e1 import EB_BAT_COLS, EB_BOWL_COLS, VENUE_COLS  # noqa: E402
from player_metadata import PlayerMetadataProvider  # noqa: E402
from sim_v1_2 import (ExtrasGraftConfig, PredictionModel,  # noqa: E402
                      _OUTCOME_DIST_ZERO, _fill_outcome_dists,
                      graft_extras)
from stats_provider import wrap_with_cache  # noqa: E402
from transformer_t1 import BOS, T1Model  # noqa: E402

CLASS_NAMES = ["dot", "one", "two", "four", "six", "wicket"]
N_FEATS = len(EB_BAT_COLS) + len(EB_BOWL_COLS) + len(VENUE_COLS) + 8
_COUNT_SUFFIXES = ("0", "1", "2", "4", "6", "w")

# Single source for the shrinkage constants. T1 checkpoints carry no
# outcome_dist_config sidecar (unlike XGBoost artifacts): the v3 parquet was
# materialized with the parsing defaults, so these MUST stay the parsing_v2 /
# materialize-time defaults or serving features diverge from training.
K_PLAYER = 30.0
K_VENUE = 200.0

# The online overlay generates its feature names by string formatting; pin
# them to the imported training columns so a naming drift fails at import
# instead of surviving until the parity audit.
_GENERATED_PLAYER_KEYS = (
    [f"batter_p{s}" for s in _COUNT_SUFFIXES]
    + [f"batter_p{s}_vs_{label}" for label in ("pace", "spin")
       for s in _COUNT_SUFFIXES]
    + [f"bowler_p{s}" for s in _COUNT_SUFFIXES]
    + [f"bowler_p{s}_vs_{label}" for label in ("lhb", "rhb")
       for s in _COUNT_SUFFIXES]
)
assert set(_GENERATED_PLAYER_KEYS) == set(EB_BAT_COLS + EB_BOWL_COLS), (
    "OnlineT1OutcomeDists feature names diverged from the training columns"
)

# --- serving contract guard (sequence track stage 1, D4) -----------------
# A checkpoint retrained by transformer_t1.py carries a `training_contract`
# block naming the frame's delivery semantics, its venue-identity version,
# the ordered feature names, and the stats cache the frame was materialized
# from. Features are rebuilt here from live tracker state, so an i7-frame
# checkpoint served over a cache with different venue identity (or none)
# silently gets the wrong venue rows.
#
# EXACTLY what this guard checks, and what it does not (Astra round 1):
#   * contract.delivery_semantics == SERVING_DELIVERY_SEMANTICS — equality;
#   * contract.venue_alias_version == serving `_meta.venue_alias_version` —
#     equality, both non-null;
#   * serving `_meta.same_day_order_version` — PRESENCE only. It is not
#     compared with the contract's recorded ordering version;
#   * contract.stats_cache.md5 — PRESENCE only (non-null). It is NOT
#     compared against the cache being served: the wrapper is handed a
#     provider, not a file path, and cannot hash what is behind it. So a
#     contracted checkpoint whose training cache has since been rebuilt
#     LOADS here as long as the alias versions still agree;
#   * contract.feature_names == EXPECTED_FEATURE_NAMES — ordered equality.
# The training-cache md5 is compared against the live cache file by
# `scripts/sequence_track/pin_stage1.py --verify` (`_require(cache["md5"],
# stats_cache_md5, ...)`), which `run_arm.py --config` re-runs before it
# launches a registered arm. That is the boundary: this guard makes an
# identity mismatch unservable, the pin makes a *drifted* cache unrunnable.
#
# A checkpoint WITHOUT the block is a legacy v3 checkpoint: the old
# `config.data_dir` substring rule stands, and the serving cache must carry
# no venue identity at all, so the uncontracted-checkpoint-on-the-i7-cache
# pairing is refused rather than assumed benign. Since the SHOULD 2 fix,
# transformer_t1.py writes no contract at all for a frame that declares no
# delivery semantics, so such checkpoints arrive on this path by design
# rather than carrying a contract full of nulls.
from embeddings_e1 import CTX_COLS  # noqa: E402
from transformer_t1 import STATE_COLS  # noqa: E402

SERVING_DELIVERY_SEMANTICS = "inclusive_total_runs_v1"
EXPECTED_FEATURE_NAMES = tuple(
    EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS + CTX_COLS + STATE_COLS)
assert len(EXPECTED_FEATURE_NAMES) == N_FEATS, (
    "serving feature-name order diverged from N_FEATS"
)


def _serving_cache_meta(provider, mdir) -> dict:
    """Return the serving stats cache's `_meta`, or refuse.

    Production hands the wrapper a `SameDayReplayStatsProvider`, whose
    `StatsProviderCache.__getattr__` forwards to the `_TrackerStatsView`,
    which forwards `get_cache_meta()` to the `StatsProvider` it rehydrated
    from, which reads the SQLite backend's `_meta`. A provider that cannot
    expose `_meta` leaves the serving identity unverifiable, so it is
    refused instead of assumed compatible.
    """
    getter = getattr(provider, "get_cache_meta", None)
    if not callable(getter):
        raise RuntimeError(
            f"T1 checkpoint {mdir}: serving provider "
            f"{type(provider).__name__} cannot expose the stats cache "
            "`_meta` (no get_cache_meta()), so its venue identity is "
            "unverifiable. Serve through a StatsProvider-backed chain "
            "(SameDayReplayStatsProvider -> StatsProvider -> SQLite "
            "get_meta())."
        )
    try:
        return dict(getter())
    except Exception as exc:
        raise RuntimeError(
            f"T1 checkpoint {mdir}: serving provider "
            f"{type(provider).__name__}.get_cache_meta() did not yield a "
            f"`_meta` mapping ({exc!r}); refusing to serve without a "
            "verified cache identity."
        ) from exc


def _verify_training_contract(metrics: dict, mdir, provider):
    """Refuse any checkpoint / serving-cache pair whose delivery semantics
    or venue identity do not match. Returns the contract block (or None for
    a legacy, uncontracted checkpoint).

    See the boundary note above the module's guard constants: the recorded
    stats-cache md5 is required to be present but is NOT compared with the
    cache being served — that comparison is pin_stage1.py --verify's.
    """
    cfg = metrics.get("config", {})
    meta = _serving_cache_meta(provider, mdir)
    serving_alias = meta.get("venue_alias_version")
    contract = metrics.get("training_contract")

    if contract is None:
        # The online overlay counts EVERY delivery on total runs — the v3
        # materializer's rule. A checkpoint trained on another corpus (e.g.
        # the I5 legal/off-bat frame, which counts legal off-bat runs only)
        # would silently receive wrong features here.
        trained_data_dir = cfg.get("data_dir")
        if trained_data_dir is not None and (
                "xgb_data_v3" not in str(trained_data_dir)):
            raise RuntimeError(
                f"T1 checkpoint {mdir} was trained on {trained_data_dir!r}; "
                "this wrapper implements the v3 inclusive_total_runs_v1 "
                "delivery semantics and only serves data/xgb_data_v3 "
                "checkpoints."
            )
        if serving_alias is not None:
            raise RuntimeError(
                f"T1 checkpoint {mdir} carries no training_contract "
                f"(config.data_dir={trained_data_dir!r}), but the serving "
                "stats cache declares _meta.venue_alias_version="
                f"{serving_alias!r} (expected absent). An uncontracted "
                "checkpoint may only be served from a cache with no venue "
                "identity; retrain on the i7 frame to serve the i7 cache."
            )
        return None

    if not isinstance(contract, dict):
        raise RuntimeError(
            f"T1 checkpoint {mdir}: training_contract is "
            f"{type(contract).__name__}, expected an object."
        )

    semantics = contract.get("delivery_semantics")
    if semantics != SERVING_DELIVERY_SEMANTICS:
        raise RuntimeError(
            f"T1 checkpoint {mdir}: training_contract.delivery_semantics="
            f"{semantics!r}, this wrapper serves "
            f"{SERVING_DELIVERY_SEMANTICS!r}."
        )

    contract_alias = contract.get("venue_alias_version")
    if contract_alias is None:
        raise RuntimeError(
            f"T1 checkpoint {mdir}: training_contract.venue_alias_version="
            f"{contract_alias!r}; a contracted checkpoint must name the "
            "venue identity of the frame it was trained on."
        )

    stats_cache = contract.get("stats_cache")
    cache_md5 = (stats_cache.get("md5")
                 if isinstance(stats_cache, dict) else None)
    if cache_md5 is None:
        raise RuntimeError(
            f"T1 checkpoint {mdir}: training_contract.stats_cache.md5="
            f"{cache_md5!r} (stats_cache={stats_cache!r}); a checkpoint "
            "trained without a resolved --stats-cache-role cannot be "
            "served."
        )

    if serving_alias != contract_alias:
        raise RuntimeError(
            f"T1 checkpoint {mdir}: venue_alias_version mismatch — "
            f"training_contract.venue_alias_version={contract_alias!r}, "
            f"serving cache _meta.venue_alias_version={serving_alias!r}."
        )

    order_version = meta.get("same_day_order_version")
    if not order_version:
        raise RuntimeError(
            f"T1 checkpoint {mdir}: serving cache "
            f"_meta.same_day_order_version={order_version!r} (expected "
            f"present, e.g. the contract's "
            f"{(stats_cache or {}).get('same_day_order_version')!r}); "
            "same-day ordering is a versioned data contract."
        )

    names = list(contract.get("feature_names") or [])
    expected = list(EXPECTED_FEATURE_NAMES)
    if names != expected:
        if len(names) != len(expected):
            where = f"length {len(names)} != {len(expected)}"
        else:
            i = next(j for j in range(len(names)) if names[j] != expected[j])
            where = f"position {i}: {names[i]!r} != {expected[i]!r}"
        raise RuntimeError(
            f"T1 checkpoint {mdir}: training_contract.feature_names does "
            f"not match this wrapper's serving order ({where}); "
            f"training_contract.feature_names={names!r}, "
            f"wrapper order={expected!r}."
        )
    return contract


def _outcome_class(runs: int, wicket: int) -> int:
    if wicket:
        return 5
    v = 2 if runs == 3 else 4 if runs == 5 else 6 if runs >= 7 else runs
    return {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}[v]


class OnlineT1OutcomeDists:
    """Causal within-match overlay for T1's 36 player EB features.

    The v3 materializer updates its player tracker after every delivery, but
    the SQLite provider deliberately returns a pre-match snapshot. This class
    adds simulated delivery counts to that snapshot using the exact legacy
    six-class counting and hierarchical shrinkage rules. Venue distributions
    remain pre-match, matching materialization (venue state updates only after
    the match).
    """

    def __init__(self, provider, metadata):
        self.provider = provider
        self.metadata = metadata
        # The 2.4e-7 parity audit certifies ONLY the live same-day count
        # path (a SameDayReplayStatsProvider). The old fallback read
        # date-snapshot SQLite counts that EXCLUDE earlier same-day matches,
        # so serving features diverged from training on multi-fixture days;
        # its last consumer (run_sim_eval_t1's plain StatsProvider) now
        # builds a replay provider, so the fallback is deleted outright.
        if not hasattr(provider, "get_t1_outcome_counts"):
            raise RuntimeError(
                "OnlineT1OutcomeDists requires a provider with live "
                "same-day counts (get_t1_outcome_counts): snapshot counts "
                "are NOT covered by the T1 parity audit and diverge from "
                "training features on multi-fixture days. Serve through a "
                "SameDayReplayStatsProvider (see run_t1_sim_ppc.py or "
                "run_sim_eval_t1.py)."
            )
        self._state = None
        self._processed = 0
        self._local_batter = defaultdict(self._zeros)
        self._local_bowler = defaultdict(self._zeros)
        self._local_batter_type = defaultdict(self._zeros)
        self._local_bowler_hand = defaultdict(self._zeros)

    @staticmethod
    def _zeros():
        return np.zeros(6, dtype=np.int64)

    @staticmethod
    def _shrink(counts, prior, k: float):
        total = float(sum(counts))
        return tuple((float(counts[i]) + k * float(prior[i]))
                     / (total + k) for i in range(6))

    def _reset(self, state):
        self._state = state
        self._processed = 0
        self._local_batter.clear()
        self._local_bowler.clear()
        self._local_batter_type.clear()
        self._local_bowler_hand.clear()

    @staticmethod
    def _lineup(state, team_idx: int):
        return state.team1_lineup if team_idx == 0 else state.team2_lineup

    def _sync(self, state):
        if self._state is not state:
            if state.history_idx:
                raise RuntimeError(
                    "T1 online EB state cannot attach to a copied/in-progress "
                    "MatchState without replaying its feature-bearing history"
                )
            self._reset(state)
        if state.history_idx < self._processed:
            raise RuntimeError("T1 online EB history moved backwards")

        for row in state.history[self._processed:state.history_idx]:
            batting_team = int(row[5])
            bowling_team = int(row[7])
            batter = self._lineup(state, batting_team).players[int(row[6])]
            bowler = self._lineup(state, bowling_team).players[int(row[8])]
            batter_id = str(batter.player_id)
            bowler_id = str(bowler.player_id)
            bucket = _outcome_class(int(row[3]), int(row[4]))
            self._local_batter[batter_id][bucket] += 1
            self._local_bowler[bowler_id][bucket] += 1

            batter_meta = self.metadata.get_player_metadata(batter_id)
            bowler_meta = self.metadata.get_player_metadata(bowler_id)
            is_pace = bowler_meta["is_pace"]
            batter_hand = batter_meta["batter_hand"]
            if is_pace is not None:
                self._local_batter_type[(batter_id, int(not is_pace))][bucket] += 1
            if batter_hand in {"left", "right"}:
                hand_code = 0 if batter_hand == "left" else 1
                self._local_bowler_hand[(bowler_id, hand_code)][bucket] += 1
        self._processed = state.history_idx

    def _base(self, kind: str, entity: str, date, cell=None):
        return np.asarray(
            self.provider.get_t1_outcome_counts(kind, entity, date, cell=cell),
            dtype=np.int64,
        )

    @staticmethod
    def _named(prefix: str, values):
        return {f"{prefix}{suffix}": values[i]
                for i, suffix in enumerate(_COUNT_SUFFIXES)}

    def features(self, state, batter_id: str, bowler_id: str,
                 k_player: float = K_PLAYER, k_venue: float = K_VENUE):
        self._sync(state)
        date = state.match_date
        prior = tuple(self.provider.get_t1_outcome_prior())
        batter_counts = (
            self._base("batter", batter_id, date)
            + self._local_batter[batter_id]
        )
        bowler_counts = (
            self._base("bowler", bowler_id, date)
            + self._local_bowler[bowler_id]
        )
        batter_parent = self._shrink(batter_counts, prior, k_player)
        bowler_parent = self._shrink(bowler_counts, prior, k_player)
        out = {
            **self._named("batter_p", batter_parent),
            **self._named("bowler_p", bowler_parent),
        }
        for cell, label in ((0, "pace"), (1, "spin")):
            counts = (
                self._base("batter_type", batter_id, date, cell)
                + self._local_batter_type[(batter_id, cell)]
            )
            values = self._shrink(counts, batter_parent, k_player)
            out.update({f"batter_p{suffix}_vs_{label}": values[i]
                        for i, suffix in enumerate(_COUNT_SUFFIXES)})
        for cell, label in ((0, "lhb"), (1, "rhb")):
            counts = (
                self._base("bowler_hand", bowler_id, date, cell)
                + self._local_bowler_hand[(bowler_id, cell)]
            )
            values = self._shrink(counts, bowler_parent, k_player)
            out.update({f"bowler_p{suffix}_vs_{label}": values[i]
                        for i, suffix in enumerate(_COUNT_SUFFIXES)})
        out.update(self.provider.get_venue_outcome_dist(
            state.venue, date, k=k_venue))
        return out


class TransformerT1SimModel(PredictionModel):
    """Accepts (and ignores) TransformerModelV1's constructor kwargs so the
    frozen runner's call site works unchanged after the class swap."""

    # Class-level default: only bare instances built via __new__ (focused
    # formula tests, `for_feature_audit`) see it. __init__ always installs a
    # live overlay; the None branch in `_ball_features` is the explicit
    # snapshot path for those bare instances, never a silent fallback.
    online_outcome_dists = None

    def __init__(self, model_path=None, batter_encoder_path=None,
                 bowler_encoder_path=None, feature_columns_path=None,
                 scaler_path=None, config_path=None, stats_provider=None,
                 player_metadata=None, matchup_encoder_path=None,
                 venue_encoder_path=None, max_seq_len=120, device="cpu",
                 use_mlx=False):
        self.stats_provider = wrap_with_cache(stats_provider)
        self.player_metadata = player_metadata or PlayerMetadataProvider(
            os.environ.get("T1_PLAYER_METADATA_PATH",
                           "data/all_players_enriched.csv"))
        self.online_outcome_dists = OnlineT1OutcomeDists(
            self.stats_provider, self.player_metadata)
        mdir = Path(os.environ.get("T1_SIM_MODEL_DIR", "models/embeddings/t1"))
        with (mdir / "metrics.json").open() as fh:
            metrics = json.load(fh)
        cfg = metrics.get("config", {})
        # Delivery semantics, venue identity and feature order of the frame
        # this checkpoint was trained on, checked against the cache actually
        # being served. See `_verify_training_contract`.
        _verify_training_contract(metrics, mdir, self.stats_provider)
        self.arm = cfg.get("arm", "full")
        self.model = T1Model(
            N_FEATS,
            dmodel=int(cfg.get("dmodel", 128)),
            layers=int(cfg.get("layers", 2)),
            heads=int(cfg.get("heads", 4)),
            arm=self.arm,
        )
        sd = torch.load(mdir / "model.pt", map_location="cpu",
                        weights_only=True)
        self.model.load_state_dict(sd)
        requested_device = os.environ.get("T1_SIM_DEVICE", device)
        if requested_device == "auto":
            # Same preference order as transformer_t1's --device auto.
            requested_device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("T1_SIM_DEVICE=mps requested but MPS is unavailable")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "T1_SIM_DEVICE=cuda requested but CUDA is unavailable")
        self.device = torch.device(requested_device)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cpu":
            torch.set_num_threads(4)

        # The frozen runner passes max_seq_len=120 for its older transformer.
        # T1 trained on whole delivery-row innings (validation max 138; test
        # max 144), and its position table is the authoritative capacity.
        self.runner_max_seq_len = int(max_seq_len)
        self.max_seq_len = (
            int(self.model.pos_emb.num_embeddings)
            if hasattr(self.model, "pos_emb") else 1
        )

        # Explicitly match the v3 training and promoted i7 serving contract.
        self.delivery_semantics = "inclusive_total_runs_v1"
        self.extras_process = None
        self.extras_graft = None
        graft_path = os.environ.get("T1_EXTRAS_GRAFT_PATH")
        if graft_path:
            self.extras_graft = ExtrasGraftConfig.from_path(Path(graft_path))

        # A cache is valid only for the exact mutable MatchState instance that
        # produced it. Desync now fails closed; fabricating missing historical
        # features from the last known token silently changes the model input.
        self._cache = self._empty_cache()
        # Opt-in O(L) incremental forward (vs the default full re-forward,
        # whose innings cost is quadratic in deliveries). Kept opt-in because
        # the certified PPC raws were produced with the full re-forward and
        # the incremental path differs at float epsilon (equivalence pinned
        # by tests/test_sim_t1_prefix_cache.py).
        self.prefix_cache_enabled = (
            os.environ.get("T1_SIM_PREFIX_CACHE") == "1"
            and self.arm in ("full", "no_history"))
        print(f"TransformerT1SimModel loaded from {mdir} "
              f"({sum(p.numel() for p in self.model.parameters()):,} params; "
              f"arm={self.arm}; device={self.device}; "
              f"context_capacity={self.max_seq_len}; "
              f"prefix_cache={'ON' if self.prefix_cache_enabled else 'off'}; "
              f"delivery_semantics={self.delivery_semantics})")
        if self.extras_graft is not None:
            print(self.extras_graft.banner())
            print(f"  sidecar: {self.extras_graft.source}")

    # Bare __new__ instances (focused tests) see the class default.
    prefix_cache_enabled = False

    @staticmethod
    def _empty_cache():
        return {"state": None, "innings": None, "feats": [], "prev": [],
                "kv": None, "kv_len": 0}

    @classmethod
    def for_feature_audit(cls, provider, metadata):
        """Feature-path-only instance for the parity audit: the exact serving
        `_ball_features` path (cache wrap + online overlay) without loading a
        torch checkpoint. `wrap_with_cache` must be the identity for a
        replay provider — assert it so a future cache change cannot silently
        split the audited and served paths."""
        wrapper = cls.__new__(cls)
        wrapper.stats_provider = wrap_with_cache(provider)
        assert wrapper.stats_provider is provider, (
            "wrap_with_cache re-wrapped the replay provider — the audit and "
            "serving feature paths have diverged"
        )
        wrapper.player_metadata = metadata
        wrapper.online_outcome_dists = OnlineT1OutcomeDists(
            wrapper.stats_provider, metadata)
        return wrapper

    # engine contract: extract_features(state) -> whatever predict accepts
    def extract_features(self, state):
        return state

    def _ball_features(self, state) -> np.ndarray:
        striker, bowler = state.current_striker, state.current_bowler
        f = dict(_OUTCOME_DIST_ZERO)
        if self.online_outcome_dists is not None:
            f.update(self.online_outcome_dists.features(
                state, str(striker.player_id), str(bowler.player_id),
                k_player=K_PLAYER, k_venue=K_VENUE))
        else:
            # Explicit snapshot path: reachable only on bare __new__ instances
            # (class default None) — the focused formula tests. __init__
            # always installs the overlay, and OnlineT1OutcomeDists itself
            # fails closed on non-replay providers.
            _fill_outcome_dists(
                f, self.stats_provider, str(striker.player_id),
                str(bowler.player_id), state.venue, state.match_date,
                balls_bowled=state.balls, k_player=K_PLAYER, k_venue=K_VENUE)
        team = state.current_team_idx
        balls = state.balls
        score = float(state.runs[team])
        # Exact parsing_v2.calculate_basic_features formula. In particular,
        # after an opening wide/no-ball score can be positive while balls=0;
        # training uses the 0.1-over floor rather than forcing RR to zero.
        rr = score / max(balls / 6.0, 0.1)
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

    @staticmethod
    def _current_innings_rows(state) -> np.ndarray:
        hist = state.history[:state.history_idx]
        return hist[hist[:, 0] == state.innings] if len(hist) else hist

    def _sequence_inputs(self, state, current: np.ndarray):
        """Return exact causal inputs for the current pre-ball state.

        Every prior feature token must have been observed before its delivery;
        history stores outcomes but cannot reconstruct historical EB features.
        A new or copied state that already has current-innings history is
        therefore unsupported and fails closed instead of using filler data.
        """
        rows = self._current_innings_rows(state)
        n_history = len(rows)
        c = self._cache

        if c["state"] is not state or c["innings"] != state.innings:
            if n_history:
                raise RuntimeError(
                    "T1 cache cannot attach to a new state with existing "
                    "current-innings history; historical pre-ball features "
                    "are unavailable. Start from innings BOS or use a "
                    "feature-bearing replay adapter."
                )
            c = {"state": state, "innings": state.innings,
                 "feats": [], "prev": []}
            self._cache = c

        if len(c["feats"]) != n_history:
            raise RuntimeError(
                "T1 cache/history desynchronization: "
                f"cached_tokens={len(c['feats'])}, "
                f"history_deliveries={n_history}. Refusing filler tokens."
            )
        if n_history >= self.max_seq_len:
            raise RuntimeError(
                f"T1 innings exceeds checkpoint position capacity "
                f"({self.max_seq_len} delivery rows)"
            )

        previous = (
            _outcome_class(int(rows[-1][3]), int(rows[-1][4]))
            if n_history else BOS
        )
        c["feats"].append(current)
        c["prev"].append(previous)
        return c["feats"], c["prev"]

    def _compose_delivery_probs(self, probs: np.ndarray) -> dict[str, float]:
        d = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        if self.extras_graft is not None:
            return graft_extras(d, self.extras_graft)
        # Legacy-style flat 1% + 1% extras graft. NOT byte-identical to
        # production `_build_outcome_dict`, which sets 0.01/0.01 and then
        # renormalizes the whole dict (extras land at ~0.0098): here the
        # six-class block is scaled by 0.98 and extras pinned at exactly
        # 0.01 (~2% relative difference in extras mass).
        for name in CLASS_NAMES:
            d[name] *= 0.98
        d["wide"] = 0.01
        d["no_ball"] = 0.01
        return d

    def _incremental_last_hidden(self, feats, prev):
        """Advance the per-innings prefix cache by one token; return the new
        token's final hidden state.

        Exact-math equivalent of the full causal re-forward's last position:
        under the causal mask, position t attends only to positions <= t at
        every layer, so earlier per-layer key/value projections never change
        once computed. The fused full forward differs only at float epsilon
        (pinned by tests/test_sim_t1_prefix_cache.py). Dropout is inactive
        (eval mode); the encoder is norm_first with ReLU feed-forward and no
        terminal norm, which this mirrors term by term.
        """
        c = self._cache
        n = len(feats)
        if c.get("kv") is None:
            c["kv"] = [{"k": [], "v": []} for _ in self.model.encoder.layers]
            c["kv_len"] = 0
        if c["kv_len"] != n - 1:
            raise RuntimeError(
                f"T1 prefix-cache desynchronization: cached={c['kv_len']}, "
                f"sequence={n}. Refusing to guess the missing tokens.")
        position = n - 1
        prev_token = BOS if self.arm == "no_history" else int(prev[-1])
        token = (
            self.model.feat_proj(
                torch.tensor(feats[-1], device=self.device))
            + self.model.out_emb(
                torch.tensor(prev_token, device=self.device))
            + self.model.pos_emb(
                torch.tensor(position, device=self.device))
        )
        x = token
        for layer, kv in zip(self.model.encoder.layers, c["kv"]):
            attn = layer.self_attn
            h = layer.norm1(x)
            qkv = torch.nn.functional.linear(
                h, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            heads = attn.num_heads
            head_dim = q.shape[-1] // heads
            kv["k"].append(k.view(heads, head_dim))
            kv["v"].append(v.view(heads, head_dim))
            keys = torch.stack(kv["k"], dim=1)      # [H, t, hd]
            values = torch.stack(kv["v"], dim=1)
            scores = (keys @ q.view(heads, head_dim, 1)).squeeze(-1)
            weights = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
            context = (weights.unsqueeze(1) @ values).squeeze(1)  # [H, hd]
            x = x + torch.nn.functional.linear(
                context.reshape(-1), attn.out_proj.weight,
                attn.out_proj.bias)
            hidden = layer.norm2(x)
            x = x + layer.linear2(
                torch.nn.functional.relu(layer.linear1(hidden)))
        c["kv_len"] = n
        return x

    def predict_next_ball(self, state):
        cur = self._ball_features(state)
        if self.arm == "mlp":
            feats, prev = [cur], [BOS]
        else:
            feats, prev = self._sequence_inputs(state, cur)
            if self.prefix_cache_enabled:
                with torch.no_grad():
                    hidden = self._incremental_last_hidden(feats, prev)
                    probs = torch.softmax(
                        self.model.head(hidden), dim=-1).cpu().numpy()
                return self._compose_delivery_probs(probs)
        with torch.no_grad():
            ft = torch.tensor(np.stack(feats)[None, :, :],
                              device=self.device)
            pt = torch.tensor(np.array(prev, dtype=np.int64)[None, :],
                              device=self.device)
            pad = torch.zeros_like(pt, dtype=torch.bool)
            out = self.model(ft, pt, pad)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
        return self._compose_delivery_probs(probs)
