"""B5 unit check — teacher-forced replay fidelity + continuation smoke.

Part 1 (no model, fast): replay innings 1 for EVERY in-scope test match and
hard-assert parity vs an independent aggregation of the raw JSON (final
runs / legal balls / wickets exact; every checkpoint snapshot at exactly
6*cp legal balls). prepare_match() raises on any parity violation, so a
pass here proves the replay is total-preserving on the whole eval corpus.

Part 2 (verification-only peek): at each checkpoint, the snapshot's crease
pair {striker, non_striker} must equal the ACTUAL pair on the first
delivery of the next over. Peeking at over cp is legal here — it verifies
the state, it never feeds a quote. Retired-hurt oddities and stolen-run
rotations on extras can produce rare mismatches, so this is a >=95%
soft-assert (hard-fail below), with mismatches printed.

Part 3 (live model path): for the first in-scope match, 3 seeded
continuations from the over-6 state through the REAL default sim path
(venue-ON autoload + v1 vector calibrator + EmpiricalBowlerSelector);
asserts remaining runs are sane, the innings terminates legally, and the
same seed reproduces the same remaining total byte-for-byte.

Run: uv run python scripts/auto/b5_unit_check.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from b5_inplay_quotes import (  # noqa: E402
    CHECKPOINTS,
    ReplayError,
    prepare_match,
    quote_checkpoint,
)

TEST_DIR = REPO / "data" / "polymarket_test"


def main():
    from sim_eval.loaders import TestMatchLoader  # noqa: E402
    loader = TestMatchLoader()
    files = sorted(TEST_DIR.glob("*.json"))
    assert files, f"no test files under {TEST_DIR}"

    # ---- Part 1: full-corpus replay parity -------------------------------
    ok, skips = 0, []
    pair_total, pair_match, pair_mismatches = 0, 0, []
    for fp in files:
        try:
            match_id, snaps, meta = prepare_match(fp, loader)
        except ReplayError as e:
            skips.append((fp.name, str(e)))
            continue
        ok += 1

        # ---- Part 2: crease-pair verification at each checkpoint ---------
        with open(fp) as f:
            data = json.load(f)
        inn1 = data["innings"][0]
        overs_by_num = {ov["over"]: ov for ov in inn1.get("overs", [])}
        for cp, snap in snaps.items():
            nxt = overs_by_num.get(cp)
            if not nxt or not nxt.get("deliveries"):
                continue
            d0 = nxt["deliveries"][0]
            actual_pair = {d0["batter"], d0["non_striker"]}
            lineup = snap.batting_lineup.players
            snap_pair = {lineup[snap.striker_idx].name,
                         lineup[snap.non_striker_idx].name}
            pair_total += 1
            if snap_pair == actual_pair:
                pair_match += 1
            else:
                pair_mismatches.append(
                    (fp.name, cp, sorted(snap_pair), sorted(actual_pair)))

    print(f"Part 1: replay parity PASS on {ok}/{len(files)} matches "
          f"({len(skips)} out of scope)")
    reasons = {}
    for _, r in skips:
        key = r.split("(")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1
    for r, n in sorted(reasons.items()):
        print(f"  skip reason x{n}: {r}")
    assert ok > 0, "no matches survived scope filters"

    rate = pair_match / max(pair_total, 1)
    print(f"Part 2: crease-pair match {pair_match}/{pair_total} "
          f"({100*rate:.1f}%)")
    for m in pair_mismatches[:10]:
        print(f"  mismatch: {m}")
    assert rate >= 0.95, f"crease-pair match rate {rate:.3f} < 0.95"

    # ---- Part 3: live model-path smoke -----------------------------------
    import joblib
    from sim_v1_2 import (EmpiricalBowlerSelector, SimulationEngine,
                          T20Rules, XGBoostModelV2)
    from stats_provider import StatsProvider
    from player_metadata import PlayerMetadataProvider

    print("Part 3: live-path smoke (first in-scope match, cp=6, 3 sims x2)")
    stats_provider = StatsProvider("models", version="v3")
    player_metadata = PlayerMetadataProvider("data/all_players_enriched.csv")
    model = XGBoostModelV2(
        model_path="models/xgb_v3/xgboost_model_v3.pkl",
        batter_encoder_path="models/xgb_v3/batter_encoder_v3.pkl",
        bowler_encoder_path="models/xgb_v3/bowler_encoder_v3.pkl",
        feature_columns_path="models/xgb_v3/feature_columns_v3.txt",
        stats_provider=stats_provider,
        player_metadata=player_metadata,
        ball_calibrator=joblib.load(
            "models/xgb_v3/vector_scaling_calibrator_v1.pkl"),
    )
    assert model.venue_encoder is not None, \
        "venue encoder sidecar failed to autoload — not the default path"
    rules = T20Rules(EmpiricalBowlerSelector())
    engine = SimulationEngine(model, rules)

    snaps = meta = None
    for fp in files:
        try:
            match_id, snaps, meta = prepare_match(fp, loader)
        except ReplayError:
            continue
        if 6 in snaps and meta["legal"] > 36:
            break
    assert snaps is not None and 6 in snaps

    cp_state = snaps[6]
    rem_a = quote_checkpoint(cp_state, engine, rules, n_sims=3, seed=43)
    rem_b = quote_checkpoint(cp_state, engine, rules, n_sims=3, seed=43)
    print(f"  {match_id}: remaining draws {rem_a.tolist()} "
          f"(repeat {rem_b.tolist()})")
    assert (rem_a == rem_b).all(), "same seeds must reproduce exactly"
    assert (rem_a >= 0).all() and (rem_a <= 400).all(), "insane remaining"
    # cp_state must be untouched by the continuations (copy semantics).
    assert int(cp_state.balls) == 36, "checkpoint state mutated by sims"

    print("\nB5 unit check: ALL PASS")


if __name__ == "__main__":
    main()
