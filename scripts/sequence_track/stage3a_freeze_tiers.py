# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Freeze the stage 3 Block B (3a) training-match lists and step budget.

Night 3 draft § "Block B" (v5). Everything this script writes is frozen
BEFORE any Block B run and before any performance read, so the negative-
transfer screen's training subsets are a registered input rather than
something chosen after seeing a number.

Written under ``experiments/stage3a/``:

* ``target_P_matches.json`` — target P (premium leagues): the TRAIN matches
  whose frame ``competition_tier`` is 3, i.e. the pre-match event-name rule
  ``parsing_v2.classify_match_context`` applies to ``PREMIUM_LEAGUES``;
* ``target_E_matches.json`` — target E (elite): P union the TRAIN matches
  that are internationals (frame ``is_international == 1``) in which BOTH
  teams are ``parsing_v2.ICC_FULL_MEMBERS``. Qualifiers and any match
  involving an associate are excluded by the both-full-members rule;
* ``rowmatch_P_matches.json`` / ``rowmatch_E_matches.json`` — the row-matched
  controls. ALL train matches are shuffled with seed 42 and added in that
  order until the cumulative row count first reaches or exceeds the target's
  row count. The overshoot (at most one match) is recorded;
* ``target_E_validation_matches.json`` — the SAME elite rule evaluated over
  the VALIDATION matches. The training list cannot double as the evaluation
  slice: train and validation matches are disjoint, so a `target_E` slice read
  off the training list would have zero validation rows (code gate finding 1);
* ``big3_validation_matches.json`` — the descriptive ``big3`` readout slice:
  VALIDATION matches inside target E (BOTH teams full members) with at least
  one of India, Australia, England. The "E ∩" restriction is deliberate (code
  gate finding 2): a full member against an associate is not a big-three
  fixture in the sense the slice is meant to read. No family attaches to it;
* ``steps.json`` — the shared optimisation control: one optimiser step is one
  batch of 128 innings (the final partial batch of an epoch is kept), so
  ``E = ceil(n_train_innings / 128)`` and ``S = 30 * E``;
* ``manifest.json`` — per list: match, innings and row counts, the sha256 of
  the file as written, and the train/validation disjointness assertion.

How a match id is resolved to team names (recorded because the frame does not
carry team names). The frame spells an innings as ``<innings number>_<match
id>`` and the match id is a Cricsheet id, so the teams are read from
``data/t20s_json/<match id>.json`` -> ``info.teams``, the same corpus and the
same spelling ``scripts/loaders_common.py`` iterates. Only the matches that
need a team test are opened (the internationals of the train split, plus the
validation split for ``big3``). A match id with no Cricsheet file is reported
and excluded from the team-based sets rather than silently dropped.

Usage:
    uv run --no-sync python scripts/sequence_track/stage3a_freeze_tiers.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import transformer_t1 as t1  # noqa: E402
from parsing_v2 import ICC_FULL_MEMBERS  # noqa: E402

DEFAULT_DATA_DIR = REPO / "data" / "xgb_data_i7"
DEFAULT_CRICSHEET_DIR = REPO / "data" / "t20s_json"
DEFAULT_OUT_DIR = REPO / "experiments" / "stage3a"
# The target-P membership rule, as a frame column value.
PREMIUM_TIER = 3
# The `big3` readout slice (the user's view that these sit a tier above the
# other full members). Descriptive only; no family attaches to it.
BIG3 = ("India", "Australia", "England")
# One optimiser step = one batch of this many innings, matching the trainer's
# loop (`--batch 128`, final partial batch kept).
BATCH_INNINGS = 128
# The registered number of epoch-equivalents the step budget stands for.
BUDGET_EPOCHS = 30
# The shuffle seed of the row-matched controls, frozen here.
ROWMATCH_SEED = 42


def per_match(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match: row count, innings count, tier, international flag.

    The tier and the international flag are match-level properties, so this
    refuses a match where either varies across rows instead of picking one.
    """
    frame = pd.DataFrame({
        "match_id": [t1.match_id_of(value)
                     for value in df["innings_id"].to_numpy()],
        "innings_id": df["innings_id"].to_numpy(),
        "tier": t1.tier_codes(df),
        "is_international": pd.to_numeric(
            df["is_international"], errors="coerce").fillna(0).astype(int)
        if "is_international" in df.columns else 0,
    })
    grouped = frame.groupby("match_id", sort=True)
    out = pd.DataFrame({
        "n_rows": grouped.size(),
        "n_innings": grouped["innings_id"].nunique(),
        "tier": grouped["tier"].max(),
        "tier_min": grouped["tier"].min(),
        "is_international": grouped["is_international"].max(),
        "intl_min": grouped["is_international"].min(),
    }).reset_index()
    varying = out[(out["tier"] != out["tier_min"])
                  | (out["is_international"] != out["intl_min"])]
    if len(varying):
        raise RuntimeError(
            "competition_tier / is_international vary within match(es) "
            f"{varying['match_id'].tolist()[:5]}")
    return out.drop(columns=["tier_min", "intl_min"])


def teams_of(match_id: str, cricsheet_dir: Path) -> set[str] | None:
    """``info.teams`` of one Cricsheet match, or None if the file is absent."""
    path = Path(cricsheet_dir) / f"{match_id}.json"
    if not path.is_file():
        return None
    try:
        info = json.loads(path.read_text()).get("info") or {}
    except (json.JSONDecodeError, OSError):
        return None
    teams = info.get("teams")
    return set(teams) if isinstance(teams, list) else None


def both_full_members(teams: set[str] | None) -> bool:
    return bool(teams) and len(teams) == 2 and teams <= set(ICC_FULL_MEMBERS)


def rowmatch(matches: pd.DataFrame, target_rows: int,
             seed: int = ROWMATCH_SEED) -> tuple[list[str], int]:
    """Whole train matches shuffled at `seed`, added until rows >= target.

    Deterministic given the pinned train parquet: the match ids are sorted
    before the shuffle, so the permutation does not depend on parquet order.
    Returns ``(match ids in selection order, overshoot in rows)``.
    """
    ids = matches["match_id"].to_numpy()
    rows = matches.set_index("match_id")["n_rows"]
    order = np.random.default_rng(seed).permutation(len(ids))
    picked: list[str] = []
    total = 0
    for position in order:
        match_id = str(ids[position])
        picked.append(match_id)
        total += int(rows[match_id])
        if total >= target_rows:
            break
    return picked, total - target_rows


def write_list(path: Path, match_ids, definition: str, **extra) -> dict:
    """Write one frozen match list and return its manifest entry.

    `match_ids` is written SORTED: the trainer turns the list into a set, so
    order carries no meaning, and a stable order keeps the file's sha256 — the
    digest that enters the training signature — a function of the membership
    alone.
    """
    ids = sorted({str(value) for value in match_ids})
    payload = {"definition": definition, "n_matches": len(ids),
               "match_ids": ids, **extra}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"file": path.name, "n_matches": len(ids),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k: v for k, v in extra.items() if k != "match_ids"}}


def counts_for(matches: pd.DataFrame, ids) -> dict:
    selected = matches[matches["match_id"].isin(set(map(str, ids)))]
    return {"n_matches": int(len(selected)),
            "n_innings": int(selected["n_innings"].sum()),
            "n_rows": int(selected["n_rows"].sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--frame-version", default=None)
    ap.add_argument("--cricsheet-dir", type=Path,
                    default=DEFAULT_CRICSHEET_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch", type=int, default=BATCH_INNINGS)
    ap.add_argument("--epochs", type=int, default=BUDGET_EPOCHS)
    ap.add_argument("--seed", type=int, default=ROWMATCH_SEED)
    args = ap.parse_args(argv)

    version = t1.resolve_frame_version(args.data_dir, args.frame_version)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def read(split: str) -> tuple[pd.DataFrame, Path]:
        path = t1.split_path(args.data_dir, version, split)
        cols = ["innings_id", t1.TIER_COL, "is_international"]
        return pd.read_parquet(path, columns=cols), path

    train, train_path = read("train")
    val, val_path = read("validation")
    train_matches = per_match(train)
    val_matches = per_match(val)
    n_train_innings = int(train_matches["n_innings"].sum())
    print(f"train: {len(train)} rows, {n_train_innings} innings, "
          f"{len(train_matches)} matches", flush=True)

    # --- targets ---------------------------------------------------------
    target_p = sorted(train_matches.loc[train_matches["tier"] == PREMIUM_TIER,
                                        "match_id"].astype(str))
    intl = sorted(train_matches.loc[train_matches["is_international"] == 1,
                                    "match_id"].astype(str))
    elite_only, missing = [], []
    for match_id in intl:
        teams = teams_of(match_id, args.cricsheet_dir)
        if teams is None:
            missing.append(match_id)
            continue
        if both_full_members(teams):
            elite_only.append(match_id)
    target_e = sorted(set(target_p) | set(elite_only))
    if missing:
        print(f"WARNING: {len(missing)} train internationals have no "
              f"cricsheet file under {args.cricsheet_dir} and are excluded "
              f"from target E: {missing[:5]}", flush=True)

    p_rows = counts_for(train_matches, target_p)["n_rows"]
    e_rows = counts_for(train_matches, target_e)["n_rows"]
    rowmatch_p, overshoot_p = rowmatch(train_matches, p_rows, args.seed)
    rowmatch_e, overshoot_e = rowmatch(train_matches, e_rows, args.seed)

    # --- validation-side slices ------------------------------------------
    # Target E's rule, applied to the VALIDATION matches: this is the slice
    # the `target_E` family is read on. Tier 3 membership comes from the frame
    # column, the elite-international part from the team test, exactly as on
    # the training side.
    val_tier3 = set(val_matches.loc[val_matches["tier"] == PREMIUM_TIER,
                                    "match_id"].astype(str))
    val_intl = set(val_matches.loc[val_matches["is_international"] == 1,
                                   "match_id"].astype(str))
    target_e_val, big3, big3_missing = [], [], []
    for match_id in val_matches["match_id"].astype(str):
        teams = teams_of(match_id, args.cricsheet_dir)
        if teams is None:
            big3_missing.append(match_id)
            continue
        elite_intl = match_id in val_intl and both_full_members(teams)
        if match_id in val_tier3 or elite_intl:
            target_e_val.append(match_id)
        # big3 = E-intersected: both teams full members AND one of the three.
        if both_full_members(teams) and (teams & set(BIG3)):
            big3.append(match_id)

    # --- step budget ------------------------------------------------------
    steps_per_epoch = math.ceil(n_train_innings / args.batch)
    total_steps = args.epochs * steps_per_epoch
    steps = {
        "rule": ("one optimiser step = one batch of --batch innings, final "
                 "partial batch of an epoch kept; "
                 "E = ceil(n_train_innings / batch), S = epochs * E"),
        "n_train_innings": n_train_innings,
        "batch": int(args.batch),
        "epochs": int(args.epochs),
        "eval_every": int(steps_per_epoch),
        "max_steps": int(total_steps),
    }
    (out_dir / "steps.json").write_text(
        json.dumps(steps, indent=2, sort_keys=True) + "\n")

    # --- write the lists --------------------------------------------------
    entries = [
        write_list(out_dir / "target_P_matches.json", target_p,
                   "train matches with frame competition_tier == 3 "
                   "(parsing_v2.PREMIUM_LEAGUES)"),
        write_list(out_dir / "target_E_matches.json", target_e,
                   "target P union train internationals "
                   "(is_international == 1) where BOTH cricsheet info.teams "
                   "are in parsing_v2.ICC_FULL_MEMBERS",
                   n_from_tier3=len(target_p), n_from_elite_intl=len(
                       set(elite_only) - set(target_p)),
                   n_internationals_without_cricsheet_file=len(missing)),
        write_list(out_dir / "rowmatch_P_matches.json", rowmatch_p,
                   f"seed-{args.seed} shuffle of ALL train matches, added in "
                   "order until cumulative rows first >= target P rows",
                   target="target_P_matches.json", target_rows=int(p_rows),
                   overshoot_rows=int(overshoot_p)),
        write_list(out_dir / "rowmatch_E_matches.json", rowmatch_e,
                   f"seed-{args.seed} shuffle of ALL train matches, added in "
                   "order until cumulative rows first >= target E rows",
                   target="target_E_matches.json", target_rows=int(e_rows),
                   overshoot_rows=int(overshoot_e)),
        write_list(out_dir / "target_E_validation_matches.json",
                   target_e_val,
                   "the target E rule over VALIDATION matches: frame "
                   "competition_tier == 3, or is_international == 1 with BOTH "
                   "cricsheet info.teams in parsing_v2.ICC_FULL_MEMBERS. This "
                   "is the slice the target E family is READ on; the training "
                   "list is disjoint from it by construction",
                   split="validation",
                   n_from_tier3=len(val_tier3),
                   n_validation_matches_without_cricsheet_file=len(
                       big3_missing)),
        write_list(out_dir / "big3_validation_matches.json", big3,
                   "VALIDATION matches inside target E (BOTH cricsheet "
                   "info.teams in parsing_v2.ICC_FULL_MEMBERS) with at least "
                   "one of India, Australia, England; descriptive readout "
                   "slice, no family",
                   split="validation",
                   rule="E-intersected: both full members AND one of "
                        "India/Australia/England",
                   n_validation_matches_without_cricsheet_file=len(
                       big3_missing)),
    ]

    # --- manifest ---------------------------------------------------------
    train_ids = set(train_matches["match_id"].astype(str))
    val_ids = set(val_matches["match_id"].astype(str))
    lists = {"target_P": target_p, "target_E": target_e,
             "rowmatch_P": rowmatch_p, "rowmatch_E": rowmatch_e}
    disjoint = {}
    for name, ids in lists.items():
        leaked = sorted(set(map(str, ids)) & val_ids)
        disjoint[name] = {"n_in_validation": len(leaked),
                          "examples": leaked[:5]}
    if any(row["n_in_validation"] for row in disjoint.values()):
        raise RuntimeError(
            "a frozen TRAINING match list contains match ids that also "
            f"appear in the validation parquet: {disjoint}")
    manifest = {
        "generator": "scripts/sequence_track/stage3a_freeze_tiers.py",
        "frame": {"dir": Path(args.data_dir).as_posix(), "version": version,
                  "train_parquet_md5": t1.md5_file(train_path),
                  "validation_parquet_md5": t1.md5_file(val_path)},
        "cricsheet_dir": Path(args.cricsheet_dir).as_posix(),
        # Reviewer MUST-FIX 1 (night 3): the team-name set the E and big3
        # rules were actually evaluated against, written out in full rather
        # than named. `parsing_v2.ICC_FULL_MEMBERS` can be edited; a rule that
        # only NAMES it cannot be re-checked, and a reader cannot tell which
        # spellings counted. Cricsheet `info.teams` names are compared
        # VERBATIM against this set — no alias, normalisation or case folding
        # is applied anywhere in this script — so `aliases_applied` is empty
        # by construction and is recorded as such rather than left to
        # inference.
        "team_name_rules": {
            "source": "parsing_v2.ICC_FULL_MEMBERS",
            "icc_full_members": sorted(set(ICC_FULL_MEMBERS)),
            "n_icc_full_members": len(set(ICC_FULL_MEMBERS)),
            "big3": sorted(BIG3),
            "aliases_applied": {},
            "matching_rule": (
                "cricsheet data/t20s_json/<match id>.json -> info.teams, "
                "compared verbatim (exact string equality, no alias table, "
                "no normalisation, no case folding) against "
                "icc_full_members; a match is elite when it has exactly two "
                "teams and both are in that set"),
        },
        "match_id_to_teams": (
            "innings_id is '<innings number>_<cricsheet match id>'; teams are "
            "data/t20s_json/<match id>.json -> info.teams"),
        "steps": steps,
        "rowmatch_seed": int(args.seed),
        "train": {"n_rows": int(train_matches["n_rows"].sum()),
                  "n_innings": n_train_innings,
                  "n_matches": len(train_ids)},
        "validation": {"n_rows": int(val_matches["n_rows"].sum()),
                       "n_innings": int(val_matches["n_innings"].sum()),
                       "n_matches": len(val_ids)},
        "lists": {},
        "train_validation_disjoint": disjoint,
        "assertions": [
            "no match id in any frozen TRAINING list appears in the "
            "validation parquet",
            "competition_tier and is_international are constant within a "
            "match",
        ],
    }
    for entry, (name, ids) in zip(entries[:4], lists.items()):
        manifest["lists"][name] = {**entry, **counts_for(train_matches, ids)}
    manifest["lists"]["target_E_validation"] = {
        **entries[4], **counts_for(val_matches, target_e_val)}
    manifest["lists"]["big3_validation"] = {
        **entries[5], **counts_for(val_matches, big3)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for name, row in manifest["lists"].items():
        print(f"{name}: {row['n_matches']} matches, {row['n_innings']} "
              f"innings, {row['n_rows']} rows, sha256 {row['sha256'][:12]}",
              flush=True)
    print(f"steps: S={steps['max_steps']} E={steps['eval_every']}",
          flush=True)
    print(f"wrote {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
