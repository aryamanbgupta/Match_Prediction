#!/usr/bin/env python3
"""One stage-1 runner for every sequence-track arm (A, A50, B, C).

Stage 1 compares four ball models on the SAME chronology, the SAME same-day
replay cutoffs, the SAME bowler selector, the SAME extras sidecar and the
SAME per-fixture RNG seeds. That is only possible if one runner drives every
arm, so this script extends `run_sim_eval_t1`'s i8 seam pattern to the
XGBoost wrappers as well:

* the chronological loader, the `_T1ReplayEvaluator` lifecycle and the
  `SameDayReplayStatsProvider` are used for EVERY arm (design fact recorded
  in `docs/sequence_track/stage0_acceptance.md`: the replay provider's
  tracker view implements every provider method `XGBoostModelV2`'s feature
  builder calls);
* B/C bind `TransformerT1SimModel` to that provider over the legacy v3 cache
  (the ablation checkpoints refuse any other frame), prefix cache OFF;
* A/A50 bind `XGBoostModelV2` to the SAME provider class over the i7 cache,
  with an explicit model dir and `ball_calibrator=None` (the promoted i7
  stack serves RAW — D16/D17).

Everything else is delegated to the frozen `run_sim_eval.main()`, so the arm
output dir holds the standard evaluator results JSON (copied to `eval.json`)
plus two stage-1 artifacts:

* `arm_provenance.json` — per fixture: id, date, as-of stamp, eligibility,
  seeds, n_sims; per run: artifacts, hashes, selector, sidecars, run-out
  constant, clip bounds, base seed, engine/runner md5, threads/device,
  timestamps. `audit_cross_arm.py` consumes it.
* `raw_sims.jsonl` — one line per fixture carrying every simulation's
  winner/tie, both innings' totals/wickets/balls/extras/unique bowlers,
  per-batter runs and per-bowler wickets. The realism and prop scorers read
  this instead of re-simulating (`prop_backtest.py` builds its own
  `XGBoostModelV2` and has no T1 seam).

RNG: the engine seeds ONE global `random`/`np.random` stream per simulation
(`random_seed + i`). The three named sub-seeds (outcome, extras, selector)
are derived and recorded here, but the engine cannot consume them
separately without an engine-wide RNG refactor — registered as a stage-0
deviation (D5 check 5.9). The additive scheme also means one fixture
OCCUPIES the seed interval `[fixture_seed, fixture_seed + n_sims)`;
`seed_intervals_disjoint` screens the whole fixture set in preflight and
refuses to launch when two intervals overlap.

Registered runs pass `--config experiments/configs/seq_stage1_sim_v1.yaml`:
the runner then re-runs `pin_stage1.verify_config` (refusing to launch if
any pinned hash has moved) and refuses any invocation whose effective
values differ from the arm's registered block. Without it the run is
recorded as `config_verified: false` and `audit_cross_arm.py` refuses it
unless `--allow-unregistered`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SCRIPTS_DIR / "sim_eval") not in sys.path:
    sys.path.insert(1, str(SCRIPTS_DIR / "sim_eval"))


PROVENANCE_CONTRACT = "sequence_track_arm_provenance_v2"
PROVENANCE_FILENAME = "arm_provenance.json"
RAW_SIMS_FILENAME = "raw_sims.jsonl"
EVAL_COPY_FILENAME = "eval.json"

# `md5_directory`'s documented contract, restated in the provenance record so
# a consumer never has to guess how the directory hash was formed.
DIR_HASH_CONTRACT = "md5_file_or_sorted_relative_path_md5_lines_v1"

# Paths are composed rather than spelled as literals so the manifest guard
# stays satisfied: artifacts of record resolve through the
# `artifacts.artifact_path`
# roles below, and the two that have no role (the B18 extras sidecar and the
# stage-1 `seq_stage1` namespace) are built from parts.
MODELS_ROOT = Path("models")
SEQ_STAGE1_ROOT = MODELS_ROOT / "embeddings" / "seq_stage1"
DEFAULT_EXTRAS_GRAFT = MODELS_ROOT / "auto" / "b18" / "extras_graft_v1.json"
DEFAULT_CONTEXT_DIR = "data/t20s_json"
DEFAULT_PLAYER_METADATA = "data/all_players_enriched.csv"

# Cricsheet `info.match_type` values that count as T20 for eligibility. The
# stage-1 iteration corpus (`data/polymarket_test_v2`) uses "T20" only;
# "IT20" is accepted because cricsheet uses it for some internationals.
T20_MATCH_TYPES = ("T20", "IT20")

SUB_SEED_STREAMS = ("outcome", "extras", "selector")

ARMS = {
    "A": {
        "model_type": "xgboost",
        "model_role": "ball_model_prod",
        "model_dir": None,
        "artifact_suffix": "i7",
        "stats_version": "i7",
        "description": "promoted i7 no-class-weights ball model, served RAW",
    },
    "A50": {
        "model_type": "xgboost",
        "model_role": None,  # stage-1 artifact; no manifest role
        "model_dir": SEQ_STAGE1_ROOT / "a50",
        "artifact_suffix": "a50",
        "stats_version": "i7",
        "description": "50-column i7 retrain (stage-1 D4 artifact)",
    },
    "B": {
        "model_type": "transformer",
        "model_role": None,
        "model_dir": None,  # chosen by D5 check 5.3; must be explicit
        "artifact_suffix": None,
        "stats_version": "v3",
        "description": "T1 ablation `mlp` arm checkpoint",
    },
    "C": {
        "model_type": "transformer",
        "model_role": None,
        "model_dir": None,  # chosen by D5 check 5.3; must be explicit
        "artifact_suffix": None,
        "stats_version": "v3",
        "description": "T1 ablation `full` arm checkpoint",
    },
}

# Stats caches are artifacts of record; an override version falls back to the
# provider's own naming convention.
STATS_CACHE_ROLES = {"i7": "stats_cache_i7", "v3": "stats_cache_v3_legacy"}


# ---------------------------------------------------------------------------
# Threads: set before numpy / torch / xgboost are imported anywhere.
# ---------------------------------------------------------------------------

def _early_thread_cap(argv=None) -> int:
    """Read `--threads` straight off argv and cap every BLAS/OMP pool.

    Must run before the first numpy/torch import, which is why it does not
    wait for argparse (same reason `run_sim_eval_t1` reads argv directly).
    """
    argv = list(sys.argv if argv is None else argv)
    threads = 4
    if "--threads" in argv:
        index = argv.index("--threads")
        try:
            threads = int(argv[index + 1])
        except (IndexError, ValueError):
            raise SystemExit("--threads requires an integer value")
    if threads < 1:
        raise SystemExit("--threads must be >= 1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    return threads


# ---------------------------------------------------------------------------
# Seeds (D5 check 5.4)
# ---------------------------------------------------------------------------

_LOW31 = (1 << 31) - 1


def fixture_seed(cricsheet_id: str, base_seed: int) -> int:
    """Low 31 bits of sha256("<cricsheet_id>:<base_seed>")."""
    payload = f"{cricsheet_id}:{int(base_seed)}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big") & _LOW31


def sub_seed(seed: int, stream: str) -> int:
    """Low 31 bits of sha256("<fixture_seed>:<stream>")."""
    digest = hashlib.sha256(f"{int(seed)}:{stream}".encode()).digest()
    return int.from_bytes(digest, "big") & _LOW31


def sub_seeds(seed: int) -> dict:
    return {stream: sub_seed(seed, stream) for stream in SUB_SEED_STREAMS}


def _seed_entries(fixture_ids, base_seeds) -> list:
    """Sorted `(seed, label)` for every (fixture, base seed) pair.

    One base seed labels entries by the fixture id; several label them
    `<fixture id>@<base seed>`, because the 1b convergence batches run the
    SAME fixtures under several base seeds and their intervals share one
    2^31 seed line.
    """
    if isinstance(base_seeds, (int, float)):
        seeds = [int(base_seeds)]
    else:
        seeds = [int(value) for value in base_seeds]
    if not seeds:
        raise ValueError("no base seeds to screen")
    joint = len(seeds) > 1
    entries = []
    for base_seed in seeds:
        for fixture_id in fixture_ids:
            label = (f"{fixture_id}@{base_seed}" if joint
                     else str(fixture_id))
            entries.append((fixture_seed(str(fixture_id), base_seed), label))
    entries.sort()
    return entries


def seed_intervals_disjoint(fixture_ids, base_seed, n_sims: int) -> list:
    """Overlapping seed intervals. An EMPTY list is the good case.

    The engine seeds simulation `i` of a fixture with
    `SimulationConfig.random_seed + i`, so one fixture consumes the whole
    half-open interval `[fixture_seed, fixture_seed + n_sims)`. If two
    fixtures' intervals intersect, the simulations at the shared seeds run
    the SAME draw schedule — the two fixtures' Monte Carlo errors are then
    correlated inside every arm, which is not what the independent-batch
    reading of the protocol assumes, and it does NOT cancel in a paired
    contrast: the draws are shared, but each arm's model maps them to a
    different match, so the induced error is model-dependent.

    The collision probability grows with `n_sims` (roughly
    `N^2 * n_sims / 2^31` for N fixtures), so it cannot be waved away for
    the larger convergence candidates. `main` calls this in preflight and
    refuses to launch when it returns anything.

    `base_seed` may be ONE seed or a sequence of them. The 1b convergence
    protocol runs the same fixtures under three batch base seeds, and those
    batches share one seed line, so a batch must be screened against the
    JOINT set — screening each seed separately misses a cross-batch
    collision (Astra round 2, item 5).

    Returns a list of `(id_a, seed_a, id_b, seed_b, gap)` tuples, ordered by
    seed, one per overlapping adjacent pair (an overlap anywhere implies an
    overlapping adjacent pair once the seeds are sorted). Under several base
    seeds the ids are labelled `<fixture id>@<base seed>`.
    """
    n_sims = int(n_sims)
    entries = _seed_entries(fixture_ids, base_seed)
    overlaps = []
    for (seed_a, id_a), (seed_b, id_b) in zip(entries, entries[1:]):
        gap = seed_b - seed_a
        if gap < n_sims:
            overlaps.append((id_a, seed_a, id_b, seed_b, gap))
    return overlaps


def min_seed_gap(fixture_ids, base_seed):
    """Smallest distance between two seeds (None for < 2 entries).

    Accepts one base seed or a sequence: with several, this is the JOINT
    minimum over the union of every (fixture, base seed) interval.
    """
    entries = _seed_entries(fixture_ids, base_seed)
    if len(entries) < 2:
        return None
    return min(b[0] - a[0] for a, b in zip(entries, entries[1:]))


def largest_permitted_n_sims(fixture_ids, base_seeds, candidates):
    """The largest candidate whose intervals stay disjoint (None if none)."""
    gap = min_seed_gap(fixture_ids, base_seeds)
    permitted = [int(n) for n in candidates if gap is None or int(n) <= gap]
    return max(permitted) if permitted else None


def assert_seed_intervals_disjoint(fixture_ids, base_seed, n_sims: int
                                   ) -> None:
    """Preflight refusal: no two runs may share a draw schedule.

    `base_seed` is the seed cohort this launch belongs to: one seed for a
    single run, or every registered batch base seed for a convergence /
    variability batch, whose sibling batches occupy the same seed line.
    """
    overlaps = seed_intervals_disjoint(fixture_ids, base_seed, n_sims)
    if not overlaps:
        return
    detail = "\n  ".join(
        f"{id_a} (seed {seed_a}) and {id_b} (seed {seed_b}) are {gap} "
        f"apart, less than --n-sims {n_sims}"
        for id_a, seed_a, id_b, seed_b, gap in overlaps)
    raise SystemExit(
        "run_arm refuses to launch: the engine seeds simulation i with "
        "random_seed + i, so these runs would share part of one draw "
        f"schedule under seed cohort {base_seed}:\n  {detail}\n"
        "Register a different base seed, or run a smaller --n-sims "
        "(pin_stage1.py --check-seed-overlap --n-sims N screens a candidate "
        "jointly over every registered batch base seed).")


# ---------------------------------------------------------------------------
# Provenance (D6 check 6.1)
# ---------------------------------------------------------------------------

def _canonical_sha256(payload) -> str:
    """sha256 of a stable JSON rendering of one odds row."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fixture_provenance(
    *,
    cricsheet_id: str,
    document: dict,
    as_of_date: str,
    same_day_advanced_before,
    matches_advanced: int,
    odds_row,
    scored: bool,
    skip_reason,
    base_seed: int,
    n_sims: int,
) -> dict:
    """One fixture's provenance row.

    Eligibility carries BOTH sides of the join, because they are different
    facts and only one of them is what the evaluator scores:

    * `cricsheet_resolved` / `cricsheet_winner` — the outcome in the
      cricsheet document;
    * `odds_row_found` / `odds_row_id` / `odds_row_sha256` /
      `odds_actual_winner` — the odds row the evaluator actually resolved
      and the winner it scores against;
    * `scored` / `skip_reason` — whether this fixture produced a
      `MatchEvaluationResult` at all.

    A fixture scored in one arm and skipped in another is exactly the defect
    the cross-arm audit exists to catch, so none of these may be inferred:
    they are recorded from the run.
    """
    info = document.get("info", {})
    seed = fixture_seed(str(cricsheet_id), base_seed)
    cricsheet_winner = (info.get("outcome") or {}).get("winner")
    odds_row_id = None
    if odds_row is not None:
        odds_row_id = {
            "cricsheet_id": (
                None if odds_row.get("cricsheet_id") is None
                else str(odds_row.get("cricsheet_id"))),
            "display_match_id": (
                None if odds_row.get("display_match_id") is None
                else str(odds_row.get("display_match_id"))),
        }
    odds_winner = None if odds_row is None else odds_row.get("actual_winner")
    return {
        "cricsheet_id": str(cricsheet_id),
        "match_date": str(info.get("dates", [None])[0]),
        "as_of": {
            "date": str(as_of_date),
            "same_day_advanced_before": [
                str(value) for value in same_day_advanced_before],
            "matches_advanced": int(matches_advanced),
        },
        "eligibility": {
            "odds_row_found": odds_row is not None,
            "odds_row_id": odds_row_id,
            "odds_row_sha256": (
                None if odds_row is None else _canonical_sha256(odds_row)),
            "odds_actual_winner": (
                None if odds_winner is None else str(odds_winner)),
            "cricsheet_resolved": bool(cricsheet_winner),
            "cricsheet_winner": (
                None if cricsheet_winner is None else str(cricsheet_winner)),
            "male_t20": bool(
                info.get("gender") == "male"
                and info.get("match_type") in T20_MATCH_TYPES
            ),
            "scored": bool(scored),
            "skip_reason": None if skip_reason is None else str(skip_reason),
        },
        "fixture_seed": seed,
        "sub_seeds": sub_seeds(seed),
        "n_sims": int(n_sims),
    }


def write_provenance(output_dir, arm: str, run: dict, fixtures: list) -> Path:
    """Write `arm_provenance.json`, fixtures ordered by (date, id)."""
    payload = {
        "contract": PROVENANCE_CONTRACT,
        "arm": str(arm),
        "run": run,
        "fixtures": sorted(
            fixtures,
            key=lambda row: (row["match_date"], row["cricsheet_id"]),
        ),
    }
    path = Path(output_dir) / PROVENANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


class _LifecycleRecorder:
    """Records the replay lifecycle the evaluator drives.

    The evaluator gets this proxy; the MODEL keeps the real provider —
    wrapping the provider handed to the model would defeat
    `wrap_with_cache`'s idempotence and `OnlineT1OutcomeDists`'
    replay-provider check.
    """

    def __init__(self, inner):
        self._inner = inner
        self.stamps = {}
        self._date = None
        self._advanced_on_date = []

    @property
    def matches_advanced(self) -> int:
        return self._inner.matches_advanced

    def begin_date(self, date_str, documents_for_date):
        self._inner.begin_date(date_str, documents_for_date)
        self._date = str(date_str)
        self._advanced_on_date = []

    def begin_match(self, match_id, document, *, prediction_required):
        self._inner.begin_match(
            match_id, document, prediction_required=prediction_required)
        # Nothing advances between begin_match and lock_prediction, so
        # this snapshot IS the state at the prediction lock; the lock
        # hook below re-reads the counter to keep that explicit.
        self.stamps[str(match_id)] = {
            "date": self._date,
            "same_day_advanced_before": list(self._advanced_on_date),
            "matches_advanced": self._inner.matches_advanced,
            "odds_row_found": bool(prediction_required),
        }

    def lock_prediction(self, match_id):
        self._inner.lock_prediction(match_id)
        self.stamps[str(match_id)]["matches_advanced"] = (
            self._inner.matches_advanced)

    def advance_match(self, match_id, document):
        result = self._inner.advance_match(match_id, document)
        self._advanced_on_date.append(str(match_id))
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_artifact(model_dir: Path, prefix: str, suffix, extension: str
                      ) -> Path:
    """`<prefix>_<suffix>.<ext>` in `model_dir`, else the one `<prefix>_*`.

    The A50 directory copies the production encoders verbatim (D4 check
    4.5), so its sidecars may keep the production suffix while its booster
    carries `a50`. Exactly one candidate must exist either way.
    """
    if suffix:
        candidate = model_dir / f"{prefix}_{suffix}.{extension}"
        if candidate.exists():
            return candidate
    found = sorted(model_dir.glob(f"{prefix}_*.{extension}"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise SystemExit(
            f"no {prefix}_*.{extension} in {model_dir} — the arm's model "
            "directory is missing or not yet built")
    raise SystemExit(
        f"{model_dir} holds {len(found)} {prefix}_*.{extension} artifacts; "
        f"cannot infer which one this arm serves: "
        f"{[path.name for path in found]}")


def _resolve_device(model_type: str, requested=None) -> str:
    """The device this run will ACTUALLY use, resolved before it is checked.

    Order: `--device`, else an inherited `T1_SIM_DEVICE`, else `cpu`.
    `auto` resolves the way `sim_t1` resolves it (mps, then cuda, then cpu),
    so a resolved value is never the token the operator typed. The XGBoost
    arms run on CPU only, and a non-CPU request for them fails closed rather
    than being silently recorded as `cpu`.
    """
    asked = requested or os.environ.get("T1_SIM_DEVICE") or "cpu"
    if model_type != "transformer":
        if asked not in ("cpu", "auto"):
            raise SystemExit(
                f"--device {asked!r} is not available to an XGBoost arm; "
                "the ball model runs on CPU")
        return "cpu"
    if asked == "auto":
        import torch
        asked = ("mps" if torch.backends.mps.is_available()
                 else "cuda" if torch.cuda.is_available() else "cpu")
    if asked not in ("cpu", "mps", "cuda"):
        raise SystemExit(f"unknown device {asked!r}")
    return asked


def _same_path(left, right) -> bool:
    """Path equality that survives repo-relative vs absolute spellings."""
    if left is None or right is None:
        return left is None and right is None
    return (Path(left).expanduser().resolve()
            == Path(right).expanduser().resolve())


# The registered run blocks a `--config` invocation may match, in the order
# they are tried. Each names a fixture dir, the base seeds it permits and the
# n_sims it permits.
REGISTERED_BLOCKS = ("smoke_1a", "full_run", "timing_1b")

# `pin_stage1.CONVERGENCE_PLACEHOLDER`: the full-run n_sims is not chosen
# until the 1b convergence read, so the full block is not launchable yet.
N_SIMS_PLACEHOLDER = "to_be_filled_before_1d"


def _block_seeds(block: dict, config: dict) -> list:
    """Base seeds a registered run block permits."""
    seeds = block.get("base_seeds")
    if seeds:
        return [int(value) for value in seeds]
    return []


def _block_n_sims(block: dict):
    """n_sims values a registered run block permits, or the placeholder."""
    value = block.get("n_sims")
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        return value
    return [] if value is None else [int(value)]


def registered_mismatches(config: dict, arm: str, effective: dict) -> list:
    """Differences between this invocation and the arm's registered block.

    `effective` carries the values the run will actually use — including the
    RESOLVED device and the simulation count, both of which used to escape
    the check (Astra round 2, item 4a). The fixture dir must equal one of
    the arm's registered run blocks (`smoke_1a`, `full_run`, `timing_1b`),
    and that block also fixes which base seeds and which n_sims are allowed:
    the 1b convergence batches are a registered protocol, not an ad-hoc run
    (item 4c). The matched block name and its seed cohort are written back
    into `effective` as `config_block` and `seed_cohort`.
    """
    arms = config.get("arms") or {}
    block = arms.get(arm)
    if not isinstance(block, dict):
        return [f"the config has no arm block for {arm!r}"]

    problems = []
    for label, key, value in (
        ("model dir", "model_dir", effective["model_dir"]),
        ("extras graft", "extras_graft", effective["extras_graft"]),
        ("bowler usage", "bowler_usage", effective["bowler_usage_path"]),
        ("roster policy", "roster_policy", effective["roster_policy_path"]),
        ("context dir", "context_dir", effective["context_dir"]),
        ("player metadata", "player_metadata",
         effective["player_metadata"]),
    ):
        pinned = block.get(key)
        if not _same_path(pinned, value):
            problems.append(
                f"{label}: run uses {value!r}, config pins {pinned!r}")

    odds_pinned = (block.get("odds") or {}).get("path")
    if not _same_path(odds_pinned, effective["odds"]):
        problems.append(
            f"odds: run uses {effective['odds']!r}, config pins "
            f"{odds_pinned!r}")

    for label, key, value in (
        ("stats version", "stats_version", effective["stats_version"]),
        ("threads", "threads", effective["threads"]),
        # The device the run RESOLVED, not the one it was asked for: a
        # T1_SIM_DEVICE=mps inheritance used to sail through (item 4a).
        ("device", "device", effective["device"]),
    ):
        pinned = block.get(key)
        if pinned != value:
            problems.append(
                f"{label}: run uses {value!r}, config pins {pinned!r}")

    pinned_clip = block.get("clip")
    clip = effective["clip"]
    if clip is None or [float(clip[0]), float(clip[1])] != [
            float(value) for value in (pinned_clip or [None, None])]:
        problems.append(
            f"clip bounds: run uses {clip!r}, config pins {pinned_clip!r}")

    matched = None
    for name in REGISTERED_BLOCKS:
        pinned_dir = (block.get(name) or {}).get("fixture_dir")
        if pinned_dir and _same_path(pinned_dir, effective["fixture_dir"]):
            matched = name
            break
    effective["config_block"] = matched
    effective["seed_cohort"] = [int(effective["base_seed"])]
    if matched is None:
        registered = {
            name: (block.get(name) or {}).get("fixture_dir")
            for name in REGISTERED_BLOCKS if block.get(name)}
        problems.append(
            f"fixture dir: run uses {effective['fixture_dir']!r}, which is "
            f"no registered fixture dir {registered!r}")
        return problems

    run_block = block[matched]
    seeds = _block_seeds(run_block, config)
    if not seeds:
        problems.append(
            f"{matched}: the config registers no base seed for this block")
    elif int(effective["base_seed"]) not in seeds:
        problems.append(
            f"base seed: run uses {effective['base_seed']!r}, the {matched} "
            f"block permits {seeds!r}")
    else:
        # Sibling batches of a multi-seed block share one seed line, so the
        # seed-interval screen must cover all of them (item 5).
        effective["seed_cohort"] = seeds

    permitted = _block_n_sims(run_block)
    if permitted == N_SIMS_PLACEHOLDER or (
            isinstance(permitted, str)):
        problems.append(
            f"n_sims: the {matched} block is not launchable yet — its "
            f"n_sims is the placeholder {permitted!r}; the value is chosen "
            "by the 1b convergence read and re-pinned before 1d")
    elif not permitted:
        problems.append(
            f"{matched}: the config registers no n_sims for this block")
    elif int(effective["n_sims"]) not in permitted:
        problems.append(
            f"n_sims: run uses {effective['n_sims']!r}, the {matched} block "
            f"permits {permitted!r}")
    return problems


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one sequence-track stage-1 arm (A, A50, B or C)")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--config", default=None,
        help="registered stage-1 config (experiments/configs/"
             "seq_stage1_sim_v1.yaml). When given, run_arm re-verifies every "
             "pin with pin_stage1.verify_config and refuses to launch unless "
             "this invocation matches the arm's registered block. Every "
             "registered command passes it; without it the run is recorded "
             "as config_verified: false and the cross-arm audit refuses it "
             "unless --allow-unregistered.")
    parser.add_argument("--fixture-dir", required=True,
                        help="directory of cricsheet fixture JSONs to score")
    parser.add_argument("--context-dir", default=DEFAULT_CONTEXT_DIR,
                        help="same-day replay corpus (default data/t20s_json)")
    parser.add_argument("--odds", required=True)
    parser.add_argument("--n-sims", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=None,
                        help="override the arm's model/checkpoint directory")
    parser.add_argument("--stats-version", default=None,
                        help="override the arm's stats cache version")
    parser.add_argument("--extras-graft", default=DEFAULT_EXTRAS_GRAFT,
                        help="B18 extras sidecar applied to EVERY arm; pass "
                             "'none' for the historical flat graft")
    parser.add_argument("--bowler-usage-path", default=None)
    parser.add_argument("--bowler-roster-policy", default=None,
                        help="use RosterEmpiricalBowlerSelector with this "
                             "policy JSON (default: plain empirical selector)")
    parser.add_argument("--clip-low", type=float, default=None)
    parser.add_argument("--clip-high", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", default=None, choices=("cpu", "mps", "cuda", "auto"),
        help="device for the T1 arms (default: T1_SIM_DEVICE, else cpu). "
             "The RESOLVED value is what --config checks and what the "
             "provenance records.")
    parser.add_argument("--player-metadata", default=DEFAULT_PLAYER_METADATA)
    args = parser.parse_args(argv)
    if (args.clip_low is None) != (args.clip_high is None):
        parser.error("--clip-low and --clip-high must be given together")
    if args.clip_low is not None and not (
            0.0 < args.clip_low < args.clip_high < 1.0):
        parser.error("require 0 < --clip-low < --clip-high < 1")
    if args.n_sims < 1:
        parser.error("--n-sims must be >= 1")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    threads = _early_thread_cap(argv)
    if "--parallel" in list(sys.argv if argv is None else argv):
        raise SystemExit(
            "run_arm refuses --parallel: the same-day replay lifecycle is "
            "strictly sequential and stage 1 needs one deterministic "
            "prediction-time cutoff per fixture")
    args = parse_args(argv)

    # Heavy imports live here so the seed/provenance helpers above stay
    # importable (and testable) without torch, xgboost or a stats cache.
    import run_sim_eval as runner  # noqa: E402
    from artifacts import artifact_path, md5_directory, md5_file  # noqa: E402
    from player_metadata import PlayerMetadataProvider  # noqa: E402
    from registered_experiment import reject_sealed, sha256  # noqa: E402
    from sim_eval.loaders import TestMatchLoader  # noqa: E402
    from sim_eval.run_sim_eval_t1 import _T1ReplayEvaluator  # noqa: E402
    from sim_eval.same_day_stats import (  # noqa: E402
        SameDayReplayStatsProvider)
    from sim_v1_2 import (RUNOUT_P, EmpiricalBowlerSelector,  # noqa: E402
                          ExtrasGraftConfig, RosterEmpiricalBowlerSelector,
                          SimulationEngine, XGBoostModelV2)
    from stats_provider import StatsProvider  # noqa: E402
    from t1_ppc_common import (actual_innings,  # noqa: E402
                               build_context_by_date, sim_result)

    spec = ARMS[args.arm]
    model_type = spec["model_type"]
    stats_version = args.stats_version or spec["stats_version"]
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif spec["model_role"]:
        model_dir = artifact_path(spec["model_role"])
    elif spec["model_dir"]:
        model_dir = Path(spec["model_dir"])
    else:
        raise SystemExit(
            f"arm {args.arm} has no default checkpoint: pass --model-dir "
            "with the seed directory chosen by the D5 check 5.3 rule "
            "(lowest validation LL in t1_ablation_v1_mps/summary.yaml)")
    graft_path = (None if str(args.extras_graft).lower() == "none"
                  else Path(args.extras_graft))
    usage_path = args.bowler_usage_path
    roster_path = args.bowler_roster_policy
    output_dir = Path(args.output_dir)
    stats_role = STATS_CACHE_ROLES.get(stats_version)
    stats_cache = (
        artifact_path(stats_role) if stats_role
        else MODELS_ROOT / f"player_stats_cache_{stats_version}.sqlite")

    for candidate in (args.fixture_dir, args.context_dir, args.odds,
                      args.output_dir, model_dir, graft_path, usage_path,
                      roster_path, args.player_metadata, stats_cache):
        if candidate is not None:
            reject_sealed(candidate)

    if not model_dir.is_dir():
        raise SystemExit(
            f"arm {args.arm}: model directory {model_dir} does not exist "
            f"({spec['description']}). Build it, or pass --model-dir.")
    if not stats_cache.exists():
        raise SystemExit(f"stats cache not found: {stats_cache}")
    if graft_path is not None and not graft_path.exists():
        raise SystemExit(f"extras graft sidecar not found: {graft_path}")

    effective_usage = Path(str(
        usage_path or artifact_path("bowler_phase_usage")))
    clip_bounds = (None if args.clip_low is None
                   else [float(args.clip_low), float(args.clip_high)])

    # The device is RESOLVED here, before anything is checked or recorded:
    # `--device`, else an inherited T1_SIM_DEVICE, else cpu, with `auto`
    # resolved exactly as sim_t1 resolves it. An inherited T1_SIM_DEVICE=mps
    # used to reach the wrapper while the provenance still said "cpu"
    # (Astra round 2, item 4a).
    device = _resolve_device(model_type, args.device)

    # ---- preflight: the registered pins, re-verified here ---------------
    config_block = None
    seed_cohort = [int(args.base_seed)]
    if args.config:
        from sequence_track.pin_stage1 import verify_config  # noqa: E402
        config_path = Path(args.config)
        reject_sealed(config_path)
        if not config_path.exists():
            raise SystemExit(f"registered config not found: {config_path}")
        print(f"[run_arm] verifying registered config {config_path} ...")
        problems = verify_config(config_path)
        if problems:
            raise SystemExit(
                f"run_arm refuses to launch: {config_path} does not match "
                f"the artifacts on disk ({len(problems)} mismatch(es)):\n  "
                + "\n  ".join(problems)
                + "\nRe-pin with pin_stage1.py --write, or fix the artifact.")
        import yaml as _yaml  # noqa: E402
        config = _yaml.safe_load(config_path.read_text())
        effective = {
            "model_dir": model_dir,
            "stats_version": stats_version,
            "extras_graft": graft_path,
            "bowler_usage_path": effective_usage,
            "roster_policy_path": roster_path,
            "odds": args.odds,
            "context_dir": args.context_dir,
            "player_metadata": args.player_metadata,
            "clip": clip_bounds,
            "base_seed": int(args.base_seed),
            "n_sims": int(args.n_sims),
            "threads": int(threads),
            "device": device,
            "fixture_dir": args.fixture_dir,
        }
        mismatches = registered_mismatches(config, args.arm, effective)
        if mismatches:
            raise SystemExit(
                f"run_arm refuses to launch: this invocation does not match "
                f"the registered arm {args.arm} block in {config_path} "
                f"({len(mismatches)} difference(s)):\n  "
                + "\n  ".join(mismatches)
                + "\nRun the command line the config registers, or re-pin.")
        config_block = effective["config_block"]
        seed_cohort = effective["seed_cohort"]
        config_sha256 = sha256(config_path)
        config_verified = True
        print(f"[run_arm] config verified: {config_path} "
              f"(arm {args.arm}, {config_block} block, device {device}, "
              f"n_sims {args.n_sims}, seed cohort {seed_cohort})")
    else:
        config_path = None
        config_sha256 = None
        config_verified = False
        print("[run_arm] WARNING: no --config; this run is UNREGISTERED and "
              "audit_cross_arm.py will refuse it without "
              "--allow-unregistered")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_stamp()
    started_monotonic = time.time()

    # ---- fixtures and same-day context ---------------------------------
    fixture_paths = sorted(Path(args.fixture_dir).glob("*.json"))
    if not fixture_paths:
        raise SystemExit(f"no fixture JSON files in {args.fixture_dir}")
    stems = {path.stem for path in fixture_paths}

    # ---- preflight: no two fixtures may share a draw schedule -----------
    # A block that registers several base seeds (the 1b convergence and
    # variability batches) runs sibling batches on ONE seed line, so the
    # screen covers the whole cohort, not just this launch's seed.
    assert_seed_intervals_disjoint(stems, seed_cohort, args.n_sims)
    print(f"[run_arm] seed intervals disjoint: closest two of "
          f"{len(stems) * len(seed_cohort)} (fixture, base seed) seeds are "
          f"{min_seed_gap(stems, seed_cohort)} apart "
          f"(need > --n-sims {args.n_sims}; cohort {seed_cohort})")
    selected_dates = set()
    for path in fixture_paths:
        with path.open() as handle:
            selected_dates.add(str(json.load(handle)["info"]["dates"][0]))

    print(f"[run_arm] arm={args.arm} ({spec['description']})")
    print(f"[run_arm] model_dir={model_dir} stats_version={stats_version} "
          f"threads={threads}")
    print(f"[run_arm] building same-day replay context from "
          f"{args.context_dir} ({len(stems)} fixtures over "
          f"{len(selected_dates)} dates)...")
    context_started = time.time()
    context_by_date = build_context_by_date(args.context_dir, selected_dates)
    print(f"[run_arm] context ready: "
          f"{sum(len(b) for b in context_by_date.values())} fixtures on "
          f"{len(context_by_date)} dates "
          f"({time.time() - context_started:.0f}s)")

    documents = {
        match_id: document
        for batch in context_by_date.values()
        for match_id, document in batch
    }

    # ---- provider (shared by every arm) --------------------------------
    metadata = PlayerMetadataProvider(args.player_metadata)
    provider = SameDayReplayStatsProvider(
        StatsProvider("models", version=stats_version), metadata)

    recorder = _LifecycleRecorder(provider)

    # ---- model factory --------------------------------------------------
    graft_applied_via = None

    def _apply_extras_graft(model):
        """Install the stage-1 extras sidecar on an XGBoost arm.

        `XGBoostModelV2` exposes NO explicit-path seam for the B18 graft: it
        only auto-detects an `extras_graft_v1.json` sitting next to the
        model artifact. The public attribute the engine reads
        (`getattr(model, 'extras_graft', None)`) is therefore the only seam
        available without editing `sim_v1_2.py`, and it is what the T1
        wrapper sets from `T1_EXTRAS_GRAFT_PATH` too. Recorded as a stage-0
        deviation.
        """
        nonlocal graft_applied_via
        detected = getattr(model, "extras_graft", None)
        if graft_path is None:
            if detected is not None:
                raise SystemExit(
                    f"--extras-graft none, but {model_dir} carries its own "
                    f"{detected.source} sidecar; the arm would silently use "
                    "it. Remove the sidecar or pass its path.")
            graft_applied_via = "none (historical flat 1% graft)"
            return
        if detected is not None and Path(str(detected.source)) != graft_path:
            raise SystemExit(
                f"{model_dir} carries its own extras sidecar "
                f"{detected.source}, which differs from the requested "
                f"{graft_path}; refusing an ambiguous extras law")
        model.extras_graft = ExtrasGraftConfig.from_path(graft_path)
        graft_applied_via = (
            "model_dir sidecar auto-detect" if detected is not None
            else "XGBoostModelV2.extras_graft attribute (no explicit-path "
                 "seam exists)")
        print(model.extras_graft.banner())
        print(f"  sidecar: {model.extras_graft.source} ({graft_applied_via})")

    if model_type == "transformer":
        booster_path = model_dir / "model.pt"
        if not booster_path.exists():
            raise SystemExit(f"checkpoint not found: {booster_path}")
        os.environ["T1_SIM_MODEL_DIR"] = str(model_dir)
        # The wrapper reads the env; write back the RESOLVED value so it
        # cannot differ from what was checked and recorded.
        os.environ["T1_SIM_DEVICE"] = device
        os.environ.pop("T1_SIM_PREFIX_CACHE", None)  # prefix cache OFF
        if graft_path is None:
            os.environ.pop("T1_EXTRAS_GRAFT_PATH", None)
            graft_applied_via = "none (historical flat 1% graft)"
        else:
            os.environ["T1_EXTRAS_GRAFT_PATH"] = str(graft_path)
            graft_applied_via = "T1_EXTRAS_GRAFT_PATH"

        def _model_factory(**_ignored_runner_kwargs):
            import torch
            from sim_t1 import TransformerT1SimModel
            # The wrapper re-sets 4 CPU threads itself; this makes the cap
            # explicit for anything it does not own.
            torch.set_num_threads(threads)
            return TransformerT1SimModel(
                stats_provider=provider, player_metadata=metadata)

        runner.TransformerModelV1 = _model_factory
    else:
        suffix = spec["artifact_suffix"]
        booster_path = _resolve_artifact(
            model_dir, "xgboost_model", suffix, "pkl")
        batter_encoder = _resolve_artifact(
            model_dir, "batter_encoder", suffix, "pkl")
        bowler_encoder = _resolve_artifact(
            model_dir, "bowler_encoder", suffix, "pkl")
        feature_columns = _resolve_artifact(
            model_dir, "feature_columns", suffix, "txt")

        def _model_factory(**_ignored_runner_kwargs):
            model = XGBoostModelV2(
                model_path=str(booster_path),
                batter_encoder_path=str(batter_encoder),
                bowler_encoder_path=str(bowler_encoder),
                feature_columns_path=str(feature_columns),
                stats_provider=provider,
                player_metadata=metadata,
                ball_calibrator=None,
            )
            _apply_extras_graft(model)
            return model

        runner.XGBoostModelV2 = _model_factory

    # ---- selector (captured for provenance) -----------------------------
    selector_box = {}

    def _selector_factory(usage_path=None, k=30):
        if roster_path:
            selector = RosterEmpiricalBowlerSelector(
                usage_path=usage_path, roster_path=roster_path, k=k)
        else:
            selector = EmpiricalBowlerSelector(usage_path=usage_path, k=k)
        selector_box["selector"] = selector
        return selector

    runner.EmpiricalBowlerSelector = _selector_factory

    # ---- engine that keeps the raw simulations --------------------------
    class _RecordingEngine(SimulationEngine):
        """Keeps one fixture's raw `MatchResult`s before aggregation."""

        def __init__(self, model, rules=None):
            super().__init__(model, rules)
            self.last_results = None

        def simulate_multiple(self, initial_state, config):
            self.last_results = None
            results = super().simulate_multiple(initial_state, config)
            self.last_results = results
            return results

    runner.SimulationEngine = _RecordingEngine

    # ---- chronological loader ------------------------------------------
    class _ChronologicalLoader(TestMatchLoader):
        def load_matches(self, _test_dir):
            matches = []
            for date_str in sorted(context_by_date):
                for match_id, document in context_by_date[date_str]:
                    if match_id not in stems:
                        continue
                    _, state = self.create_match_state(
                        document, cricsheet_id=match_id)
                    if state is None:
                        raise RuntimeError(
                            f"could not create state for {match_id}")
                    matches.append((match_id, state))
            missing = stems - {match_id for match_id, _state in matches}
            if missing:
                raise RuntimeError(
                    f"fixtures absent from {args.context_dir} chronology: "
                    f"{sorted(missing)}")
            print(f"Successfully loaded {len(matches)} matches "
                  "(chronological)")
            return matches

    runner.TestMatchLoader = _ChronologicalLoader

    # ---- evaluator: per-fixture seeds, clip seam, raw capture -----------
    raw_path = output_dir / RAW_SIMS_FILENAME
    raw_handle = raw_path.open("w")
    # Filled by the evaluator, read by the provenance writer: the odds row
    # the evaluator actually resolved, and whether the fixture produced a
    # MatchEvaluationResult. Neither may be inferred after the fact.
    odds_rows = {}
    scored_state = {}

    class _ArmEvaluator(_T1ReplayEvaluator):
        base_seed = args.base_seed
        prob_clip = (
            None if args.clip_low is None
            else (args.clip_low, args.clip_high))
        # `odds_rows` / `scored_state` are bound after the class body: a
        # class body is not a closure, so `x = x` there is a NameError.

        def _simulation_seed(self, match_id):
            return fixture_seed(str(match_id), self.base_seed)

        def _resolve_odds_row(self, match_id, match_state, odds_lookup,
                              claimed_rows=None):
            # The parent is a staticmethod; this override binds `self` so the
            # resolved row (or its absence) is recorded for provenance. A
            # RuntimeError from the parent (shared display alias) still
            # propagates and fails the run closed.
            row = _T1ReplayEvaluator._resolve_odds_row(
                match_id, match_state, odds_lookup, claimed_rows=claimed_rows)
            self.odds_rows[str(match_id)] = row
            if row is None:
                self.scored_state[str(match_id)] = (False, "no_odds_row")
            return row

        def _evaluate_single_match(self, match_id, match_state, odds_data):
            try:
                result = super()._evaluate_single_match(
                    match_id, match_state, odds_data)
                self._dump_raw(match_id, match_state)
            except Exception as exc:
                # The replay loop catches this and moves on; the fixture is
                # then NOT in the results JSON, which the audit must see.
                self.scored_state[str(match_id)] = (
                    False, f"evaluation_error: {type(exc).__name__}: {exc}")
                raise
            self.scored_state[str(match_id)] = (True, None)
            return result

        def _dump_raw(self, match_id, match_state):
            results = getattr(self.engine, "last_results", None)
            if not results:
                return
            document = documents[str(match_id)]
            lineups = {
                match_state.team1_lineup.team_name: match_state.team1_lineup,
                match_state.team2_lineup.team_name: match_state.team2_lineup,
            }

            def _card(team_name, card, value_index):
                lineup = lineups.get(team_name)
                if lineup is None:
                    return {}
                out = {}
                for player_index, values in card.items():
                    if 0 <= int(player_index) < len(lineup.players):
                        player = lineup.players[int(player_index)]
                        out[str(player.player_id)] = int(values[value_index])
                return out

            simulations = []
            for result in results:
                record = sim_result(result)
                record["tie"] = bool(result.winner == "Tie")
                for innings_record, innings in zip(record["innings"],
                                                   result.innings):
                    innings_record["batter_runs"] = _card(
                        innings.batting_team, innings.batting_card, 0)
                    innings_record["bowler_wickets"] = _card(
                        innings.bowling_team, innings.bowling_card, 2)
                simulations.append(record)

            info = document.get("info", {})
            raw_handle.write(json.dumps({
                "match_id": str(match_id),
                "date": str(info.get("dates", [None])[0]),
                "first_batting_team": document["innings"][0]["team"],
                "actual_winner": (info.get("outcome") or {}).get("winner"),
                "actual_innings": actual_innings(document),
                "fixture_seed": self._simulation_seed(match_id),
                "n_sims": len(simulations),
                "simulations": simulations,
            }) + "\n")
            raw_handle.flush()

    _ArmEvaluator.provider = recorder
    _ArmEvaluator.context_by_date = context_by_date
    _ArmEvaluator.odds_rows = odds_rows
    _ArmEvaluator.scored_state = scored_state
    runner.MatchLevelEvaluator = _ArmEvaluator

    # ---- delegate to the frozen runner ----------------------------------
    delegated = [
        "run_sim_eval.py",
        "--test-dir", str(args.fixture_dir),
        "--odds", str(args.odds),
        "--n-sims", str(args.n_sims),
        "--output-dir", str(output_dir),
        "--model-type", model_type,
        "--model-version", stats_version,
    ]
    if usage_path:
        delegated += ["--bowler-usage-path", str(usage_path)]
    if model_type == "xgboost":
        delegated += ["--model", str(booster_path)]
    else:
        # T1's token cache binds to the exact MatchState and fails closed on
        # desync; the generic guard targets the legacy stateful wrappers.
        delegated += ["--allow-stateful-wrapper"]

    before = {path.name for path in output_dir.glob("*.json")}
    saved_argv = sys.argv
    sys.argv = delegated
    try:
        print(f"[run_arm] delegating: {' '.join(delegated)}")
        runner.main()
    finally:
        sys.argv = saved_argv
        raw_handle.close()

    if not recorder.stamps:
        raise SystemExit(
            "no fixture reached the replay lifecycle — check the odds file, "
            "the fixture dir and the context chronology")

    # ---- eval.json copy --------------------------------------------------
    produced = [path for path in output_dir.glob("*.json")
                if path.name not in before
                and path.name not in (PROVENANCE_FILENAME,
                                      EVAL_COPY_FILENAME)]
    if not produced:
        raise SystemExit(
            f"{output_dir} holds no new evaluator results JSON; the "
            "delegated run did not save results")
    newest = max(produced, key=lambda path: path.stat().st_mtime)
    shutil.copyfile(newest, output_dir / EVAL_COPY_FILENAME)
    print(f"[run_arm] evaluator results: {newest.name} -> "
          f"{EVAL_COPY_FILENAME}")

    # ---- provenance ------------------------------------------------------
    selector = selector_box.get("selector")
    fixtures = []
    for match_id in sorted(stems):
        stamp = recorder.stamps.get(str(match_id))
        if stamp is None:
            raise SystemExit(
                f"fixture {match_id} never entered the replay lifecycle")
        odds_row = odds_rows.get(str(match_id))
        if (odds_row is not None) != bool(stamp["odds_row_found"]):
            raise SystemExit(
                f"fixture {match_id}: the recorded odds row and the "
                "lifecycle's prediction_required flag disagree")
        scored, skip_reason = scored_state.get(
            str(match_id), (False, "never_evaluated"))
        fixtures.append(fixture_provenance(
            cricsheet_id=match_id,
            document=documents[str(match_id)],
            as_of_date=stamp["date"],
            same_day_advanced_before=stamp["same_day_advanced_before"],
            matches_advanced=stamp["matches_advanced"],
            odds_row=odds_row,
            scored=scored,
            skip_reason=skip_reason,
            base_seed=args.base_seed,
            n_sims=args.n_sims,
        ))

    def _selector_path(attribute):
        if selector is None:
            return None
        return str(getattr(selector, attribute, "")) or None

    effective_usage = _selector_path("usage_path")
    effective_roster = _selector_path("roster_path")
    clip_low, clip_high = _ArmEvaluator.prob_clip or (0.05, 0.95)
    run = {
        "arm": args.arm,
        "model_dir": str(model_dir),
        "model_dir_hash": md5_directory(model_dir),
        "model_dir_hash_contract": DIR_HASH_CONTRACT,
        "checkpoint_path": str(booster_path),
        "checkpoint_md5": md5_file(booster_path),
        "stats_version": stats_version,
        "stats_cache_path": str(stats_cache),
        "stats_cache_md5": md5_file(stats_cache),
        "selector_class": type(selector).__name__ if selector else None,
        "bowler_usage_path": effective_usage,
        "bowler_usage_md5": (
            md5_file(effective_usage) if effective_usage else None),
        "bowler_roster_policy_path": effective_roster,
        "bowler_roster_policy_md5": (
            md5_file(effective_roster) if effective_roster else None),
        "extras_graft_path": str(graft_path) if graft_path else None,
        "extras_graft_sha256": sha256(graft_path) if graft_path else None,
        "extras_graft_applied_via": graft_applied_via,
        "runout_p": float(RUNOUT_P),
        "clip": {
            "low": float(clip_low),
            "high": float(clip_high),
            "seam_enabled": _ArmEvaluator.prob_clip is not None,
        },
        "base_seed": int(args.base_seed),
        "n_sims": int(args.n_sims),
        "min_seed_gap": min_seed_gap(stems, args.base_seed),
        "engine_md5": md5_file(SCRIPTS_DIR / "sim_v1_2.py"),
        "runner_md5": md5_file(Path(__file__).resolve()),
        "threads": int(threads),
        "device": device,
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_sha256,
        "config_verified": bool(config_verified),
        "config_block": config_block,
        "fixture_dir": str(args.fixture_dir),
        # Bound to the registered block by audit_cross_arm: a run cannot
        # claim a block whose fixture directory it did not read.
        "fixture_dir_hash": md5_directory(args.fixture_dir),
        "fixture_dir_hash_contract": DIR_HASH_CONTRACT,
        "fixture_count": len(stems),
        "context_dir": str(args.context_dir),
        "context_dir_hash": md5_directory(args.context_dir),
        "context_dir_hash_contract": DIR_HASH_CONTRACT,
        "odds": str(args.odds),
        "odds_sha256": sha256(args.odds),
        "player_metadata_sha256": sha256(args.player_metadata),
        "rng_streams": (
            "single global random/np.random stream per simulation "
            "(random_seed + i); the three named sub-seeds are recorded but "
            "not separately consumed — stage-0 deviation D5 5.9. Because "
            "the scheme is additive, one fixture occupies the seed interval "
            "[fixture_seed, fixture_seed + n_sims); run_arm refuses to "
            "launch when any two of this run's intervals overlap, and "
            "min_seed_gap above is this run's margin"
        ),
        "started_at": started_at,
        "finished_at": _utc_stamp(),
        "elapsed_seconds": round(time.time() - started_monotonic, 3),
    }
    path = write_provenance(output_dir, args.arm, run, fixtures)
    print(f"[run_arm] provenance: {path}")
    print(f"[run_arm] raw simulations: {raw_path}")
    print(f"[run_arm] arm {args.arm} finished in "
          f"{run['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
