#!/usr/bin/env python3
"""D7 — the ownership dependency test for one stage 2 checkpoint.

`docs/sequence_track/stage2_acceptance.md` D7.1–7.4. The claim a masked
relay-free arm makes is that the target row reads NOTHING outside

    S(i) = attention_set(i) union {history-source rows of i}

(arm register, "Dependency set S(i)"). This script certifies that claim
numerically on a trained checkpoint rather than on a hand-built innings: for
each sampled target row it

  1. computes S(i) with `transformer_t1.dependency_set` — the SAME function
     `scripts/tests/test_stage2_arms.py` asserts against, so the script and
     the unit tests cannot disagree about what is allowed (D7.1);
  2. replaces the 50 features of every row of that innings outside S(i) with
     N(0, 9) noise;
  3. replaces the RAW OUTCOME of every row outside S(i) with a different
     random class, and rebuilds every derived input from the perturbed
     outcomes — `prev_y` (innings-previous), `prev_bat`/`prev_bowl`
     (participant-aligned, via `transformer_t1.aligned_history`) and the
     own-outcome key tokens. Perturbing the raw labels rather than the
     derived arrays is what makes a leak through the aligned-input path
     visible (D7.2, Astra MUST-FIX 7);
  4. runs the model on CPU in eval mode on the original and the perturbed
     innings and records max |delta logits| at the target row.

Pass rule (D7.3): max |delta| <= 1e-6, expected exactly 0 under -inf masking.

The standard-wiring arms (`full`, `aligned_hist`) are POSITIVE CONTROLS
(D7.4): they relay through their neighbours' hidden states, so a max |delta|
above 1e-3 is what proves the test can see relay at all. D7.4 asks for them
"under the same perturbation", and that qualifier is load-bearing: a standard
arm's OWN attention set is the whole causal prefix, so the only rows outside
its own S(i) are future rows, which no causal arm reads — scored against its
own S(i) every control would read exactly 0 for the wrong reason. A control is
therefore scored against the S(i) of the masked arm that shares its history
input at k = 30 (`recency` for innings-previous, `same_entity` for
participant-aligned), recorded in the output as `dependency_set_arm` /
`dependency_set_k` and overridable with `--set-arm` / `--set-k`. For a
relay-free checkpoint the set is always its own, and `--set-arm` is refused:
nothing may loosen the certificate the arm is being certified against.

`pass` is the D7.3 rule verbatim for every arm; `positive_control_expected`
says which reading is the wanted one.

Reads the VALIDATION split only, through `transformer_t1`'s own loaders with
`--no-kit` semantics. The test split is never loaded, and nothing under
`data/golden/` or `data/forward_holdout/` is reachable from here.

Usage:
    uv run --no-sync python scripts/sequence_track/ownership_dependency_test.py \
        --checkpoint models/embeddings/seq_stage2/smoke/same_entity_k30/seed_7 \
        --out models/embeddings/seq_stage2/dependency/same_entity_k30.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import transformer_t1 as t1  # noqa: E402

# The registered pass rule and the positive-control floor (D7.3 / D7.4).
PASS_TOLERANCE = 1e-6
CONTROL_FLOOR = 1e-3
# N(0, 9): the acceptance file's variance, i.e. standard deviation 3.
NOISE_SD = 3.0
SPLIT = "validation"
# The masked counterpart whose S(i) scores a standard-wiring positive control,
# by the control's history input (see the module docstring). k = 30 is the
# registered window of the night's `recency` arm and of `same_entity_k30`.
CONTROL_SET_ARM = {"innings_previous": ("recency", 30),
                   "participant_aligned": ("same_entity", 30)}


class DependencyTestError(RuntimeError):
    """A refusal: a checkpoint, a frame or an arm this test cannot certify."""


def load_checkpoint(directory: Path) -> tuple[dict, dict]:
    """`model.pt` state dict and `metrics.json` of one run directory."""
    directory = Path(directory)
    model_path, metrics_path = directory / "model.pt", directory / "metrics.json"
    for path in (model_path, metrics_path):
        if not path.is_file():
            raise DependencyTestError(f"{path} not found")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    metrics = json.loads(metrics_path.read_text())
    if "arm_params" not in metrics:
        raise DependencyTestError(
            f"{metrics_path} carries no arm_params block; this is not a "
            "stage 2 checkpoint")
    return state, metrics


def build_model(metrics: dict, state: dict, device: str) -> t1.T1Model:
    """The model the checkpoint was trained as, loaded strictly.

    Architecture and arm come from `metrics.json`, never from a flag, so a
    checkpoint cannot be certified as an arm it was not trained as.
    """
    arm_params, config = metrics["arm_params"], metrics["config"]
    arm, k = arm_params["arm"], arm_params["k"]
    if arm in t1.ARMS_NEEDING_BASE_LOGITS:
        raise DependencyTestError(
            f"arm {arm!r} reads base log-probabilities; D7 covers the masked "
            "arms and the two standard-wiring positive controls only")
    if t1.ARM_WIRING[arm] == "recurrent":
        raise DependencyTestError(
            f"arm {arm!r} is recurrent: it has no attention set, so S(i) is "
            "not defined for it and D7 does not cover it")
    model = t1.T1Model(t1.N_FEATS, int(config["dmodel"]),
                       int(config["layers"]), int(config["heads"]),
                       arm=arm, k=k)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_validation(metrics: dict):
    """The validation split of the frame the checkpoint was trained on."""
    config = metrics["config"]
    data_dir = Path(config["data_dir"])
    version = config.get("frame_version") or t1.resolve_frame_version(data_dir)
    if not data_dir.is_dir():
        raise DependencyTestError(
            f"frame {data_dir.as_posix()} (from the checkpoint's config) is "
            "not a directory on this machine")
    df = t1.load_split(SPLIT, data_dir, version)
    feats = t1.build_features(df)
    y = df["y"].to_numpy().astype(np.int64)
    innings = t1.build_innings(df)
    batter = pd.factorize(df["batter_id"])[0].astype(np.int64)
    bowler = pd.factorize(df["bowler_id"])[0].astype(np.int64)
    return feats, y, innings, batter, bowler


def _forward(model: t1.T1Model, feats, y, batter, bowler, device):
    """Logits of one innings, with every derived input rebuilt from `y`."""
    rows = [np.arange(len(y))]
    prev_bat, prev_bowl = t1.aligned_history(y, batter, bowler, rows)
    batch = t1.collate(rows, feats, y, device, batter=batter, bowler=bowler,
                       prev_bat=prev_bat, prev_bowl=prev_bowl)
    with torch.no_grad():
        out, _ = model(batch.feats, batch.prev_y, batch.pad,
                       prev_bat=batch.prev_bat, prev_bowl=batch.prev_bowl,
                       batter=batch.batter, bowler=batch.bowler,
                       own_y=batch.y)
    return out[0]


def perturb_outside(rng, feats, y, keep: set[int]):
    """Copies of one innings' features and raw outcomes, perturbed outside S(i).

    Features become N(0, 9) noise; each outcome becomes a DIFFERENT random
    class (uniform over the other five), so a perturbation that should be
    visible always is — a redraw that happened to land on the original label
    would silently weaken the positive controls.
    """
    new_feats, new_y = feats.copy(), y.copy()
    outside = np.array(sorted(set(range(len(y))) - keep), dtype=np.int64)
    if len(outside) == 0:
        return new_feats, new_y, 0
    new_feats[outside] = (rng.standard_normal(
        (len(outside), feats.shape[1])) * NOISE_SD).astype(np.float32)
    shift = rng.integers(1, t1.N_OUTCOME_CLASSES, size=len(outside))
    new_y[outside] = (new_y[outside] + shift) % t1.N_OUTCOME_CLASSES
    return new_feats, new_y, int(len(outside))


def resolve_dependency_set_arm(arm: str, k, set_arm: str | None,
                               set_k=None) -> tuple[str, object]:
    """Which arm's S(i) the perturbation is scored against.

    A relay-free arm is always scored against its own set — that is the
    certificate. A standard-wiring control is scored against its masked
    counterpart's set, because its own set leaves only future rows outside
    (see the module docstring).
    """
    if t1.ARM_WIRING[arm] == "relay_free":
        if set_arm is not None:
            raise DependencyTestError(
                f"--set-arm is refused for the relay-free arm {arm!r}: its "
                "certificate is its OWN dependency set")
        return arm, k
    if set_arm is not None:
        if set_arm not in t1.ALL_ARMS:
            raise DependencyTestError(f"--set-arm {set_arm!r} is not an arm")
        return set_arm, set_k
    try:
        return CONTROL_SET_ARM[t1.ARM_HISTORY[arm]]
    except KeyError:
        raise DependencyTestError(
            f"arm {arm!r} reads no history input, so no masked counterpart "
            "defines a perturbation for it; pass --set-arm explicitly"
        ) from None


def run(checkpoint: Path, n_targets: int, seed: int, device: str,
        set_arm: str | None = None, set_k=None) -> dict:
    state, metrics = load_checkpoint(checkpoint)
    arm_params = metrics["arm_params"]
    arm, k = arm_params["arm"], arm_params["k"]
    dep_arm, dep_k = resolve_dependency_set_arm(arm, k, set_arm, set_k)
    model = build_model(metrics, state, device)
    feats, y, innings, batter, bowler = load_validation(metrics)

    # Uniform over validation rows (D7.1 stratifies nothing), seed 29.
    rng = np.random.default_rng(seed)
    row_to_innings = np.empty(len(y), dtype=np.int64)
    row_to_position = np.empty(len(y), dtype=np.int64)
    for index, rows in enumerate(innings):
        row_to_innings[rows] = index
        row_to_position[rows] = np.arange(len(rows))
    sampled = rng.choice(len(y), size=min(int(n_targets), len(y)),
                         replace=False)

    deltas = np.zeros(len(sampled), dtype=np.float64)
    n_nothing_perturbed = 0
    started = time.time()
    for order, row in enumerate(sampled):
        rows = innings[row_to_innings[row]]
        target = int(row_to_position[row])
        inn_feats, inn_y = feats[rows], y[rows]
        inn_bat, inn_bowl = batter[rows], bowler[rows]
        keep = t1.dependency_set(dep_arm, dep_k, target, inn_bat, inn_bowl,
                                 len(rows))
        new_feats, new_y, n_outside = perturb_outside(rng, inn_feats, inn_y,
                                                     keep)
        if n_outside == 0:
            n_nothing_perturbed += 1
        before = _forward(model, inn_feats, inn_y, inn_bat, inn_bowl, device)
        after = _forward(model, new_feats, new_y, inn_bat, inn_bowl, device)
        deltas[order] = float((before[target] - after[target]).abs().max())

    max_abs = float(deltas.max()) if len(deltas) else 0.0
    positive_control = t1.ARM_WIRING[arm] == "standard"
    return {
        "arm": arm,
        "k": k,
        "wiring": t1.ARM_WIRING[arm],
        "dependency_set_arm": dep_arm,
        "dependency_set_k": dep_k,
        "key_construction": arm_params.get("key_construction"),
        "checkpoint": Path(checkpoint).as_posix(),
        "checkpoint_md5": t1.md5_file(Path(checkpoint) / "model.pt"),
        "split": SPLIT,
        "n_rows": int(len(y)),
        "n_innings": int(len(innings)),
        "n_targets": int(len(sampled)),
        "n_targets_with_nothing_outside_s": n_nothing_perturbed,
        "seed": int(seed),
        "noise_sd": NOISE_SD,
        "tolerance": PASS_TOLERANCE,
        "control_floor": CONTROL_FLOOR,
        "max_abs_delta": max_abs,
        "p99_abs_delta": (float(np.percentile(deltas, 99)) if len(deltas)
                          else 0.0),
        "n_nonzero_gt_1e-6": int((deltas > PASS_TOLERANCE).sum()),
        "pass": bool(max_abs <= PASS_TOLERANCE),
        "positive_control_expected": positive_control,
        "positive_control_observed": bool(max_abs > CONTROL_FLOOR),
        "device": device,
        "wall_seconds": round(time.time() - started, 1),
    }


def summary_line(result: dict) -> str:
    role = ("positive control" if result["positive_control_expected"]
            else "masked arm")
    if result["positive_control_expected"]:
        verdict = ("SEES RELAY" if result["positive_control_observed"]
                   else "NO RELAY SEEN")
    else:
        verdict = "PASS" if result["pass"] else "FAIL"
    scored = ("" if result["dependency_set_arm"] == result["arm"]
              else f" vs S(i) of {result['dependency_set_arm']}"
                   f" k={result['dependency_set_k']}")
    return (f"{result['arm']} k={result['k']} ({role}{scored}): n="
            f"{result['n_targets']} max|d|={result['max_abs_delta']:.3e} "
            f"p99={result['p99_abs_delta']:.3e} "
            f"n>1e-6={result['n_nonzero_gt_1e-6']} {verdict}")


def exit_code(result: dict) -> int:
    """1 when the certification (or its positive control) failed, else 0.

    A completed invocation that exited 0 regardless of the numbers made this
    script unusable as a gate: a masked arm leaking relay, or a control that
    cannot see relay (so the test proves nothing), has to be a non-zero exit
    (Astra SHOULD 1). The two rules are D7.3 (max |delta| <= 1e-6 for a masked
    arm) and D7.4 (max |delta| > 1e-3 for a positive control).
    """
    if result["positive_control_expected"]:
        return 0 if result["positive_control_observed"] else 1
    return 0 if result["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="a run directory holding model.pt and metrics.json")
    ap.add_argument("--n-targets", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu",
                    help="D7 runs on the CPU; anything else is opt-in")
    ap.add_argument("--set-arm", default=None,
                    help="score a standard-wiring positive control against "
                         "THIS arm's S(i) instead of its registered masked "
                         "counterpart; refused for a relay-free arm")
    ap.add_argument("--set-k", default=None,
                    help="the k of --set-arm (an int >= 0 or 'unr')")
    args = ap.parse_args(argv)
    try:
        set_k = t1.parse_k(args.set_k)
    except ValueError as exc:
        ap.error(str(exc))
    result = run(args.checkpoint, args.n_targets, args.seed, args.device,
                 set_arm=args.set_arm, set_k=set_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(summary_line(result), flush=True)
    return exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
