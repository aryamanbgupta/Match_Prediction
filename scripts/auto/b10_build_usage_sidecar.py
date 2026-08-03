"""B10 — build the opt-in usage sidecar for the who-bowls alignment arm.

Produces two artifacts under `models/auto/b10/` (gitignored):

  usage_corpus.pkl            copy of `models/auto/b9/usage_corpus.pkl`
                              (per-player XI-appearance bowling log, keyed by
                              cricsheet NAME; rebuildable with
                              `scripts/auto/b9_usage_baseline.py --rebuild-corpus`)
  bowler_phase_usage_b10.json deep copy of `models/bowler_phase_usage.json`
                              plus ONE new top-level key `b10_asof_usage`,
                              which is what activates the B10 branch inside
                              `sim_v1_2.EmpiricalBowlerSelector`.

The production prior `models/bowler_phase_usage.json` is READ ONLY — its md5
is asserted unchanged before and after.

`k_usage` is read from `scripts/auto/b9_usage_baseline.K_USAGE` (never
hardcoded) so the sim-side as-of expected-balls model is the same statistic
B9 measured the gap with.

Run: uv run python scripts/auto/b10_build_usage_sidecar.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROD_USAGE = REPO / "models" / "bowler_phase_usage.json"
B9_CORPUS = REPO / "models" / "auto" / "b9" / "usage_corpus.pkl"
OUT_DIR = REPO / "models" / "auto" / "b10"
OUT_CORPUS = OUT_DIR / "usage_corpus.pkl"
OUT_USAGE = OUT_DIR / "bowler_phase_usage_b10.json"

MIN_ELIGIBLE = 5
MIN_SHARE = 0.01


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_b9_k_usage() -> float:
    p = REPO / "scripts" / "auto" / "b9_usage_baseline.py"
    spec = importlib.util.spec_from_file_location("b9_usage_baseline", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b9_usage_baseline"] = mod
    spec.loader.exec_module(mod)
    return float(mod.K_USAGE)


def main() -> None:
    assert PROD_USAGE.exists(), f"missing {PROD_USAGE}"
    assert B9_CORPUS.exists(), (
        f"missing {B9_CORPUS} — rebuild with "
        "`uv run python scripts/auto/b9_usage_baseline.py --rebuild-corpus`")

    md5_before = md5(PROD_USAGE)
    print(f"models/bowler_phase_usage.json md5 BEFORE: {md5_before}")

    k_usage = load_b9_k_usage()
    print(f"K_USAGE read from b9_usage_baseline.py: {k_usage}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(B9_CORPUS, OUT_CORPUS)
    print(f"corpus copied -> {OUT_CORPUS} (md5 {md5(OUT_CORPUS)})")

    with open(PROD_USAGE) as f:
        payload = json.load(f)
    assert "b10_asof_usage" not in payload, "production prior already tagged"
    payload["b10_asof_usage"] = {
        "corpus_path": "models/auto/b10/usage_corpus.pkl",
        "k_usage": k_usage,
        "min_eligible": MIN_ELIGIBLE,
        "min_share": MIN_SHARE,
    }
    with open(OUT_USAGE, "w") as f:
        json.dump(payload, f)
    print(f"sidecar prior -> {OUT_USAGE} (md5 {md5(OUT_USAGE)})")
    print(f"  b10_asof_usage = {json.dumps(payload['b10_asof_usage'])}")

    md5_after = md5(PROD_USAGE)
    print(f"models/bowler_phase_usage.json md5 AFTER:  {md5_after}")
    assert md5_after == md5_before, "production usage prior was modified!"
    print("OK: production prior unchanged.")


if __name__ == "__main__":
    main()
