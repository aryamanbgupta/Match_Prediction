"""D12 — build the production-feature-set parquet from the m3_unfrozen superset.

Production (`models/xgb_match_v3_m7_production`) trains on 46 numeric features
+ venue/competition_tier encodings (48 total, `feature_columns.txt`). No
dedicated parquet with exactly that schema survives on disk;
`data/xgb_match_data_v3_m3_unfrozen` (90 cols) is the unfrozen superset the
M3-era runs materialized. This script subsets it to metadata + the production
46 numeric columns IN THE EXACT production feature order (colsample makes
column order load-bearing — A9 lesson) and writes data/auto/d12/.

golden_test.parquet is deliberately NOT copied (golden is loop-forbidden).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "data/xgb_match_data_v3_m3_unfrozen"
OUT = ROOT / "data/auto/d12"
PROD = ROOT / "models/xgb_match_v3_m7_production/feature_columns.txt"

METADATA = ["match_id", "cricsheet_id", "match_date", "team1", "team2",
            "venue", "competition_tier", "team1_wins"]
ENCODED = {"venue_id_encoded", "competition_tier_encoded"}


def main() -> None:
    prod_cols = [c for c in PROD.read_text().split() if c]
    numeric = [c for c in prod_cols if c not in ENCODED]
    assert len(prod_cols) == 48 and len(numeric) == 46, \
        f"unexpected production feature list: {len(prod_cols)}/{len(numeric)}"
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(SRC / f"{split}.parquet")
        missing = [c for c in METADATA + numeric if c not in df.columns]
        assert not missing, f"{split}: missing {missing}"
        sub = df[METADATA + numeric]
        sub.to_parquet(OUT / f"{split}.parquet", index=False)
        print(f"  {split}: {len(df.columns)} -> {len(sub.columns)} cols, "
              f"{len(sub):,} rows")
    print(f"wrote {OUT} (metadata {len(METADATA)} + numeric {len(numeric)}, "
          "numeric in production feature_columns.txt order)")


if __name__ == "__main__":
    sys.exit(main())
