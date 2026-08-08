"""Join DeepCrease shot/line/length/control annotations onto the
Match_Prediction feature parquets.

Source: ~/Projects/stat-generator/data/t20_parquet/*.parquet (680k balls,
2021-2026, 11 T20 comps). p_match IS the cricsheet match id; their `over`
is 1-based. Ball-within-over numbering may disagree on extras, so the join
key is sequence POSITION within (match, innings, over) on both sides, and
every joined row is validated by comparing runs off the bat
(theirs `batruns` vs ours `batter_runs`); mismatches are dropped.

Output: models/embeddings/deepcrease_labels/{split}.parquet with columns
row_idx (position in the split parquet), shot, line, length, control.

Usage: uv run python scripts/deepcrease_join.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DC_DIR = Path("/Users/aryamangupta/Projects/stat-generator/data/t20_parquet")
OUT = Path("models/embeddings/deepcrease_labels")


def load_deepcrease() -> pd.DataFrame:
    frames = []
    for p in sorted(DC_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(
            p, columns=["p_match", "inns", "over", "ball", "batruns",
                        "shot", "line", "length", "control"]))
    dc = pd.concat(frames, ignore_index=True)
    dc["p_match"] = dc["p_match"].astype(str)
    dc["over0"] = dc["over"].astype(int) - 1
    dc = dc.sort_values(["p_match", "inns", "over0", "ball"])
    dc["pos"] = dc.groupby(["p_match", "inns", "over0"]).cumcount()
    print(f"deepcrease: {len(dc):,} balls, {dc['p_match'].nunique():,} matches",
          flush=True)
    return dc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dc = load_deepcrease()

    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(
            f"data/xgb_data_v3/cricket_data_v3_{split}.parquet",
            columns=["innings_id", "over_idx", "batter_runs"])
        df = df.reset_index().rename(columns={"index": "row_idx"})
        parts = df["innings_id"].str.split("_", n=1, expand=True)
        df["inns"] = parts[0].astype(int)
        df["p_match"] = parts[1]
        df["pos"] = df.groupby(["p_match", "inns", "over_idx"]).cumcount()

        merged = df.merge(
            dc, left_on=["p_match", "inns", "over_idx", "pos"],
            right_on=["p_match", "inns", "over0", "pos"], how="inner")
        agree = (merged["batter_runs"] == merged["batruns"])
        n_match, n_agree = len(merged), int(agree.sum())
        merged = merged[agree]
        keep = merged[["row_idx", "shot", "line", "length", "control"]]
        keep = keep[(keep["shot"].notna()) & (keep["shot"] != "-")]
        keep.to_parquet(OUT / f"{split}.parquet", index=False)
        print(f"{split}: {len(df):,} rows | joined {n_match:,} "
              f"| runs-agree {n_agree:,} ({n_agree/max(n_match,1):.1%}) "
              f"| labeled kept {len(keep):,} "
              f"({len(keep)/len(df):.1%} of split)", flush=True)


if __name__ == "__main__":
    main()
