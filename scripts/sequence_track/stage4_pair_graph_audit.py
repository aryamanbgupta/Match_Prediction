# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Stage 4 batter x bowler exposure-graph audit for rung 4a.

4a proposes learning matchup structure on the bipartite batter-bowler
graph. That only works if the graph is actually connected and if the
held-out splits contain pairs the graph can reach. This script measures
both on the i7 frame:

  * connected components of the train-split bipartite graph (edge = at
    least one ball bowled), and the size of the largest;
  * per competition_tier, the share of train edges internal to the tier
    (cross-tier edges are what let league and international play inform
    each other);
  * for the validation split, the share of rows whose pair is seen in
    train, whose pair is unseen but both players are seen, and whose
    batter or bowler is unseen.

The test split is sealed for Stage 4: this audit reports train and
validation only (the guard lives in stage4_references.check_split).

Writes research/reports/embeddings/STAGE4_PAIR_GRAPH_AUDIT.md.

Usage:
    uv run --no-sync python scripts/sequence_track/stage4_pair_graph_audit.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from stage4_guard import assert_permitted_inputs  # noqa: E402
from stage4_references import ROOT, split_path  # noqa: E402

# split_path() runs assert_permitted_inputs before any read, so the sealed
# test parquet, symlinks wearing a permitted name, and golden /
# forward_holdout paths are refused before the file is opened.

REPORT = ROOT / "research/reports/embeddings/STAGE4_PAIR_GRAPH_AUDIT.md"
COLS = ["batter_id", "bowler_id", "competition_tier"]


def components(edges: set[tuple[str, str]]) -> tuple[int, int]:
    """Connected components of the bipartite graph via union-find.
    Batters and bowlers are namespaced so a shared player id on both
    sides does not silently merge two roles into one node."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for bat, bowl in edges:
        a, b = find("bat:" + bat), find("bowl:" + bowl)
        if a != b:
            parent[a] = b
    sizes = pd.Series([find(n) for n in parent]).value_counts()
    return int(len(sizes)), int(sizes.iloc[0])


def main() -> None:
    # Validate every input up front, before the first read.
    assert_permitted_inputs([split_path("train"), split_path("validation")])
    train = pd.read_parquet(split_path("train"), columns=COLS)
    edges_df = train.drop_duplicates(["batter_id", "bowler_id"])
    edges = set(zip(edges_df["batter_id"], edges_df["bowler_id"]))
    n_comp, largest = components(edges)

    # Tier locality: for each tier, the share of that tier's distinct
    # edges that never appear in any other tier.
    tier_edges = train.drop_duplicates(["batter_id", "bowler_id",
                                        "competition_tier"])
    pair = list(zip(tier_edges["batter_id"], tier_edges["bowler_id"]))
    tier_edges = tier_edges.assign(_pair=pair)
    n_tiers_per_pair = tier_edges.groupby("_pair")["competition_tier"].nunique()
    tier_edges["_internal"] = tier_edges["_pair"].map(n_tiers_per_pair).eq(1)
    tier_rows = (tier_edges.groupby("competition_tier")
                 .agg(edges=("_internal", "size"),
                      internal_share=("_internal", "mean"))
                 .reset_index())

    bat_seen = set(train["batter_id"])
    bowl_seen = set(train["bowler_id"])

    lines = [
        "# Stage 4 — batter x bowler exposure-graph audit (rung 4a)",
        "",
        f"Built {time.strftime('%Y-%m-%d')} · frame `data/xgb_data_i7` · "
        "`scripts/sequence_track/stage4_pair_graph_audit.py`",
        "",
        "## Train-split bipartite graph",
        "",
        f"- balls: {len(train):,}",
        f"- distinct batters: {len(bat_seen):,}; distinct bowlers: "
        f"{len(bowl_seen):,}",
        f"- distinct (batter, bowler) edges: {len(edges):,}",
        f"- connected components: **{n_comp}**",
        f"- largest component: **{largest:,} nodes** "
        f"({largest / (len(bat_seen) + len(bowl_seen)):.2%} of all "
        "batter/bowler nodes)",
        "",
        "## Edge locality by competition tier",
        "",
        "`internal` = the pair is only ever seen inside this tier.",
        "",
        "| tier | distinct edges | internal to tier |",
        "|---|---|---|",
    ]
    for r in tier_rows.itertuples():
        lines.append(f"| {r.competition_tier} | {r.edges:,} | "
                     f"{r.internal_share:.1%} |")

    lines += ["", "## Held-out pair coverage", "",
              "| split | rows | pair seen | pair unseen, both players seen | "
              "batter or bowler unseen |", "|---|---|---|---|---|"]
    for name in ("validation",):
        df = pd.read_parquet(split_path(name), columns=COLS)
        pairs = list(zip(df["batter_id"], df["bowler_id"]))
        seen = np.fromiter((p in edges for p in pairs), dtype=bool,
                           count=len(pairs))
        known = np.fromiter(
            ((b in bat_seen) and (w in bowl_seen) for b, w in pairs),
            dtype=bool, count=len(pairs))
        unseen_player = ~known
        unseen_pair_known = known & ~seen
        lines.append(
            f"| {name} | {len(df):,} | {seen.mean():.1%} | "
            f"{unseen_pair_known.mean():.1%} | {unseen_player.mean():.1%} |")
        print(f"{name}: seen {seen.mean():.1%}, unseen-pair "
              f"{unseen_pair_known.mean():.1%}, unseen-player "
              f"{unseen_player.mean():.1%}", flush=True)

    lines += [
        "",
        "Expected from the ladder's earlier accounting: 40.7 / 46.0 / 13.4 % "
        "on validation. The test split is sealed for Stage 4 and is not "
        "reported here.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"\ncomponents={n_comp}, largest={largest:,}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
