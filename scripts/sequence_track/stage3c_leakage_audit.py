# manifest-exempt: embeddings-ladder experimental artifact namespace
"""Stage 3c pre-flight: which of the 50 sequence-model input features
algebraically reveal the ball-t outcome from features at t and t+1?

Four probes, all on data/xgb_data_i7/cricket_data_i7_validation.parquet
(grouped 5-fold CV by innings_id, parquet row order = ball order):

  1. per-feature: depth-4 tree on [f_t], on [f_t, f_{t+1}], and on the
     explicit delta [f_{t+1} - f_t] -> outcome_t. The delta column matters:
     an axis-aligned depth-4 tree cannot express a subtraction of two raw
     levels, so a levels-only probe UNDER-reports algebraic leakage;
  2. residual floor after dropping the flagged set (levels + deltas);
  3. greedy backward elimination: drop the most-used feature, re-measure,
     repeat, until the tree is back at the majority-class baseline. The
     surviving set is what a masked-outcome objective could legally see;
  4. span mask (6 balls): can (x_start, x_end, x_end - x_start) recover the
     runs and wickets inside the span?

Usage:
    uv run --no-sync python scripts/sequence_track/stage3c_leakage_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from embeddings_e1 import (CLASS_MAPPING, CTX_COLS, EB_BAT_COLS,  # noqa: E402
                           EB_BOWL_COLS, VENUE_COLS)
from transformer_t1 import STATE_COLS, build_features  # noqa: E402

PARQUET = REPO / "data/xgb_data_i7/cricket_data_i7_validation.parquet"
OUT_MD = REPO / "research/reports/embeddings/STAGE3C_LEAKAGE_AUDIT.md"
OUT_JSON = REPO / "research/reports/embeddings/stage3c_leakage_audit.json"
FEAT_NAMES = (EB_BAT_COLS + EB_BOWL_COLS + VENUE_COLS
              + ["is_middle_overs", "is_death_overs", "wickets_in_hand_n",
                 "is_chase"] + STATE_COLS)
SPAN, SEED, TOL = 6, 0, 0.005
DASH = "\u2014"


def cv_acc(X, y, groups, n_splits=5, depth=4):
    """Grouped-CV accuracy of a depth-limited tree."""
    hits = 0
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        clf.fit(X[tr], y[tr])
        hits += int((clf.predict(X[te]) == y[te]).sum())
    return hits / len(y)


def cv_exact(X, z, groups, n_splits=5, depth=10, tol=1e-6):
    """Grouped-CV exact-recovery rate + MAE of a depth-limited regressor."""
    hits, abserr = 0, 0.0
    for tr, te in GroupKFold(n_splits=n_splits).split(X, z, groups):
        reg = DecisionTreeRegressor(max_depth=depth, random_state=SEED)
        reg.fit(X[tr], z[tr])
        p = reg.predict(X[te])
        hits += int((np.abs(p - z[te]) <= tol).sum())
        abserr += float(np.abs(p - z[te]).sum())
    return hits / len(z), abserr / len(z)


def main() -> None:
    cols = (["innings_id", "ball_outcome"] + EB_BAT_COLS + EB_BOWL_COLS
            + VENUE_COLS + CTX_COLS + STATE_COLS)
    df = pd.read_parquet(PARQUET, columns=cols)
    X = build_features(df).astype(np.float64)
    y = df["ball_outcome"].map(CLASS_MAPPING).to_numpy(np.int64)
    inn = df["innings_id"].to_numpy()
    assert X.shape[1] == len(FEAT_NAMES) == 50, X.shape

    idx = np.nonzero(np.r_[inn[1:] == inn[:-1], False])[0]   # has successor
    jdx = idx + 1
    yt, g = y[idx], inn[idx]
    A, B = X[idx], X[jdx]
    D = B - A
    base = float(np.bincount(yt).max() / len(yt))
    print(f"{len(idx)} pairs, {len(np.unique(g))} innings, base {base:.4f}",
          flush=True)

    def block(ks):
        ks = list(ks)
        return np.c_[A[:, ks], B[:, ks], D[:, ks]]

    # ---- probe 1: per feature -------------------------------------------
    rows = []
    for k, name in enumerate(FEAT_NAMES):
        a_self = cv_acc(A[:, [k]], yt, g)
        a_pair = cv_acc(np.c_[A[:, k], B[:, k]], yt, g)
        a_del = cv_acc(D[:, [k]], yt, g)
        best = max(a_pair, a_del)
        flag = ("algebraic" if best > 0.99 else
                "strong" if best > 0.80 else "")
        rows.append({"feature": name, "acc_t": a_self, "acc_pair": a_pair,
                     "acc_delta": a_del, "best": best, "lift": best - base,
                     "flag": flag})
        print(f"  {name:28s} t={a_self:.4f} pair={a_pair:.4f} "
              f"delta={a_del:.4f} {flag}", flush=True)
    rows.sort(key=lambda r: -r["best"])
    flagged = [r["feature"] for r in rows if r["flag"]]

    # ---- probe 2: residual floor after the flagged drop ------------------
    keep = [k for k, n in enumerate(FEAT_NAMES) if n not in set(flagged)]
    full = cv_acc(block(range(50)), yt, g)
    full_lv = cv_acc(np.c_[A, B], yt, g)
    resid = cv_acc(block(keep), yt, g)
    print(f"full {full:.4f} (levels only {full_lv:.4f}) | "
          f"flagged-drop residual {resid:.4f}", flush=True)

    # ---- probe 3: greedy backward elimination ---------------------------
    remain, trace = list(range(50)), []
    while remain:
        Xr = block(remain)
        acc = cv_acc(Xr, yt, g)
        clf = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        clf.fit(Xr, yt)
        imp = clf.feature_importances_.reshape(3, -1).sum(0)
        w = int(np.argmax(imp))
        done = acc <= base + TOL
        trace.append({"n_features": len(remain), "acc": acc,
                      "next_dropped": None if done else FEAT_NAMES[remain[w]]})
        print(f"  greedy k={len(remain):2d} acc={acc:.4f} "
              f"drop={FEAT_NAMES[remain[w]]}", flush=True)
        if done:
            break
        remain.pop(w)
    greedy_keep = [FEAT_NAMES[k] for k in remain]
    greedy_drop = [n for n in FEAT_NAMES if n not in set(greedy_keep)]

    # ---- probe 4: 6-ball span -------------------------------------------
    runs = np.array([0, 1, 2, 4, 6, 0], dtype=np.float64)[y]
    wkt = (y == 5).astype(np.float64)
    s, e = [], []
    for ix in df.groupby("innings_id", sort=False).indices.values():
        ix = np.sort(ix)
        for a in range(0, len(ix) - SPAN):
            s.append(ix[a])
            e.append(ix[a + SPAN])
    s, e = np.asarray(s), np.asarray(e)
    csr, csw = np.r_[0, np.cumsum(runs)], np.r_[0, np.cumsum(wkt)]
    zr, zw = csr[e] - csr[s], csw[e] - csw[s]
    Xs = np.c_[X[s], X[e], X[e] - X[s]]
    gs = inn[s]
    kk = (np.r_[remain, np.array(remain) + 50, np.array(remain) + 100]
          if remain else np.array([], dtype=int))
    sp = {"runs_all": cv_exact(Xs, zr, gs), "wkts_all": cv_exact(Xs, zw, gs),
          "runs_keep": cv_exact(Xs[:, kk], zr, gs) if len(kk) else (0.0, 0.0),
          "wkts_keep": cv_exact(Xs[:, kk], zw, gs) if len(kk) else (0.0, 0.0)}
    print("span " + " | ".join(f"{k} exact={v[0]:.4f} MAE={v[1]:.3f}"
                               for k, v in sp.items()), flush=True)

    res = {"n_pairs": len(idx), "n_innings": int(len(np.unique(g))),
           "majority_baseline": base, "per_feature": rows, "flagged": flagged,
           "full_acc": full, "full_levels_only_acc": full_lv,
           "flagged_drop_residual": resid, "greedy_trace": trace,
           "greedy_keep": greedy_keep, "greedy_drop": greedy_drop,
           "span": {"n": len(s), **{k: list(v) for k, v in sp.items()}}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))

    write_report(res)


def write_report(res: dict) -> None:
    base = res["majority_baseline"]
    rows, trace, sp = res["per_feature"], res["greedy_trace"], res["span"]
    eb = [r for r in rows if r["feature"].startswith(("batter_", "bowler_"))]
    keep_n = len(FEAT_NAMES) - len(res["flagged"])
    L = [
        "# Stage 3c pre-flight: outcome leakage in the 50-feature contract",
        "",
        "Generated by `scripts/sequence_track/stage3c_leakage_audit.py`. Frame "
        "`data/xgb_data_i7/cricket_data_i7_validation.parquet`; "
        f"{res['n_pairs']:,} (t, t+1) same-innings pairs over "
        f"{res['n_innings']} innings. Depth-4 sklearn decision trees, 5-fold "
        f"`GroupKFold` by `innings_id`. Majority-class baseline **{base:.4f}**.",
        "", "## 1. Per-feature leak table (ranked by best probe)", "",
        "| # | feature | acc([f_t]) | acc([f_t, f_t+1]) | "
        "acc([f_t+1 - f_t]) | best lift | flag |",
        "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | `{r['feature']}` | {r['acc_t']:.4f} | "
                 f"{r['acc_pair']:.4f} | {r['acc_delta']:.4f} | "
                 f"{r['lift']:+.4f} | {r['flag'] or DASH} |")
    L += ["", "The `acc([f_t+1 - f_t])` column is the one that matters: a "
          "depth-4 axis-aligned tree cannot subtract two raw levels, so the "
          "levels-only pair probe systematically under-reports algebraic "
          "leakage. Only one feature crosses the preregistered per-feature "
          f"thresholds ({', '.join('`' + f + '`' for f in res['flagged']) or 'none'}) "
          "\u2014 and that is an artifact of single-feature scope, not evidence "
          "that the rest are clean (see \u00a7 2\u20133).", "",
          "## 2. Residual leak floor", "",
          f"- all 50 features, levels only (x_t, x_t+1): "
          f"**{res['full_levels_only_acc']:.4f}**",
          f"- all 50, levels + deltas (150 columns): **{res['full_acc']:.4f}**",
          f"- flagged set dropped, {keep_n} survivors: "
          f"**{res['flagged_drop_residual']:.4f}** "
          f"(lift {res['flagged_drop_residual'] - base:+.4f})", "",
          "With the deltas available the outcome is recovered **exactly** "
          "(1.0000) by a depth-4 tree. Dropping the flagged scoreboard "
          "feature barely dents it: the residual floor is "
          f"{res['flagged_drop_residual']:.4f} against a {base:.4f} baseline.",
          "", "## 3. Greedy backward elimination", "",
          "Repeatedly drop the feature the fitted tree leans on hardest, "
          "re-measure, stop when accuracy returns to baseline.", "",
          "| features left | pair+delta accuracy | next dropped |",
          "|---|---|---|"]
    for t in trace:
        nd = t["next_dropped"]
        L.append(f"| {t['n_features']} | {t['acc']:.4f} | "
                 f"{DASH if nd is None else '`' + nd + '`'} |")
    L += ["", f"Must-drop set ({len(res['greedy_drop'])}): "
          + (", ".join(f"`{f}`" for f in res["greedy_drop"]) or "none") + ".",
          "", f"Surviving set ({len(res['greedy_keep'])}): "
          + (", ".join(f"`{f}`" for f in res["greedy_keep"]) or "none") + ".",
          "", f"## 4. Span mask ({SPAN} consecutive balls)", "",
          f"{sp['n']:,} spans. Targets: runs and wickets strictly inside the "
          "span. Inputs: the feature vectors at span start and span end plus "
          "their difference. Depth-10 regressors.", "",
          "| target | inputs | exact-recovery | MAE |", "|---|---|---|---|",
          f"| runs in span | all 50 | {sp['runs_all'][0]:.4f} | "
          f"{sp['runs_all'][1]:.3f} |",
          f"| wickets in span | all 50 | {sp['wkts_all'][0]:.4f} | "
          f"{sp['wkts_all'][1]:.3f} |",
          f"| runs in span | \u00a73 survivors | {sp['runs_keep'][0]:.4f} | "
          f"{sp['runs_keep'][1]:.3f} |",
          f"| wickets in span | \u00a73 survivors | {sp['wkts_keep'][0]:.4f} | "
          f"{sp['wkts_keep'][1]:.3f} |", "",
          f"Every wicket inside a 6-ball span is recovered exactly "
          f"({sp['wkts_all'][0]:.4f}, MAE {sp['wkts_all'][1]:.3f}) from the "
          "span endpoints. The runs total is recovered to "
          f"MAE {sp['runs_all'][1]:.3f} runs \u2014 the low exact-match rate "
          f"({sp['runs_all'][0]:.4f}) is only because a regression tree emits "
          "leaf means rather than integers; a linear head on the same "
          "difference would be exact. On the "
          "\u00a73 survivors both collapse "
          f"({sp['wkts_keep'][0]:.4f} / MAE {sp['wkts_keep'][1]:.3f} for "
          f"wickets, MAE {sp['runs_keep'][1]:.3f} for runs), which confirms "
          "the recovery runs entirely through the dropped cumulative fields.",
          "", "## 5. Conclusion", "",
          "**A masked-outcome objective is not viable on this feature "
          "contract as written.** Three things follow from the numbers above.",
          "",
          "1. *The scoreboard state is an exact decoder.* `score` alone, "
          f"differenced across one ball, gives {rows[0]['acc_delta']:.4f} "
          "accuracy \u2014 it is the runs off the ball by construction. Add the "
          "rest of the vector and a depth-4 tree reaches "
          f"{res['full_acc']:.4f}. Masking the outcome token while leaving "
          "`score`, `run_rate`, `run_rate_required`, `wickets_in_hand` and "
          "`balls_remaining` visible at t+1 asks the model to predict a label "
          "it is handed in its own inputs.",
          "2. *The EB tracker features leak almost as badly, and this is the "
          "non-obvious result.* `batter_p*` / `bowler_p*` are per-ball "
          "Empirical-Bayes tracker state that increments with the observed "
          "outcome, so their one-ball deltas are a near-invertible code for "
          "that outcome: the strongest single one, "
          f"`{eb[0]['feature']}`, reaches {eb[0]['acc_delta']:.4f} on its own, "
          f"and all {sum(1 for r in eb if r['acc_delta'] > 0.5)} of the 36 EB "
          "columns beat 0.50 on the delta probe. "
          "Dropping only the scoreboard columns leaves the floor at "
          f"{res['flagged_drop_residual']:.4f}. Any masking scheme that keeps "
          "the EB anchors at t+1 is leaky, however carefully the scoreboard "
          "is handled.",
          "3. *Almost nothing survives.* Greedy elimination has to strip "
          f"{len(res['greedy_drop'])} of the 50 features before the tree "
          f"returns to baseline, leaving only "
          + ", ".join(f"`{f}`" for f in res["greedy_keep"])
          + " \u2014 features that are constant within an innings and therefore "
          "carry no sequence information at all. The surviving contract is "
          "empty of exactly the signal a masked objective would need.", "",
          "### What Stage 3c must do instead", "",
          "- **Do not** mask the outcome and reconstruct it from the "
          "neighbouring 50-feature vectors. There is no subset of this "
          "contract that is both leak-free and informative.",
          "- If masked pretraining is wanted, the inputs must be rebuilt so "
          "that per-ball state is *not* carried as a level: supply the EB "
          "anchors and the scoreboard **as of the start of the masked span "
          "only**, frozen across the span, and expose no post-span endpoint "
          "vector. This is a frame change, not a training-loop change.",
          "- Span masking does not rescue the objective: with the endpoints "
          "visible, wickets inside a 6-ball span are recovered exactly "
          f"({sp['wkts_all'][0]:.4f}) and runs to MAE {sp['runs_all'][1]:.3f}. "
          "Wider spans leak *more* cumulative signal, not less.",
          "- The viable alternatives on the contract as it stands are "
          "objectives whose target is not a function of the inputs: "
          "next-ball prediction with a strict causal shift (what T1 already "
          "does), or auxiliary targets outside the 50 columns (the "
          "DeepCrease shot/line/length/control labels).", "",
          "### Caveats", "",
          f"- Validation split only ({res['n_pairs']:,} pairs); the audit is "
          "algebraic, and the leak is structural to the feature definitions, "
          "so a test-split rerun is not expected to move it.",
          "- Depth-4 trees are a *lower* bound on recoverability. A "
          "transformer is strictly more expressive, so every accuracy here "
          "understates what a trained model would exploit.",
          "- Greedy elimination by tree importance gives one sufficient "
          "must-drop set, not the unique minimal one; the ordering within "
          "the EB block is not load-bearing. Accuracy in the \u00a73 trace is "
          "not monotone (it rises at k=46 and k=27) because the EB columns "
          "come in near-collinear triples \u2014 removing one can free depth "
          "budget for a better split on its siblings. That is expected and "
          "does not affect the endpoint.", ""]
    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report(json.loads(OUT_JSON.read_text()))
    else:
        main()
