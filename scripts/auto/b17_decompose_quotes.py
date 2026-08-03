"""B17 Task 1 — decompose the in-play continuation P50 bias on EXISTING quotes.

Diagnostic only. Reads quote JSONs already produced by
`scripts/auto/b5_inplay_quotes.py` (B16's i7 run, B15's legacy run) and
decomposes the per-checkpoint P50 bias (sim_p50 - actual_remaining) into

  * per-checkpoint mean bias (must REPRODUCE the logged headline first),
  * bias per remaining over and per remaining legal ball,
  * paired segment rates (6->10, 10->15, 15->20) on matches carrying both
    endpoints — the over/phase profile of the deficit,
  * bias by wickets-at-checkpoint band and by score-at-checkpoint tercile.

No engine change, no sim run, no model load. Pure arithmetic on the
quote rows.

Row schema (from b5_inplay_quotes.py):
  match_id, file, checkpoint, runs_at_cp, wkts_at_cp, actual_final,
  actual_remaining, naive_remaining, sim_p10, sim_p50, sim_p90, sim_mean,
  sim_std, n_sims

Run:
    uv run python scripts/auto/b17_decompose_quotes.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

CHECKPOINTS = (6, 10, 15)
REMAINING_OVERS = {6: 14, 10: 10, 15: 5}
REMAINING_LEGAL_BALLS = {6: 84, 10: 60, 15: 30}

# Logged headlines to reproduce (research/results.tsv rows B16 / B15).
EXPECTED = {
    "i7": {6: -4.781, 10: -3.026, 15: -1.946},
    "legacy": {6: 4.259, 10: 2.777, 15: 0.410},
}
REPRO_TOL = 0.001


def load_rows(path: Path):
    payload = json.loads(path.read_text())
    return payload["config"], payload["rows"], payload.get("skips", [])


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def band_wkts(w):
    if w <= 2:
        return "0-2"
    if w <= 5:
        return "3-5"
    return "6+"


def decompose(label: str, path: Path, out_lines: list) -> dict:
    config, rows, skips = load_rows(path)
    out_lines.append("=" * 78)
    out_lines.append(f"STACK {label}   file {path}")
    out_lines.append("=" * 78)
    out_lines.append("config: " + json.dumps(config, sort_keys=True))
    out_lines.append(f"rows: {len(rows)}   "
                     f"matches: {len({r['match_id'] for r in rows})}   "
                     f"skips: {len(skips)}")
    out_lines.append("")

    res = {"label": label, "file": str(path), "config": config,
           "n_rows": len(rows), "n_matches": len({r["match_id"] for r in rows}),
           "n_skips": len(skips)}

    by_cp = {cp: [r for r in rows if r["checkpoint"] == cp] for cp in CHECKPOINTS}
    bias = {cp: [r["sim_p50"] - r["actual_remaining"] for r in by_cp[cp]]
            for cp in CHECKPOINTS}

    # ---------- 1. headline reproduction -----------------------------------
    out_lines.append("--- 1. per-checkpoint mean P50 bias (sim_p50 - actual_remaining) ---")
    out_lines.append(f"| {'cp':>3} | {'n':>4} | {'mean bias':>10} | {'sd':>8} | "
                     f"{'logged':>8} | {'|diff|':>8} | repro |")
    out_lines.append("|" + "-" * 5 + "|" + "-" * 6 + "|" + "-" * 12 + "|" + "-" * 10
                     + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 7 + "|")
    res["per_checkpoint"] = {}
    repro_all = True
    for cp in CHECKPOINTS:
        m, sd, n = mean_std(bias[cp])
        exp = EXPECTED.get(label, {}).get(cp)
        diff = abs(m - exp) if exp is not None else float("nan")
        ok = (exp is not None) and diff <= REPRO_TOL
        repro_all = repro_all and ok
        out_lines.append(f"| {cp:>3} | {n:>4} | {m:>+10.4f} | {sd:>8.3f} | "
                         f"{('' if exp is None else f'{exp:+.3f}'):>8} | "
                         f"{diff:>8.5f} | {'OK' if ok else 'MISMATCH':>5} |")
        res["per_checkpoint"][str(cp)] = {
            "n": n, "mean_bias": m, "sd_bias": sd,
            "logged": exp, "abs_diff": diff, "reproduced": bool(ok)}
    res["reproduced_all"] = bool(repro_all)
    out_lines.append(f"REPRODUCTION (tol {REPRO_TOL}): "
                     f"{'PASS' if repro_all else 'FAIL'}")
    out_lines.append("")

    # ---------- 2. bias per remaining over / per remaining legal ball ------
    out_lines.append("--- 2. bias normalised by remaining overs / remaining legal balls ---")
    out_lines.append(f"| {'cp':>3} | {'rem overs':>9} | {'rem legal':>9} | "
                     f"{'bias':>10} | {'per over':>10} | {'per legal ball':>14} |")
    out_lines.append("|" + "-" * 5 + "|" + "-" * 11 + "|" + "-" * 11 + "|"
                     + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 16 + "|")
    res["normalised"] = {}
    for cp in CHECKPOINTS:
        m = res["per_checkpoint"][str(cp)]["mean_bias"]
        ov = REMAINING_OVERS[cp]
        lb = REMAINING_LEGAL_BALLS[cp]
        out_lines.append(f"| {cp:>3} | {ov:>9} | {lb:>9} | {m:>+10.4f} | "
                         f"{m / ov:>+10.5f} | {m / lb:>+14.6f} |")
        res["normalised"][str(cp)] = {
            "remaining_overs": ov, "remaining_legal_balls": lb,
            "bias": m, "bias_per_over": m / ov, "bias_per_legal_ball": m / lb}
    out_lines.append("")

    # ---------- 3. paired segment rates -------------------------------------
    out_lines.append("--- 3. paired segment rates (runs/over of bias accrued in the segment) ---")
    idx = {cp: {r["match_id"]: r for r in by_cp[cp]} for cp in CHECKPOINTS}
    segments = [("6->10", 6, 10, 4), ("10->15", 10, 15, 5), ("15->20", 15, None, 5)]
    res["segments"] = {}
    out_lines.append(f"| {'segment':>8} | {'n paired':>8} | {'mean seg bias':>13} | "
                     f"{'per over':>10} | {'per legal ball':>14} |")
    out_lines.append("|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 15 + "|"
                     + "-" * 12 + "|" + "-" * 16 + "|")
    for name, a, b, n_ov in segments:
        if b is None:
            vals = [r["sim_p50"] - r["actual_remaining"] for r in by_cp[a]]
        else:
            common = sorted(set(idx[a]) & set(idx[b]))
            vals = [(idx[a][mid]["sim_p50"] - idx[a][mid]["actual_remaining"])
                    - (idx[b][mid]["sim_p50"] - idx[b][mid]["actual_remaining"])
                    for mid in common]
        m, sd, n = mean_std(vals)
        out_lines.append(f"| {name:>8} | {n:>8} | {m:>+13.4f} | "
                         f"{m / n_ov:>+10.5f} | {m / (6 * n_ov):>+14.6f} |")
        res["segments"][name] = {
            "n_paired": n, "mean_segment_bias": m, "sd": sd, "overs": n_ov,
            "per_over": m / n_ov, "per_legal_ball": m / (6 * n_ov)}
    out_lines.append("")

    # ---------- 4a. bias by wickets band -----------------------------------
    out_lines.append("--- 4a. bias by wickets-fallen-at-checkpoint band ---")
    out_lines.append(f"| {'cp':>3} | {'band':>5} | {'n':>4} | {'mean bias':>10} | "
                     f"{'per over':>10} | {'mean runs@cp':>12} |")
    out_lines.append("|" + "-" * 5 + "|" + "-" * 7 + "|" + "-" * 6 + "|" + "-" * 12
                     + "|" + "-" * 12 + "|" + "-" * 14 + "|")
    res["by_wickets_band"] = {}
    for cp in CHECKPOINTS:
        res["by_wickets_band"][str(cp)] = {}
        for bnd in ("0-2", "3-5", "6+"):
            sel = [r for r in by_cp[cp] if band_wkts(r["wkts_at_cp"]) == bnd]
            vals = [r["sim_p50"] - r["actual_remaining"] for r in sel]
            m, sd, n = mean_std(vals)
            mr = float(np.mean([r["runs_at_cp"] for r in sel])) if sel else float("nan")
            out_lines.append(f"| {cp:>3} | {bnd:>5} | {n:>4} | {m:>+10.4f} | "
                             f"{m / REMAINING_OVERS[cp]:>+10.5f} | {mr:>12.2f} |")
            res["by_wickets_band"][str(cp)][bnd] = {
                "n": n, "mean_bias": m, "sd": sd,
                "per_over": m / REMAINING_OVERS[cp], "mean_runs_at_cp": mr}
    out_lines.append("")

    # ---------- 4b. bias by score tercile -----------------------------------
    out_lines.append("--- 4b. bias by score-at-checkpoint tercile (terciles within checkpoint) ---")
    out_lines.append(f"| {'cp':>3} | {'tercile':>7} | {'runs@cp range':>15} | {'n':>4} | "
                     f"{'mean bias':>10} | {'per over':>10} |")
    out_lines.append("|" + "-" * 5 + "|" + "-" * 9 + "|" + "-" * 17 + "|" + "-" * 6
                     + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    res["by_score_tercile"] = {}
    for cp in CHECKPOINTS:
        scores = np.array([r["runs_at_cp"] for r in by_cp[cp]], dtype=float)
        q1, q2 = np.percentile(scores, [100 / 3.0, 200 / 3.0])
        res["by_score_tercile"][str(cp)] = {"cut_lo": float(q1), "cut_hi": float(q2)}
        for tname, sel in (
            ("T1", [r for r in by_cp[cp] if r["runs_at_cp"] <= q1]),
            ("T2", [r for r in by_cp[cp] if q1 < r["runs_at_cp"] <= q2]),
            ("T3", [r for r in by_cp[cp] if r["runs_at_cp"] > q2]),
        ):
            vals = [r["sim_p50"] - r["actual_remaining"] for r in sel]
            m, sd, n = mean_std(vals)
            rng = (f"{min(r['runs_at_cp'] for r in sel)}-"
                   f"{max(r['runs_at_cp'] for r in sel)}") if sel else "-"
            out_lines.append(f"| {cp:>3} | {tname:>7} | {rng:>15} | {n:>4} | "
                             f"{m:>+10.4f} | {m / REMAINING_OVERS[cp]:>+10.5f} |")
            res["by_score_tercile"][str(cp)][tname] = {
                "n": n, "range": rng, "mean_bias": m, "sd": sd,
                "per_over": m / REMAINING_OVERS[cp]}
    out_lines.append("")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i7-quotes",
                    default=str(REPO / "models/auto/b16/quotes_i7_s48_n261.json"))
    ap.add_argument("--legacy-quotes",
                    default=str(REPO / "models/auto/b15/quotes_s45_n261.json"))
    ap.add_argument("--out-json",
                    default=str(REPO / "models/auto/b17/decomposition.json"))
    ap.add_argument("--out-txt",
                    default=str(REPO / "research/handoff/B17/raw/decomposition.txt"))
    args = ap.parse_args()

    lines: list = []
    lines.append("B17 TASK 1 — in-play continuation P50 bias decomposition")
    lines.append("(diagnostic only; no engine change, no sim run)")
    lines.append("")

    payload = {"stacks": {}}
    for label, path_s in (("i7", args.i7_quotes), ("legacy", args.legacy_quotes)):
        path = Path(path_s)
        if not path.exists():
            lines.append(f"!! {label} quote file ABSENT: {path} — skipped")
            lines.append("")
            payload["stacks"][label] = {"absent": str(path)}
            continue
        payload["stacks"][label] = decompose(label, path, lines)

    # paired i7 - legacy contrast at row level (same seedless row keys)
    if all(k in payload["stacks"] and "per_checkpoint" in payload["stacks"][k]
           for k in ("i7", "legacy")):
        lines.append("=" * 78)
        lines.append("PAIRED CONTRAST  i7 - legacy  (same match_id/checkpoint rows)")
        lines.append("=" * 78)
        _, rows_i7, _ = load_rows(Path(args.i7_quotes))
        _, rows_lg, _ = load_rows(Path(args.legacy_quotes))
        ki7 = {(r["match_id"], r["checkpoint"]): r for r in rows_i7}
        klg = {(r["match_id"], r["checkpoint"]): r for r in rows_lg}
        common = sorted(set(ki7) & set(klg))
        lines.append(f"common rows: {len(common)}  "
                     f"(i7 {len(ki7)}, legacy {len(klg)})")
        lines.append(f"| {'cp':>3} | {'n':>4} | {'i7 bias':>9} | {'legacy bias':>11} | "
                     f"{'delta':>9} | {'delta/over':>10} | {'delta/legal ball':>16} |")
        lines.append("|" + "-" * 5 + "|" + "-" * 6 + "|" + "-" * 11 + "|" + "-" * 13
                     + "|" + "-" * 11 + "|" + "-" * 12 + "|" + "-" * 18 + "|")
        payload["paired_contrast"] = {}
        for cp in CHECKPOINTS:
            keys = [k for k in common if k[1] == cp]
            a = np.array([ki7[k]["sim_p50"] - ki7[k]["actual_remaining"] for k in keys])
            b = np.array([klg[k]["sim_p50"] - klg[k]["actual_remaining"] for k in keys])
            d = a - b
            lines.append(f"| {cp:>3} | {len(keys):>4} | {a.mean():>+9.4f} | "
                         f"{b.mean():>+11.4f} | {d.mean():>+9.4f} | "
                         f"{d.mean() / REMAINING_OVERS[cp]:>+10.5f} | "
                         f"{d.mean() / REMAINING_LEGAL_BALLS[cp]:>+16.6f} |")
            payload["paired_contrast"][str(cp)] = {
                "n": len(keys), "i7_mean_bias": float(a.mean()),
                "legacy_mean_bias": float(b.mean()),
                "delta_mean": float(d.mean()),
                "delta_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                "delta_per_over": float(d.mean() / REMAINING_OVERS[cp]),
                "delta_per_legal_ball": float(d.mean() / REMAINING_LEGAL_BALLS[cp])}
        lines.append("")

    txt = "\n".join(lines)
    print(txt)
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt + "\n")
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_txt}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
