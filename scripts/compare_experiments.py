"""
Experiment Comparison Tool — Compare experiment results side-by-side.

Usage:
    uv run python scripts/compare_experiments.py --list
    uv run python scripts/compare_experiments.py --list --tag xgboost
    uv run python scripts/compare_experiments.py --show exp_id_1
    uv run python scripts/compare_experiments.py exp_id_1 exp_id_2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_tracker import ExperimentTracker


def format_value(val, precision=4):
    """Format a value for display."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def format_delta(val1, val2, lower_is_better=True):
    """Format a delta between two values."""
    if val1 is None or val2 is None:
        return ""
    delta = val2 - val1
    pct = (delta / abs(val1) * 100) if val1 != 0 else 0
    sign = "+" if delta > 0 else ""
    # For metrics where lower is better, negative delta is good
    if lower_is_better:
        indicator = " *" if delta < 0 else ""
    else:
        indicator = " *" if delta > 0 else ""
    return f"[{sign}{pct:.1f}%]{indicator}"


def list_experiments(tag=None, results_dir="experiments/results"):
    """List all experiments, optionally filtered by tag."""
    experiments = ExperimentTracker.list_experiments(results_dir)

    if not experiments:
        print("No experiments found.")
        return

    if tag:
        experiments = [e for e in experiments if tag in e.get("tags", [])]
        if not experiments:
            print(f"No experiments found with tag '{tag}'.")
            return

    # Header
    print(f"\n{'ID':<45} {'Status':<10} {'Log Loss':<10} {'Brier':<10} {'Flat ROI':<10} {'FK ROI':<10} {'Win %':<8} {'Tags'}")
    print("-" * 130)

    for exp in experiments:
        exp_id = exp["id"]
        if len(exp_id) > 44:
            exp_id = exp_id[:41] + "..."
        status = exp.get("status", "?")
        log_loss = format_value(exp.get("avg_log_loss"), 4) if exp.get("avg_log_loss") else "—"
        brier = format_value(exp.get("avg_brier_score"), 4) if exp.get("avg_brier_score") else "—"
        flat_roi = f"{exp['flat_roi_pct']:.1f}%" if exp.get("flat_roi_pct") is not None else "—"
        fk_roi = f"{exp['frac_kelly_roi_pct']:.1f}%" if exp.get("frac_kelly_roi_pct") is not None else "—"
        win_rate = f"{exp['flat_win_rate_pct']:.1f}" if exp.get("flat_win_rate_pct") is not None else "—"
        tags = ", ".join(exp.get("tags", []))

        print(f"{exp_id:<45} {status:<10} {log_loss:<10} {brier:<10} {flat_roi:<10} {fk_roi:<10} {win_rate:<8} {tags}")

    print(f"\nTotal: {len(experiments)} experiment(s)")


def show_experiment(exp_id, results_dir="experiments/results"):
    """Show detailed info for a single experiment."""
    try:
        data = ExperimentTracker.load_experiment(exp_id, results_dir)
    except FileNotFoundError:
        # Try partial match
        experiments = ExperimentTracker.list_experiments(results_dir)
        matches = [e for e in experiments if exp_id in e["id"]]
        if len(matches) == 1:
            data = ExperimentTracker.load_experiment(matches[0]["id"], results_dir)
        elif len(matches) > 1:
            print(f"Ambiguous ID '{exp_id}'. Matches:")
            for m in matches:
                print(f"  {m['id']}")
            return
        else:
            print(f"Experiment not found: {exp_id}")
            return

    metadata = data.get("metadata", {})
    config = data.get("config", {})
    metrics = data.get("metrics", {})
    training = data.get("training_metrics", {})

    print(f"\n{'='*60}")
    print(f"  Experiment: {data['id']}")
    print(f"{'='*60}")

    print(f"\n--- Metadata ---")
    print(f"  Status:    {metadata.get('status', '?')}")
    print(f"  Started:   {metadata.get('started_at', '?')}")
    print(f"  Finished:  {metadata.get('finished_at', '?')}")
    print(f"  Duration:  {metadata.get('total_duration_seconds', '?')}s")
    print(f"  Git hash:  {metadata.get('git', {}).get('hash', '?')}")
    print(f"  Git branch:{metadata.get('git', {}).get('branch', '?')}")
    print(f"  Dirty:     {metadata.get('git', {}).get('dirty', '?')}")
    print(f"  Tags:      {metadata.get('tags', [])}")

    if config:
        print(f"\n--- Config ---")
        print(f"  Model type:  {config.get('model', {}).get('type', '?')}")
        features = config.get("features", {})
        groups = features.get("groups", [])
        exclude = features.get("exclude", [])
        print(f"  Feature groups: {', '.join(groups)}")
        if exclude:
            print(f"  Excluded: {', '.join(exclude)}")
        hyperparams = config.get("model", {}).get("hyperparameters", {})
        if hyperparams:
            print(f"  Hyperparameters:")
            for k, v in hyperparams.items():
                print(f"    {k}: {v}")

    if metrics:
        print(f"\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {format_value(v)}")

    if training:
        print(f"\n--- Training Metrics ---")
        for k, v in training.items():
            print(f"  {k}: {format_value(v)}")

    durations = metadata.get("step_durations", {})
    if durations:
        print(f"\n--- Step Durations ---")
        for step, dur in durations.items():
            print(f"  {step}: {dur}s")


def compare_experiments(exp_ids, results_dir="experiments/results"):
    """Compare two or more experiments side by side."""
    experiments = []
    for exp_id in exp_ids:
        try:
            data = ExperimentTracker.load_experiment(exp_id, results_dir)
            experiments.append(data)
        except FileNotFoundError:
            # Try partial match
            all_exps = ExperimentTracker.list_experiments(results_dir)
            matches = [e for e in all_exps if exp_id in e["id"]]
            if len(matches) == 1:
                data = ExperimentTracker.load_experiment(matches[0]["id"], results_dir)
                experiments.append(data)
            else:
                print(f"Experiment not found: {exp_id}")
                return

    if len(experiments) < 2:
        print("Need at least 2 experiments to compare.")
        return

    # Column width
    col_w = 30
    label_w = 25

    # Header
    names = [e.get("config", {}).get("experiment", {}).get("name", e["id"][:20])
             for e in experiments]
    print(f"\n{'':>{label_w}}", end="")
    for name in names:
        print(f"  {name:<{col_w}}", end="")
    print()
    print(f"{'':>{label_w}}", end="")
    for name in names:
        print(f"  {'─' * min(len(name), col_w):<{col_w}}", end="")
    print()

    # Rows: (label, getter, lower_is_better)
    rows = [
        ("Model Type", lambda e: e.get("config", {}).get("model", {}).get("type", "?"), False),
        ("Status", lambda e: e.get("metadata", {}).get("status", "?"), False),
        ("Git Hash", lambda e: e.get("metadata", {}).get("git", {}).get("hash", "?"), False),
        ("Features", lambda e: len(e.get("config", {}).get("features", {}).get("groups", [])), False),
        ("Matches", lambda e: e.get("metrics", {}).get("matches_evaluated"), False),
        # --- Probability metrics (lower is better) ---
        ("Avg Log Loss", lambda e: e.get("metrics", {}).get("avg_log_loss"), True),
        ("Avg Brier Score", lambda e: e.get("metrics", {}).get("avg_brier_score"), True),
        ("Avg Edge %", lambda e: e.get("metrics", {}).get("avg_edge_pct"), True),
        ("Avg Signed Edge %", lambda e: e.get("metrics", {}).get("avg_signed_edge_pct"), False),
        # --- Flat staking ---
        ("Flat ROI %", lambda e: e.get("metrics", {}).get("flat_roi_pct"), False),
        ("Flat Win Rate %", lambda e: e.get("metrics", {}).get("flat_win_rate_pct"), False),
        ("Flat P&L", lambda e: e.get("metrics", {}).get("flat_pnl"), False),
        # --- Fractional Kelly ---
        ("Frac Kelly ROI %", lambda e: e.get("metrics", {}).get("frac_kelly_roi_pct"), False),
        ("Frac Kelly Win %", lambda e: e.get("metrics", {}).get("frac_kelly_win_rate_pct"), False),
        ("Frac Kelly P&L", lambda e: e.get("metrics", {}).get("frac_kelly_pnl"), False),
        # --- Full Kelly ---
        ("Full Kelly ROI %", lambda e: e.get("metrics", {}).get("full_kelly_roi_pct"), False),
        # --- Meta ---
        ("Duration (s)", lambda e: e.get("metadata", {}).get("total_duration_seconds"), True),
    ]

    for label, getter, lower_is_better in rows:
        values = [getter(e) for e in experiments]
        print(f"{label:>{label_w}}", end="")
        for i, val in enumerate(values):
            formatted = format_value(val)
            if i > 0 and values[0] is not None and val is not None:
                try:
                    delta = format_delta(float(values[0]), float(val), lower_is_better)
                    formatted = f"{formatted} {delta}"
                except (ValueError, TypeError):
                    pass
            print(f"  {formatted:<{col_w}}", end="")
        print()

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("experiments", nargs="*", help="Experiment IDs to compare")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--tag", type=str, default=None, help="Filter by tag (with --list)")
    parser.add_argument("--show", type=str, default=None, help="Show single experiment details")
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                        help="Results directory")
    args = parser.parse_args()

    if args.list:
        list_experiments(args.tag, args.results_dir)
    elif args.show:
        show_experiment(args.show, args.results_dir)
    elif args.experiments:
        if len(args.experiments) == 1:
            show_experiment(args.experiments[0], args.results_dir)
        else:
            compare_experiments(args.experiments, args.results_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
