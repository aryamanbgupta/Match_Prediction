"""Regression test for review finding PIPE1 (2026-08-14).

`build_stats_cache.main` used to return on the sqlite-up-to-date fast path
BEFORE applying `--prior-source-sqlite`, and `sqlite_up_to_date` does not
inspect prior provenance — so a rebuilt prior source never propagated into
an otherwise-current cache (run_experiment's provenance check would miss,
re-invoke the builder, and hit the same skip: no self-heal). The freeze must
run on the skip path too.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import build_stats_cache as bsc  # noqa: E402


def _run_main(monkeypatch, tmp_path, extra_argv, freeze_calls, build_calls):
    monkeypatch.setattr(bsc, "sqlite_up_to_date", lambda *a, **k: True)

    def _record_freeze(out, src):
        freeze_calls.append((Path(out), Path(src)))
        return {
            "prior_source_sqlite": str(src),
            "prior_source_sha256": "0" * 64,
        }

    monkeypatch.setattr(bsc, "freeze_priors_from_sqlite", _record_freeze)
    monkeypatch.setattr(
        bsc, "build", lambda *a, **k: build_calls.append(1)
    )
    monkeypatch.setattr(sys, "argv", [
        "build_stats_cache.py",
        "--source-dir", str(tmp_path / "corpus"),
        "--out", str(tmp_path / "cache.sqlite"),
        *extra_argv,
    ])
    return bsc.main()


def test_skip_path_still_freezes_priors(monkeypatch, tmp_path):
    freeze_calls, build_calls = [], []
    rc = _run_main(
        monkeypatch, tmp_path,
        ["--prior-source-sqlite", str(tmp_path / "prior.sqlite")],
        freeze_calls, build_calls,
    )
    assert rc == 0
    assert not build_calls, "up-to-date cache must not rebuild"
    assert freeze_calls == [
        (tmp_path / "cache.sqlite", tmp_path / "prior.sqlite"),
    ], "the skip path must still re-apply the prior freeze"


def test_skip_path_without_prior_source_does_not_freeze(monkeypatch, tmp_path):
    freeze_calls, build_calls = [], []
    rc = _run_main(monkeypatch, tmp_path, [], freeze_calls, build_calls)
    assert rc == 0
    assert not build_calls
    assert freeze_calls == []
