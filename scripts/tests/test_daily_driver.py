from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).parents[2]
MINI = ROOT / "tests" / "fixtures" / "cricsheet_mini"
GAMMA = Path(__file__).parent / "fixtures" / "gamma_recorded" / "open_t20.json"


def test_daily_driver_executes_offline_end_to_end(tmp_path):
    data = tmp_path / "data"
    base = data / "t20s_json"
    shutil.copytree(MINI, base)
    daily = tmp_path / "daily"
    state_link = data / "live_state_i7"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=60)
    run_date = start.date().isoformat()
    gamma = json.loads(GAMMA.read_text())
    gamma["events"][0]["startTime"] = start.isoformat()
    gamma_path = tmp_path / "gamma.json"
    gamma_path.write_text(json.dumps(gamma))

    result_match = json.loads((MINI / "710019.json").read_text())
    result_match["info"]["dates"] = [run_date]
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result_match))
    fetch_stub = tmp_path / "fetch_stub.py"
    fetch_stub.write_text(
        "import os, shutil\n"
        f"shutil.copyfile({str(result_path)!r}, "
        "os.path.join(os.environ['CRICML_CONTEXT_DIR'], 'new-result.json'))\n"
    )

    # The driver itself is real. This uv shim delegates ordinary daily steps,
    # stubs only the expensive state builders and model inference, and lets
    # predict_daily's real locked writer create the protocol line.
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pickle, sqlite3, subprocess, sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "assert args[:3] == ['run','--no-sync','python'], args\n"
        "script, rest = args[3], args[4:]\n"
        "def values(flag):\n"
        " return [Path(rest[i+1]) for i,v in enumerate(rest) if v == flag]\n"
        "if script.endswith('build_stats_cache.py'):\n"
        " sources=values('--source-dir')+values('--extra-source-dir')\n"
        " count=sum(len(list(p.glob('*.json'))) for p in sources)\n"
        " out=values('--out')[0]; out.parent.mkdir(parents=True,exist_ok=True)\n"
        " with sqlite3.connect(out) as conn:\n"
        "  conn.execute('CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)')\n"
        "  conn.execute(\"INSERT INTO _meta VALUES ('source_match_count', ?)\",(str(count),))\n"
        "elif script.endswith('predict_fixture.py') and '--rebuild-snapshot' in rest:\n"
        " sources=values('--tracker-source-dir')\n"
        " count=sum(len(list(p.glob('*.json'))) for p in sources)\n"
        " snapshot=values('--tracker-snapshot')[0]\n"
        " with snapshot.open('wb') as h: pickle.dump({'n_matches_walked':count},h)\n"
        " values('--out')[0].write_text('{}')\n"
        "elif script.endswith('daily/predict_daily.py'):\n"
        " sys.path.insert(0,str(Path(script).resolve().parents[1]))\n"
        " from daily.predict_daily import append_prediction, make_prediction_line\n"
        " fixtures=[json.loads(x) for x in values('--fixtures')[0].read_text().splitlines()]\n"
        " lineups=[json.loads(x) for x in values('--lineups')[0].read_text().splitlines()]\n"
        " by_id={x['fixture_id']:x for x in lineups}\n"
        " for fixture in fixtures:\n"
        "  prediction={'prediction':{fixture['team1']:0.6,fixture['team2']:0.4}}\n"
        "  row=make_prediction_line(fixture,by_id[fixture['fixture_id']],prediction,run_kind='t60',model_role='match_model_prod',model_md5='a'*32,state_as_of=fixture['scheduled_start'][:10])\n"
        "  append_prediction(values('--out')[0],row)\n"
        " print(len(fixtures))\n"
        "else:\n"
        " raise SystemExit(subprocess.run([sys.executable,script,*rest]).returncode)\n"
    )
    fake_uv.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "RUN_DATE": run_date,
        "DAILY_DIR": str(daily),
        "BASE_SOURCE_DIR": str(base),
        "CONTEXT_ROOT": str(daily / "context"),
        "STATE_PARENT": str(data),
        "STATE_LINK": str(state_link),
        "STATE_DIR": str(state_link),
        "MODEL_DIR": str(tmp_path / "model"),
        "GAMMA_RESPONSE": str(gamma_path),
        "DAILY_WRITER_KEY": str(daily / ".writer_key"),
        "CRICSHEET_FETCH_CMD": f"{shlex.quote(sys.executable)} {shlex.quote(str(fetch_stub))}",
    }
    subprocess.run(
        ["sh", str(ROOT / "scripts" / "daily" / "run_daily.sh")],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )

    fixtures = [json.loads(line) for line in
                (daily / "fixtures" / f"{run_date}.jsonl").read_text().splitlines()]
    predictions = [json.loads(line) for line in
                   (daily / "predictions.jsonl").read_text().splitlines()]
    settlements = [json.loads(line) for line in
                   (daily / "settled.jsonl").read_text().splitlines()]
    score_paths = list((daily / "scores").glob("score_v1_*.json"))
    assert len(fixtures) == len(predictions) == len(settlements) == len(score_paths) == 1
    assert {"fixture_id", "scheduled_start", "quote", "quote_ts"} <= fixtures[0].keys()
    assert {"cohort_id", "attempt_ts", "state_as_of", "p_team1"} <= predictions[0].keys()
    assert {"fixture_id", "revision", "winner", "void"} <= settlements[0].keys()
    artifact = json.loads(score_paths[0].read_text())
    assert artifact["status"] == "descriptive"
    assert artifact["n_scored"] == 1
    assert artifact["protocol_version"] == 1
