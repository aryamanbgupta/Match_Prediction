#!/usr/bin/env bash
# Deterministic overnight queue runner for the sequence and embeddings track.
#
# Every arm is pre-registered in a queue file, so unattended work is a plain
# queue: jobs in listed order, one at a time, each with a config hash, an
# output dir, an expected wall time and a machine tag. There is no model in
# the loop, no verdict logging, and no version-control command anywhere in
# this file or its helper -- the runner only launches the commands it was
# given and writes marker files.
#
# Usage:
#   ./research/sequence_track/run_queue.sh [--queue PATH] [--machine NAME] [--dry-run]
#
# Stop: touch the queue's stop_file (default research/sequence_track/STOP).
#       It is checked before each job launch, never mid-job; removing it and
#       re-running resumes, because finished jobs are skipped by config hash.
#
# Markers written into each job's output_dir:
#   COMPLETE        job exited 0; holds config_sha256 and the end stamp
#   TIMEOUT         killed by the per-job alarm (expected_hours * 2), or by
#                   the watchdog deadline standing behind it
#   KILLED_MEMORY   killed by the RSS watchdog; holds the observed RSS
#   REFUSED_MEMORY  not launched: available memory below memory_floor_gb, or
#                   the reading itself failed. Checked before every attempt,
#                   so a retry can be refused after the first attempt ran; a
#                   refusal writes no FAILED marker and the queue continues.
#   FAILED          failed twice (the runner retries once, then moves on)
#
# Logs: each job's stdout and stderr tee to output_dir/run.log, appended
#       across retries, with start / exit-code / end stamps.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUEUE_LIB="$REPO_ROOT/scripts/sequence_track/queue_lib.py"

QUEUE="$SCRIPT_DIR/queue_laptop.yaml"
MACHINE="laptop"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./research/sequence_track/run_queue.sh [--queue PATH] [--machine NAME] [--dry-run]

  --queue PATH    queue file (default:
                  research/sequence_track/queue_laptop.yaml; the stage-2
                  night is split by seed, so the mini half is
                  research/sequence_track/queue_mini.yaml)
  --machine NAME  run only jobs tagged with this machine (default: laptop)
  --dry-run       print the resolved jobs, each skip/run decision and the
                  memory reading, then exit without launching anything
  -h, --help      show this help

Exit status: 2 on bad arguments or a queue that fails schema validation
(nothing is launched); 0 otherwise, including when individual jobs fail --
a failing job is marked FAILED and the queue continues.
EOF
}

require_value() {
  # $1 = flag name, $2 = value (may be empty)
  if [[ -z "${2:-}" ]]; then
    echo "run_queue.sh: $1 requires a value" >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --queue)     require_value --queue "${2:-}"; QUEUE="$2"; shift ;;
    --queue=*)   QUEUE="${1#*=}"; require_value --queue "$QUEUE" ;;
    --machine)   require_value --machine "${2:-}"; MACHINE="$2"; shift ;;
    --machine=*) MACHINE="${1#*=}"; require_value --machine "$MACHINE" ;;
    --dry-run)   DRY_RUN=1 ;;
    -h|--help)   usage; exit 0 ;;
    *)
      echo "run_queue.sh: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log()   { printf '[%s] run_queue: %s\n' "$(stamp)" "$*"; }

PERL_BIN="$(command -v perl || true)"
if [[ -z "$PERL_BIN" ]]; then
  echo "run_queue.sh: perl not found; it provides the per-job timeout" >&2
  exit 2
fi
# caffeinate is macOS-only; without it jobs still run, the machine may sleep.
CAFFEINATE_BIN="$(command -v caffeinate || true)"

# Jobs run from the repo root so queue commands can use repo-relative paths.
cd "$REPO_ROOT" || exit 2

TSV="$(mktemp -t run_queue)" || exit 2
trap 'rm -f "$TSV"' EXIT

# Schema validation lives in the helper and happens before anything launches.
if ! uv run --no-sync python "$QUEUE_LIB" resolve --queue "$QUEUE" > "$TSV"; then
  echo "run_queue.sh: queue validation failed, nothing launched: $QUEUE" >&2
  exit 2
fi

MEM_FLOOR_GB=""
MEM_CAP_GB=""
POLL_SECONDS=""
STOP_FILE=""
JOB_COUNT=0

while IFS=$'\t' read -r kind field value _rest; do
  case "$kind" in
    DEFAULT)
      case "$field" in
        memory_floor_gb) MEM_FLOOR_GB="$value" ;;
        memory_cap_gb)   MEM_CAP_GB="$value" ;;
        poll_seconds)    POLL_SECONDS="$value" ;;
        stop_file)       STOP_FILE="$value" ;;
      esac
      ;;
    WARN) log "warning: $field" ;;
    JOB)  JOB_COUNT=$(( JOB_COUNT + 1 )) ;;
  esac
done < "$TSV"

log "queue: $QUEUE"
log "machine: $MACHINE"
log "jobs listed: $JOB_COUNT"
log "memory floor: ${MEM_FLOOR_GB} GiB; memory cap: ${MEM_CAP_GB} GiB; poll: ${POLL_SECONDS}s"
log "stop file: $STOP_FILE"

MEM_LINE="$(uv run --no-sync python "$QUEUE_LIB" mem --floor-gb "$MEM_FLOOR_GB" 2>&1 < /dev/null)"
log "$MEM_LINE"

if [[ -e "$STOP_FILE" ]]; then
  if (( DRY_RUN )); then
    log "note: stop file present; a real run would exit here without launching"
  else
    log "stop file present; exiting without launching anything"
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

while IFS=$'\t' read -r kind job_id job_machine job_hours job_timeout \
                        job_config job_sha job_outdir job_command; do
  [[ "$kind" == "JOB" ]] || continue

  decision="run"
  reason="no COMPLETE marker"
  complete_marker="$job_outdir/COMPLETE"

  if [[ "$job_machine" != "$MACHINE" ]]; then
    decision="skip"
    reason="machine '$job_machine' does not match '$MACHINE'"
  # Skip-if-complete: a finished job with an unchanged config is not re-run.
  elif [[ -f "$complete_marker" ]]; then
    recorded_sha="$(sed -n 's/^config_sha256: *//p' "$complete_marker" | head -1)"
    if [[ "$recorded_sha" == "$job_sha" ]]; then
      decision="skip"
      reason="COMPLETE with matching config_sha256"
    else
      reason="COMPLETE present but config changed (was ${recorded_sha:-none})"
    fi
  fi

  if (( DRY_RUN )); then
    printf '  %-24s machine=%-8s decision=%-4s timeout=%ss config=%s sha=%s\n' \
      "$job_id" "$job_machine" "$decision" "$job_timeout" "$job_config" "$job_sha"
    printf '  %-24s output_dir=%s\n' "" "$job_outdir"
    printf '  %-24s command=%s\n' "" "$job_command"
    printf '  %-24s reason=%s\n' "" "$reason"
    continue
  fi

  if [[ "$decision" == "skip" ]]; then
    log "skip '$job_id': $reason"
    continue
  fi

  # The stop file is checked between jobs, never mid-job.
  if [[ -e "$STOP_FILE" ]]; then
    log "stop file present ($STOP_FILE); exiting before launching '$job_id'"
    exit 0
  fi

  mkdir -p "$job_outdir" || { log "cannot create output dir for '$job_id'"; continue; }
  job_log="$job_outdir/run.log"
  rm -f "$job_outdir/COMPLETE" "$job_outdir/TIMEOUT" "$job_outdir/FAILED" \
        "$job_outdir/KILLED_MEMORY" "$job_outdir/REFUSED_MEMORY"

  attempt=1
  while :; do
    # Memory floor, freshly read before every attempt: the retry can land in a
    # machine state the first attempt never saw (and may itself be why memory
    # is gone). A reading that cannot be taken refuses the launch too -- an
    # unreadable machine is not evidence that memory is free.
    mem_line="$(uv run --no-sync python "$QUEUE_LIB" mem --floor-gb "$MEM_FLOOR_GB" 2>&1 < /dev/null)"
    mem_rc=$?
    log "'$job_id' attempt $attempt $mem_line"
    printf '=== job %s attempt %s memory check %s :: %s ===\n' \
      "$job_id" "$attempt" "$(stamp)" "$mem_line" >> "$job_log"
    if (( mem_rc != 0 )); then
      if (( mem_rc == 3 )); then
        refusal="available memory below the ${MEM_FLOOR_GB} GiB floor"
      else
        refusal="memory reading failed (rc=$mem_rc)"
      fi
      {
        printf 'job_id: %s\n' "$job_id"
        printf 'attempt: %s\n' "$attempt"
        printf 'memory_floor_gb: %s\n' "$MEM_FLOOR_GB"
        printf 'refusal: %s\n' "$refusal"
        printf 'reading: %s\n' "$mem_line"
        printf 'refused: %s\n' "$(stamp)"
      } > "$job_outdir/REFUSED_MEMORY"
      # A refusal is not a failed run: no FAILED marker, no further retry, and
      # the queue continues. On a retry the earlier attempt's marker is left in
      # place, so the output dir shows both what happened and why it stopped.
      log "refusing '$job_id' attempt $attempt: $refusal; wrote REFUSED_MEMORY"
      break
    fi

    rm -f "$job_outdir/TIMEOUT" "$job_outdir/KILLED_MEMORY"
    log "launch '$job_id' attempt $attempt (timeout ${job_timeout}s, cap ${MEM_CAP_GB} GiB)"
    {
      printf '=== job %s attempt %s start %s ===\n' "$job_id" "$attempt" "$(stamp)"
      printf '=== job %s attempt %s command: %s ===\n' "$job_id" "$attempt" "$job_command"
    } >> "$job_log"

    # Job control puts the job in its own process group, so the watchdog can
    # see and kill the whole tree. pipefail makes the subshell's status the
    # job's status rather than tee's.
    set -m
    (
      set -o pipefail
      if [[ -n "$CAFFEINATE_BIN" ]]; then
        "$CAFFEINATE_BIN" -i "$PERL_BIN" -e 'alarm shift; exec @ARGV' \
          "$job_timeout" /bin/bash -c "$job_command" 2>&1 | tee -a "$job_log"
      else
        "$PERL_BIN" -e 'alarm shift; exec @ARGV' \
          "$job_timeout" /bin/bash -c "$job_command" 2>&1 | tee -a "$job_log"
      fi
    ) < /dev/null &
    job_pid=$!
    set +m

    # The watchdog enforces both guards. The alarm above only signals the one
    # process it exec'd, so a job that backgrounds a child (or that handles
    # SIGALRM itself) survives it and holds the log pipe open; the watchdog's
    # deadline kills the whole group and writes TIMEOUT in that case.
    uv run --no-sync python "$QUEUE_LIB" watch \
      --pid "$job_pid" --cap-gb "$MEM_CAP_GB" --poll "$POLL_SECONDS" \
      --marker "$job_outdir/KILLED_MEMORY" \
      --timeout-seconds "$job_timeout" --timeout-marker "$job_outdir/TIMEOUT" \
      --job-id "$job_id" \
      < /dev/null >> "$job_log" 2>&1 &
    watch_pid=$!

    # stderr is dropped only to silence bash's job-control notice ("Alarm
    # clock", "Terminated"); the same facts are in the exit-code stamp below.
    wait "$job_pid" 2>/dev/null
    rc=$?
    kill "$watch_pid" 2>/dev/null
    wait "$watch_pid" 2>/dev/null
    # A timeout or a failure can leave stray descendants in the job's process
    # group (the alarm only reaches the process that was exec'd).
    (( rc != 0 )) && kill -TERM -"$job_pid" 2>/dev/null

    printf '=== job %s attempt %s exit=%s end %s ===\n' \
      "$job_id" "$attempt" "$rc" "$(stamp)" >> "$job_log"

    if (( rc == 0 )); then
      {
        printf 'job_id: %s\n' "$job_id"
        printf 'config: %s\n' "$job_config"
        printf 'config_sha256: %s\n' "$job_sha"
        printf 'exit_code: 0\n'
        printf 'attempts: %s\n' "$attempt"
        printf 'end: %s\n' "$(stamp)"
      } > "$complete_marker"
      log "'$job_id' COMPLETE (attempt $attempt)"
      break
    fi

    if [[ -f "$job_outdir/KILLED_MEMORY" ]]; then
      failure="killed by the memory watchdog"
    elif [[ -f "$job_outdir/TIMEOUT" ]]; then
      # Written by the watchdog when the alarm failed to reach every process.
      failure="timed out after ${job_timeout}s (killed by the watchdog)"
    elif (( rc == 142 )); then
      {
        printf 'job_id: %s\n' "$job_id"
        printf 'timeout_seconds: %s\n' "$job_timeout"
        printf 'attempt: %s\n' "$attempt"
        printf 'end: %s\n' "$(stamp)"
      } > "$job_outdir/TIMEOUT"
      failure="timed out after ${job_timeout}s"
    else
      failure="exit code $rc"
    fi

    if (( attempt == 1 )); then
      log "'$job_id' attempt 1 failed ($failure); retrying once"
      attempt=2
      continue
    fi

    {
      printf 'job_id: %s\n' "$job_id"
      printf 'config_sha256: %s\n' "$job_sha"
      printf 'exit_code: %s\n' "$rc"
      printf 'attempts: %s\n' "$attempt"
      printf 'failure: %s\n' "$failure"
      printf 'end: %s\n' "$(stamp)"
    } > "$job_outdir/FAILED"
    log "'$job_id' FAILED after $attempt attempts ($failure); continuing with the queue"
    break
  done
done < "$TSV"

if (( DRY_RUN )); then
  log "dry run complete; nothing launched"
else
  log "queue complete"
fi
exit 0
