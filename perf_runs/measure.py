"""Wall-clock + max-RSS wrapper for one subprocess.

Usage:
    uv run python perf_runs/measure.py <label> -- <argv...>

Prints:  LABEL=...  wall=...s  max_rss=...MB  exit=...
"""
import resource
import subprocess
import sys
import time

if "--" not in sys.argv:
    print("usage: measure.py <label> -- <argv...>", file=sys.stderr)
    sys.exit(2)

split = sys.argv.index("--")
label = " ".join(sys.argv[1:split])
argv = sys.argv[split + 1 :]

t0 = time.perf_counter()
proc = subprocess.run(argv)
elapsed = time.perf_counter() - t0

ru = resource.getrusage(resource.RUSAGE_CHILDREN)
# macOS: ru_maxrss is bytes; Linux: kilobytes. Detect by magnitude.
rss_bytes = ru.ru_maxrss if ru.ru_maxrss > 1_000_000 else ru.ru_maxrss * 1024
rss_mb = rss_bytes / (1024 * 1024)

print(
    f"\n[measure] LABEL={label!r}  wall={elapsed:.1f}s  "
    f"max_rss={rss_mb:.1f} MB  exit={proc.returncode}"
)
sys.exit(proc.returncode)
