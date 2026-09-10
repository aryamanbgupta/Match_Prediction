#!/usr/bin/env bash
# Install (or reinstall) the CricML daily launchd agent on the Mac mini.
# Refuses if free disk is under 5 GB (plan item 3 step 8, check C23).
set -euo pipefail
cd "$(dirname "$0")/../../.."
free_kb=$(df -k "$HOME" | awk 'NR==2 {print $4}')
if [ "$free_kb" -lt $((5 * 1024 * 1024)) ]; then echo "install: under 5 GB free, refusing" >&2; exit 1; fi
mkdir -p daily/logs "$HOME/Library/LaunchAgents"
cp scripts/daily/launchd/com.cricml.daily.plist "$HOME/Library/LaunchAgents/com.cricml.daily.plist"
launchctl unload "$HOME/Library/LaunchAgents/com.cricml.daily.plist" 2>/dev/null || true
launchctl load "$HOME/Library/LaunchAgents/com.cricml.daily.plist"
launchctl list | grep com.cricml.daily && echo "installed: com.cricml.daily (30-minute cadence, t60 only)"
