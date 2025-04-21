#!/bin/bash
# Refresh atime on all files to avoid purge — minimal I/O version

ROOT_DIR="/pscratch/sd/m/maxvarv"
LOGFILE="$ROOT_DIR/.atime_refresh.log"
MAX_LOG_LINES=5  # Number of recent log lines to keep

echo "[$(date)] Starting atime refresh..." >> "$LOGFILE"

find "$ROOT_DIR" -type f -not -path '*/.*' -print0 | xargs -0 -n 1000 -P 8 \
  bash -c 'for f; do head -c 1 "$f" > /dev/null; done' _

echo "[$(date)] Finished atime refresh." >> "$LOGFILE"

# Keep only the most recent N lines in the log
tail -n $((2 * MAX_LOG_LINES)) "$LOGFILE" > "$LOGFILE.tmp" && mv "$LOGFILE.tmp" "$LOGFILE"

