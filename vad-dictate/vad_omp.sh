#!/usr/bin/env bash
# vad_omp.sh — omp-scoped VAD supervisor. Daemon lives only while omp lives.
# Usage:
#   ./vad_omp.sh            auto-discover omp pane (prefers same window), refuse self
#   ./vad_omp.sh %2         pin explicit pane (exits if that pane dies, never retargets)
#   ./vad_omp.sh --list     print omp panes and exit
# Pause: touch ~/.vad-paused / rm ~/.vad-paused (or say "dictation pause"/"resume")
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCK="$HOME/.vad-omp.lock"
LOG="$HERE/vad-dictate.log"
MIC="${VAD_DEVICE:-@DEFAULT_SOURCE@}"
SIDECAR="http://localhost:3005/transcribe"
export PULSE_SERVER="${PULSE_SERVER:-tcp:localhost:4713}"


SELF=$(tmux display-message -p '#{pane_id}' 2>/dev/null || echo "")
SELF_WIN=$(tmux display-message -p '#{window_id}' 2>/dev/null || echo "")

# All omp panes: walk each `bun .../omp` pid up to its tmux pane.
find_all_omp_panes() {
  local map child p pane
  map=$(tmux list-panes -a -F '#{pane_pid} #{pane_id}' 2>/dev/null) || return 1
  for child in $(pgrep -f '\.bun/bin/omp' 2>/dev/null); do
    # skip helpers (worker daemons are children of omp, walk still lands on pane;
    # dedupe by pane below)
    p=$child
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
      pane=$(echo "$map" | awk -v pid="$p" '$1==pid {print $2; exit}')
      if [ -n "$pane" ]; then echo "$pane pid=$child"; break; fi
      p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    done
  done | awk '{print $1}' | sort -Vu
}

resolve_auto() {
  local all n pane w winmap first
  all=$(find_all_omp_panes)
  [ -n "$all" ] || return 1
  n=$(echo "$all" | wc -l)
  if [ "$n" -eq 1 ]; then echo "$all"; return 0; fi
  winmap=$(tmux list-panes -a -F '#{pane_id} #{window_id}')
  for pane in $all; do
    w=$(echo "$winmap" | awk -v p="$pane" '$1==p {print $2; exit}')
    if [ -n "$w" ] && [ "$w" = "$SELF_WIN" ]; then echo "$pane"; return 0; fi
  done
  first=$(echo "$all" | head -1)
  echo "vad_omp: $n omp panes, using $first (override: $0 <pane>):" >&2
  echo "$all" | sed 's/^/vad_omp:   /' >&2
  echo "$first"
}

validate_target() {
  local t=$1
  if [ -n "$SELF" ] && [ "$t" = "$SELF" ]; then
    echo "vad_omp: refusing target $t — that's this supervisor pane, not omp" >&2
    return 1
  fi
  tmux display-message -t "$t" -p '#{pane_id} cmd=#{pane_current_command}' 2>&1 || return 1
  if ! find_all_omp_panes | awk -v pane="$t" '$1==pane {found=1} END {exit !found}'; then
    echo "vad_omp: target $t has no omp process" >&2
    return 1
  fi
  return 0
}

if [ "${1:-}" = "--list" ]; then
  printf '%-6s %-7s %s\n' PANE STATE SESSION
  find_all_omp_panes | while read -r pane; do
    title=$(tmux display-message -t "$pane" -p '#{pane_title}' 2>/dev/null) || continue
    title=${title#π }
    if [[ $title == [⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]' '* ]]; then
      state=working
      title=${title#? }
    else
      state=idle
      title=${title#> }
    fi
    printf '%-6s %-7s %s\n' "$pane" "$state" "$title"
  done
  [ -z "$SELF" ] || printf '\nCurrent pane: %s\n' "$SELF"
  exit 0
fi
exec 9>"$LOCK"
flock -n 9 || { echo "vad_omp: already running"; exit 1; }

FIXED=0
if [ $# -ge 1 ]; then FIXED=1; PANE=$1; else PANE=""; fi

CHILD=""
cleanup() { [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; exit 0; }
trap cleanup INT TERM

BACKOFF=5
while true; do
  if [ -z "$PANE" ]; then
    PANE=$(resolve_auto) || { echo "vad_omp: omp not running, exiting" >&2; exit 1; }
  fi
  validate_target "$PANE" || {
    if [ "$FIXED" -eq 1 ]; then echo "vad_omp: bad target, exiting (no retarget on explicit pane)"; exit 1; fi
    PANE=""; sleep 2; continue
  }
  echo "vad_omp: omp at $PANE (self $SELF), starting daemon (log $LOG)"
  PULSE_SERVER="$PULSE_SERVER" python3 "$HERE/vad_dictate.py" \
    --target "$PANE" --device "$MIC" --sidecar "$SIDECAR" --log "$LOG" &
  CHILD=$!
  # watchdog: target pane + omp liveness while daemon runs
  while kill -0 "$CHILD" 2>/dev/null; do
    sleep 5
    if ! validate_target "$PANE" >/dev/null 2>&1; then
      echo "vad_omp: target $PANE gone — killing daemon"
      kill "$CHILD" 2>/dev/null
      if [ "$FIXED" -eq 1 ]; then wait "$CHILD" 2>/dev/null; echo "vad_omp: explicit target gone, exiting"; exit 1; fi
      PANE=""
      break
    fi
    if ! pgrep -f '\.bun/bin/omp' >/dev/null 2>&1; then
      echo "vad_omp: no omp process — stopping"
      kill "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; exit 0
    fi
  done
  wait "$CHILD" 2>/dev/null; RC=$?
  CHILD=""
  if [ -n "$PANE" ]; then
    # daemon died on its own (rc=2 means target lost → re-resolve when auto)
    if [ "$RC" -eq 2 ] && [ "$FIXED" -eq 0 ]; then PANE=""; sleep 2; continue; fi
    if ! pgrep -f '\.bun/bin/omp' >/dev/null 2>&1; then echo "vad_omp: omp gone, exiting"; exit 0; fi
    echo "vad_omp: daemon exited rc=$RC, restart in ${BACKOFF}s"
    sleep "$BACKOFF"
    BACKOFF=$(( BACKOFF < 30 ? BACKOFF + 5 : 30 ))
  else
    sleep 2  # target was cleared above; immediate re-resolve
  fi
done
