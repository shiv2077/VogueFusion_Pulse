#!/usr/bin/env bash
# Start, stop and inspect the VogueFusion processes on this machine.
#
#   scripts/vf.sh start [component...]   start (default: all), skipping ones already running
#   scripts/vf.sh stop  [component...]   stop (default: all)
#   scripts/vf.sh status                 show what is running
#
# Components, in start order: comfyui api worker-generate worker-cost
# The API also serves the web app at http://<api host>:<api port>/.
# Logs and pid files live in $VOGUEFUSION_DATA/{logs,run} (default ~/voguefusion-data).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${VOGUEFUSION_DATA:-$HOME/voguefusion-data}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
VF_PY="$CONDA_ROOT/envs/voguefusion/bin/python"
COMFY_PY="$CONDA_ROOT/envs/comfy/bin/python"
COMFY_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
ALL=(comfyui api worker-generate worker-cost)
mkdir -p "$DATA/logs" "$DATA/run"

cfg() {  # read a dotted key from the config
  PYTHONPATH="$REPO/src" "$VF_PY" -c "
import sys
from voguefusion import config
v = config.load()
for k in sys.argv[1].split('.'):
    v = v[k]
print(v)" "$1"
}

cmd_for() {
  case "$1" in
    comfyui)
      local url port
      url="$(cfg comfyui.url)"; port="${url##*:}"
      # Bound to loopback only: ComfyUI has no auth and is never exposed to clients.
      echo "cd $COMFY_DIR && exec env -u CUDA_VISIBLE_DEVICES $COMFY_PY main.py --listen 127.0.0.1 --port $port --disable-auto-launch" ;;
    api) echo "cd $REPO && exec env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $VF_PY -m voguefusion.api" ;;
    worker-generate) echo "cd $REPO && exec env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $VF_PY -m voguefusion.worker generate" ;;
    worker-cost) echo "cd $REPO && exec env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $VF_PY -m voguefusion.worker cost" ;;
    *) echo "unknown component: $1" >&2; exit 2 ;;
  esac
}

running() { [ -f "$DATA/run/$1.pid" ] && kill -0 "$(cat "$DATA/run/$1.pid")" 2>/dev/null; }

start() {
  if running "$1"; then echo "$1 already running (pid $(cat "$DATA/run/$1.pid"))"; return; fi
  nohup bash -c "$(cmd_for "$1")" >>"$DATA/logs/$1.log" 2>&1 &
  echo $! >"$DATA/run/$1.pid"
  echo "$1 started (pid $!, log $DATA/logs/$1.log)"
}

stop() {
  if ! running "$1"; then rm -f "$DATA/run/$1.pid"; echo "$1 not running"; return; fi
  local pid; pid="$(cat "$DATA/run/$1.pid")"
  kill "$pid"
  for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  rm -f "$DATA/run/$1.pid"
  echo "$1 stopped"
}

action="${1:-status}"; shift || true
targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=("${ALL[@]}")
case "$action" in
  start) for c in "${targets[@]}"; do start "$c"; done ;;
  stop) for ((i=${#targets[@]}-1; i>=0; i--)); do stop "${targets[$i]}"; done ;;
  status) for c in "${ALL[@]}"; do
            if running "$c"; then echo "$c running (pid $(cat "$DATA/run/$c.pid"))"; else echo "$c stopped"; fi
          done ;;
  *) echo "usage: $0 start|stop|status [component...]" >&2; exit 2 ;;
esac
