#!/usr/bin/env bash
# One command that proves the repo is in a good state. Exit 0 only if every
# check that can run passes. GPU checks are skipped, with a notice, when
# ComfyUI or the API is not running (start them with scripts/vf.sh start).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
BIN="$CONDA_ROOT/envs/voguefusion/bin"
export PYTHONPATH="$REPO/src"
status=0

step() { echo; echo "== $1"; }
run() { "$@" || { echo "FAILED: $*"; status=1; }; }

step "lint"
run "$BIN/ruff" check src tests scripts

step "tests (no GPU)"
run env CUDA_VISIBLE_DEVICES= "$BIN/python" -m pytest -q -p no:warnings

step "docs: no TBD, every ASSUMED tracked"
run "$BIN/python" scripts/check_docs.py

step "workflow files match the recovered graphs"
before="$(sha256sum comfyui/workflows/*.json)"
run "$BIN/python" scripts/recover_workflows.py >/dev/null
after="$(sha256sum comfyui/workflows/*.json)"
[ "$before" = "$after" ] && echo "ok: regenerating workflows is a no-op" || { echo "FAILED: workflows differ from recovery"; status=1; }

comfy_url="$("$BIN/python" -c 'from voguefusion import config; print(config.load()["comfyui"]["url"])')" \
  || { echo "FAILED: cannot read comfyui.url from config"; exit 1; }
api_url="$("$BIN/python" -c 'from voguefusion import config; c = config.load()["api"]; print("http://%s:%s" % (c["host"], c["port"]))')" \
  || { echo "FAILED: cannot read api host/port from config"; exit 1; }

step "replay a committed design through ComfyUI"
if curl -sf "$comfy_url/system_stats" >/dev/null; then
  run env CUDA_VISIBLE_DEVICES= "$BIN/python" scripts/replay.py image_9.png
else
  echo "SKIPPED: ComfyUI not reachable at $comfy_url"
fi

step "end to end through the API (v1 and v2, then cost)"
if curl -sf "$api_url/healthz" >/dev/null && curl -sf "$comfy_url/system_stats" >/dev/null; then
  run env CUDA_VISIBLE_DEVICES= "$BIN/python" scripts/e2e.py v1 v2
else
  echo "SKIPPED: API ($api_url) or ComfyUI ($comfy_url) not running"
fi

step "web app in headless Chrome (generate, colors, dye recipes, cost, status)"
if curl -sf "$api_url/healthz" >/dev/null && curl -sf "$comfy_url/system_stats" >/dev/null; then
  env CUDA_VISIBLE_DEVICES= "$BIN/python" scripts/ui_check.py
  rc=$?
  if [ $rc -eq 2 ]; then echo "SKIPPED: Playwright or Chrome not available"; elif [ $rc -ne 0 ]; then echo "FAILED: ui_check"; status=1; fi
else
  echo "SKIPPED: API or ComfyUI not running"
fi

echo
[ $status -eq 0 ] && echo "VERIFY: PASS" || echo "VERIFY: FAIL"
exit $status
