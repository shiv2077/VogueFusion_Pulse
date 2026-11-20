#!/bin/bash
# Kept for old instructions that point here. The stack is managed by scripts/vf.sh;
# see the README for setup (conda environments, ComfyUI, models).
exec "$(dirname "$0")/scripts/vf.sh" start "$@"
