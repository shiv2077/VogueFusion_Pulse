#!/usr/bin/env bash
# Download and hash-verify every model in comfyui/models.json into $COMFYUI_DIR.
# Idempotent: a file that already matches its sha256 is skipped. Partial
# downloads resume. A file whose hash does not match is moved aside, never used.
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
MANIFEST="$(cd "$(dirname "$0")" && pwd)/models.json"
status=0

while IFS=$'\t' read -r dest url sha token_var; do
  out="$COMFYUI_DIR/$dest"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok       $dest"
    continue
  fi
  auth=()
  if [ -n "$token_var" ] && [ -n "${!token_var:-}" ]; then
    auth=(-H "Authorization: Bearer ${!token_var}")
  fi
  echo "download $dest"
  if ! curl -fL --retry 5 --retry-delay 10 -C - "${auth[@]}" -o "$out.part" "$url"; then
    echo "FAILED   $dest (curl error; set ${token_var:-a token} if the host requires login)" >&2
    status=1
    continue
  fi
  got="$(sha256sum "$out.part" | cut -d' ' -f1)"
  if [ "$got" != "$sha" ]; then
    mv "$out.part" "$out.badhash"
    echo "BADHASH  $dest expected $sha got $got (kept as .badhash)" >&2
    status=1
    continue
  fi
  mv "$out.part" "$out"
  echo "verified $dest"
done < <(python3 -c '
import json, sys
for m in json.load(open(sys.argv[1]))["models"]:
    print(m["dest"], m["url"], m["sha256"], m.get("needs_token", ""), sep="\t")
' "$MANIFEST")

exit $status
