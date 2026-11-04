#!/bin/sh
# Evidence clips wrapper: forwards to src/clips.py (spec 09).
# Usage: scripts/make_clips.sh [--check] [RUN_DIR [EVENT...]] [--audit PATHS...]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/src/clips.py" "$@"
