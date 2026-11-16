#!/usr/bin/env bash
# Run all 26 proposed L-moment-only configurations, seed 0, batch 512.
# Execute from any directory inside the cloned repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python -m lmoment_lifelong.hypothesis26 --seed 0 "$@"
