#!/usr/bin/env bash
set -euo pipefail
bash run_two_stage_n.sh --n-sim 20 --out-dir outputs_smoke_test --no-png
echo "Smoke test done. Open outputs_smoke_test/visuals_compact/index.html"
