#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
ROOT=/tmp/aquan-forespark-v63
mkdir -p "$ROOT"
export TMPDIR="$ROOT/tmp"
export UV_CACHE_DIR="$ROOT/uv-cache"
export XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
python3 - <<'PY'
import shutil
assert shutil.disk_usage('/tmp/aquan-forespark-v63').free > 40*1024**3, 'Need40GiB free for isolated dependencies'
PY
if [ ! -d "$ROOT/ForeSpark/.git" ]; then
 git clone https://github.com/facebookresearch/ForeSpark.git "$ROOT/ForeSpark"
fi
cd "$ROOT/ForeSpark"
git checkout --detach 5d457ddab73a76f28db34acc577abfbd9e7c6eee
test "$(git rev-parse HEAD)" = 5d457ddab73a76f28db34acc577abfbd9e7c6eee
bash setup.sh
.venv/bin/python -m pytest tests/test_scheduler.py -q --junitxml="$R/forespark_v63_scheduler_tests.xml"
.venv/bin/python - <<'PY'
import json,importlib.metadata as m,subprocess
from pathlib import Path
result={'status':'completed','scope':'Isolated official dependency setup and CPU scheduler tests only; no model or speed result','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'versions':{k:m.version(k) for k in ['torch','vllm','speculators','transformers','numpy','forespark']}}
Path('/data/scene-rep/u/aquan/speculative-tonight-20261109/forespark_v63_setup.json').write_text(json.dumps(result,indent=2)+'\n')
PY
