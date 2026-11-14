#!/usr/bin/env bash
set -euo pipefail
R=/data/scene-rep/u/aquan/speculative-tonight-20261109
ROOT=/tmp/aquan-forespark-v63
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export WANDB_BASE_URL=https://api.wandb.ai
export SERVING_SCRATCH=/tmp/aquan-serving-v14-3043052.VFLAcm
python3 - <<'PY'
import json,xml.etree.ElementTree as ET
from pathlib import Path
r=Path('/data/scene-rep/u/aquan/speculative-tonight-20261109')
assert json.loads((r/'forespark_v63_setup.json').read_text())['status']=='completed'
suites=ET.parse(r/'forespark_v63_scheduler_tests.xml').getroot().findall('testsuite')
assert sum(int(x.get('tests','0'))-int(x.get('skipped','0')) for x in suites)>0
assert all(int(x.get('failures','0'))+int(x.get('errors','0'))==0 for x in suites)
PY
set +e
"$ROOT/ForeSpark/.venv/bin/python" "$R/check_labels_v67.py"
check_status=$?
set -e
"$SERVING_SCRATCH/env/bin/python" "$R/publish_labels_v67.py"
exit "$check_status"
