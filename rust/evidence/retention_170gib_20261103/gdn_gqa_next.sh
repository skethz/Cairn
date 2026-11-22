set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 - <<'WAIT'
import json,time
from pathlib import Path
p=Path("evidence/retention_170gib_20261103/gdn_gqa/retain12/probe.json")
for _ in range(300):
 v=json.loads(p.read_text())["status"]
 if v=="passed":break
 assert v=="running",v
 time.sleep(2)
else:raise RuntimeError("Probe did not finish")
WAIT
/usr/bin/python3 verification/retention_probe.py gdn_gqa 14 retain14 --updates 10 --save
/usr/bin/python3 verification/compare_retention.py evidence/retention_170gib_20261103/gdn_gqa/baseline_a evidence/retention_170gib_20261103/gdn_gqa/retain14 evidence/retention_170gib_20261103/gdn_gqa/retain14_comparison.json
