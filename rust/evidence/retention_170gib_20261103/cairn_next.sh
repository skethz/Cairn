set -eu
cd /scratch/hongse/mythos/cairn-rust
while ! /usr/bin/python3 -c 'import json,sys;sys.exit(0 if json.load(open("evidence/retention_170gib_20261103/cairn/retain12/probe.json"))["status"]=="passed" else 1)'; do sleep 2; done
/usr/bin/python3 verification/retention_probe.py cairn 14 retain14 --updates 10 --save
/usr/bin/python3 verification/compare_retention.py evidence/retention_170gib_20261103/cairn/baseline_a evidence/retention_170gib_20261103/cairn/retain14 evidence/retention_170gib_20261103/cairn/retain14_comparison.json
