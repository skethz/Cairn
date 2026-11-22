set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py cairn 8 baseline_a --updates 10 --save
/usr/bin/python3 verification/retention_probe.py cairn 12 retain12 --updates 10 --save
