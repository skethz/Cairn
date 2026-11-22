set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py cairn 14 candidate_b --updates 10
/usr/bin/python3 verification/retention_probe.py cairn 8 baseline_b --updates 10
