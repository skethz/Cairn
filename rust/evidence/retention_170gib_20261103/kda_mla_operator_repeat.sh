set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py kda_mla_operator 5 candidate_b --updates 10
/usr/bin/python3 verification/retention_probe.py kda_mla_operator 2 baseline_b --updates 10
