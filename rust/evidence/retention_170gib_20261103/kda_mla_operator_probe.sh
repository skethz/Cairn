set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py kda_mla_operator 2 baseline_a --updates 10 --save
/usr/bin/python3 verification/retention_probe.py kda_mla_operator 4 retain4 --updates 10 --save
