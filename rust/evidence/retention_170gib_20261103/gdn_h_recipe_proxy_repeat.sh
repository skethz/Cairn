set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py gdn_h_recipe_proxy 14 candidate_b --updates 10
/usr/bin/python3 verification/retention_probe.py gdn_h_recipe_proxy 8 baseline_b --updates 10
/usr/bin/python3 verification/retention_probe.py dense_gqa 8 queued_timing --updates 12
