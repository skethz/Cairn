set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py gdn_h_recipe_proxy 8 baseline_a --updates 10 --save
/usr/bin/python3 verification/retention_probe.py gdn_h_recipe_proxy 12 retain12 --updates 10 --save
