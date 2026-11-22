set -eu
cd /scratch/hongse/mythos/cairn-rust
/usr/bin/python3 verification/retention_probe.py gdn_gqa 14 candidate_b --updates 10
/usr/bin/python3 verification/retention_probe.py gdn_gqa 8 baseline_b --updates 10
/usr/bin/python3 verification/retention_probe.py full_mla 4 queued_timing --updates 12
