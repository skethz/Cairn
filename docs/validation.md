# Validation and source identity

This release preserves the local Rust workspace without changing its implementation. Stored records describe earlier runs; no model training or native build was rerun for this upload.

## Source comparison

`source_identity.json` compares the current files with `rust/evidence/optimization/release_manifest.json`, whose worker hash matches the worker identified by `rust/evidence/optimization_summary.json`.

Of 38 source/configuration/build entries, 37 match exactly. `src/bin/control.rs` differs. No additional files exist under `src/`, `native/`, or `configs/` beyond the historical manifest. This comparison covers source bytes and recorded hashes; it is not a reproducible-build demonstration or independent authentication of the historical records.

The later `rust/evidence/queue_consolidation_20261104T020744Z/controller_verification.json` records three passing controller tests and checks for waiting without occupying a GPU, rejecting duplicate waiters, and blocking successors when the predecessor is paused. It records a binary hash but no corresponding updated source hash. Therefore the current controller's identity with that tested binary remains unproven.

## Numerical validation

`rust/evidence/optimization_summary.json` records exact tensor agreement with the team's Python rewrite after two or three continuation updates for four approximately 1.3B models. Compared tensors include weights, optimizer/RNG state, and final gradients. This is a finite comparison, not proof of equivalence for every input, precision, configuration, or training duration.

The full-scale oracle uses an external frozen Python training reference and saved fixtures. `rust/verification/owned_local/` is a separate retained reference snapshot and does not by itself satisfy all oracle dependencies. `run_oracle.sh` contains historical machine paths. It is retained for provenance, not presented as a portable reproduction command.

## Training progress

The 4 November 2026 receipt records Cairn paused at step 27,814 with 882,864,406 valid tokens; weights and optimizer file hashes were recorded along with CPU/CUDA RNG and data cursor. These counts include restored Python progress. They do not establish completed pretraining, held-out benchmark quality, or checkpoint availability today. At the current inspection, the former `/scratch/hongse/mythos` server directory was absent.

## Pending release material

Portable reproduction environment, checkpoint fixtures, dataset preparation package, final quality benchmarks, and paper: **Coming soon.**
