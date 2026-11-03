# Data and model inputs

The manuscript identifies the source force data at <https://figshare.com/s/a14be7360925639736ba>. This package omits raw force files and trained checkpoint weights. To rerun simulations, obtain the matching 21-record reference data and the exact 5-s endpoint checkpoints whose SHA256 values are recorded in the saved provenance outputs, then pass their directories to the relevant script.

The reported result CSVs and their audits are included; the full 128-path decoded force arrays (approximately 4 GB) remain outside this compact GitHub package. The RNG state files, per-path first-onset arrays, run protocol, and file hashes are retained for audit.

The decoder/limit-cycle audit includes the fitted 2×2 observation matrices and inverse-mapped 128-path ensemble-mean trajectories. Individual latent/force Monte Carlo paths and trained checkpoints are not included; reproducing that audit requires the matching per-record `model.pt` checkpoints and `predictions.npz` files.
