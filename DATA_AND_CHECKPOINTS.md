# Data and model inputs

The manuscript identifies the source force data at <https://figshare.com/s/a14be7360925639736ba>. Raw force files are not redistributed. The 2026-11-16 addition includes 126 original checkpoints for the six trained model families discussed in the main text, with an explicit index and hashes in `main_text/MODEL_INDEX.json`. Statistical baseline fits are in `main_text/models/statistical/baseline_models.json`.

See `main_text/README.md` for the model-to-section map and single-time inference examples. To rerun complete archived experiments, supply the matching 21-record reference data and the additional input artifacts expected by each archived script. The model-only folder does not include cached force histories or full decoded trajectories.

The reported result CSVs and their audits are included; the full 128-path decoded force arrays (approximately 4 GB) remain outside this compact GitHub package. The RNG state files, per-path first-onset arrays, run protocol, and file hashes are retained for audit.

The existing decoder/limit-cycle audit includes the fitted 2×2 observation matrices and inverse-mapped 128-path ensemble-mean trajectories. Reproducing that historical audit still requires per-record `predictions.npz` files in its expected layout. Individual Monte Carlo paths are not included. No appendix-only results are added by the main-text model release.
