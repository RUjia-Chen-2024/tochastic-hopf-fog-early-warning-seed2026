# Code entry points and required inputs

Run `python validate_package.py` and `python verify_checksums.py` from the repository root to check the saved protocol, headline tables and release files. These two checks need only Python's standard library.

Install the simulation/reporting dependencies with `python -m pip install -r requirements.txt`. The requirements file lists dependencies and the recorded PyTorch version; it is not a full environment lock. Exact saved-stream replay additionally needs the recorded CUDA/device/numerical settings described in `protocol/canonical_protocol.json`.

## Main evaluation and noise intervention

`primary_seed2026/run_unified.py` evaluates the frozen five-second checkpoint project. Its `--source` argument must point to the matching project containing `code/` and `records/<record>/model.pt`, `network_inputs.npz`, calibration and reference files. `--reference` supplies the matching reference project with `data/records/<record>.npz`. These external input directories are not bundled here. A typical command from the repository root is:

```bash
python scripts/primary_seed2026/run_unified.py \
  --source /path/to/checkpoint_project \
  --reference /path/to/reference_project \
  --out /path/to/new_evaluation_output
```

See `results/primary_seed2026/README.md` for saved-stream provenance and replay details. References there to an enclosing server run directory describe the archived execution layout. This compact repository stores its result files directly under `results/primary_seed2026/`.

## Other experiments

- `component_replay/`: model/training code, no-event-loss and constant-coefficient controls, direct-risk MLP, and their evaluation scripts.
- `parameter_sensitivity/`: paired beta, frequency and diffusion-amplitude interventions using frozen checkpoints and archived inputs.
- `baselines/`: peak-clock, inter-peak survival and peak-age probability baselines. The archived reporting scripts expect the original analysis tree or the explicit input directories accepted by each script; that tree is not reconstructed by the compact saved-result bundle.
- `decoder_limit_cycle_audit/`: affine-decoder invertibility and latent/force-space limit-cycle analysis. Use `--checkpoint-root` with the matching checkpoints and `predictions.npz` files.

`primary_seed2026/execution_source/` and `primary_seed2026/analysis_body.py` retain execution/provenance code. The latter is a reporting-code fragment used with the shared analysis globals, rather than a standalone command.

## Optional collection of archived force metrics over SSH

`baselines/collect_saved_force_metrics.py` sends the adjacent `summarize_saved_forces_remote.py` to an already configured SSH server. Set `FOG_SSH_HOST` and `FOG_SSH_USER`, plus `FOG_REMOTE_RUN_ROOT` (the server directory containing `results/records/`) and `FOG_REMOTE_SOURCE_ROOT` (the matching checkpoint project containing `records/`). `FOG_REMOTE_REFERENCE_ROOT` can override the default reference-record directory. The script asks for the SSH password interactively and uses the system's known-host keys. Its output is `results/baselines_seed2026/force_metrics_seed2026.json`.
