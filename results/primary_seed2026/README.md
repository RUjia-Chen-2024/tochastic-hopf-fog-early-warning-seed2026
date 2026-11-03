# Current evaluation: seed 2026, archived random streams

Open `index.html` or read `REPORT_中文.md`. This report supersedes the earlier
mixed-stream reports for the primary evaluation, diffusion intervention and
1/2/3/5-s analysis. No model retraining or test-threshold tuning was performed.

## Single source of reported numbers

- `results/primary_evaluation.csv`: the learned-noise 1-s row.
- `results/multihorizon_evaluation.csv`: the same 1-s row plus 2/3/5-s rows.
- `results/primary_risk.csv` and `results/multihorizon_risks.csv`: their 1-s
  probabilities, labels and alarm flags have been read back and checked exactly.
- `results/noise_by_horizon.csv`: zero and learned diffusion, same new run.
- `results/consistency_checks.json`: audit, including exact path replay.

Every evaluated record and diffusion condition resets Python, NumPy and
PyTorch inference to seed **2026**, without a record-specific seed offset.
Input order, manifest order, batch size 32 and 128 learned-noise paths are fixed.
The zero-diffusion condition has one unique path, equivalent to 128 duplicates.

## What was actually archived

1. CPU/CUDA RNG state immediately before every simulation batch, including
   batch boundaries, seed, draw count, and device, in `*_rng_stream.pt`.
2. Full decoded float32 paths for every origin, `*_force_paths.npy`, retained
   on the SSH server. Approximate total size: 4 GB.
3. Per-path first-new-onset offsets and four risks in `*_first_onsets.npz`.
4. All probabilities, labels, alarms and event matches as CSV files.
5. Source/checkpoint/data snapshots with SHA256 checksums.

The local result package includes states, event paths, CSVs, source snapshots
and full-array checksums. It intentionally does not duplicate the large full
force arrays. See `results/saved_stream_manifest.csv` for their exact server
locations and hashes. They are not discarded or replaced by seeds alone.

Server directory:

`<SERVER_RUN_DIR>`

## Recreate the evaluation from the archived sources

```bash
cd <SERVER_RUN_DIR>
python -u run_unified.py \
  --source source_snapshot --reference reference_snapshot --out results_reproduced
```

The original execution used byte-identical source and raw-data files from the
existing latest 5-s project. The archived snapshot is hash-verified against
those files. A new output directory is necessary for a fresh full simulation;
COMPLETE markers otherwise resume completed results.

```bash
python replay_saved_stream.py \
  --source source_snapshot --results results --record Pt_10_FD10_1
```

The independent replay check already reproduced all 17 batches and 521 origins
of that record, including complete 128-path learned-noise trajectories, exactly.
All 42 record/condition runs additionally reproduced their saved final batch.
An exact full-path replay requires access to the server's large NPY arrays or
their downloaded copies.

Environment: PyTorch 2.6.0+cu124, RTX A6000 GPUs, deterministic algorithms enabled,
TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Bitwise replay is asserted in
the recorded environment; software and numerical settings are not left implicit.

`python analyze_unified.py` regenerates tables, figures, HTML and manuscript
replacement text from saved outputs. It needs NumPy, SciPy, pandas, Matplotlib;
simulation/replay additionally requires PyTorch. No new random simulation is
performed by the reporting script.

`execution_source/` preserves the exact scripts and dependency module executed
on the server. The portable entry points use `frozen_solver.py`, containing the
exact extracted solver and IO functions, to avoid requiring a sibling experiment
directory. Only imports were simplified; no simulation arithmetic changed.
Function and source hashes are recorded in `results/code_provenance.json`.

## Scope

The standardized stream is the frozen-checkpoint EVALUATION stream. Existing
model training used main seed 2026 and historical fixed validation noise 777;
neither weights nor that historical training operation was changed. This package
does not mislabel prior training as newly rerun under a different validation seed.

The same retrospective cycle-derived references and normal-origin screening
remain in use. Longer event windows have not been separately retrained or
calibrated. Primary and multihorizon tables must be taken together from this
version rather than spliced with prior seed-dependent results.

## Independent Brier audit

`python audit_brier.py` independently recomputes path probabilities, targets from
the archived reference onsets, tail masks from raw record ends, and scalar
pooled Brier using `math.fsum`. No reporting/simulation function is imported.
`brier_audit/index.html` and its CSVs contain aligned 1-s baseline scores and
positive/negative-target decompositions. Baseline origin sets, labels and scoring
masks are checked exactly. The probability threshold and alarm-confirmation rule
are not used in Brier computation.

Baseline risks are the existing frozen 1-s predictors, independently rescored;
they do not include Monte Carlo path simulation and were not refitted here.
Do not label those scores as 2/3/5-s baselines. The fit-only constant reference
is separately available for each longer target. A pure clock's binary 0/1 score
is labelled explicitly rather than described as a calibrated probability.

The primary report includes this audit via `attach_brier_audit.py`; rerunning
`analyze_unified.py` preserves that section when the audit CSV exists.

