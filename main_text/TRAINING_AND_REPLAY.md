# Training and replay entry points

All paths in these commands are supplied by the reader. None requires the author's
SSH account. Use a new output directory. Training is not required to use the 126
published checkpoints through `predict.py`.

## Reference input layout

Copy the small helper package from `reference_template/fog/` into your reference
directory. Put the processed NPZ records in `data/records/` and copy
`input_manifest.json` to `config/records.json`. Each NPZ has `t`, `x`, `y` at 100 Hz.
The raw input files are not redistributed here.

```text
reference_project/
  fog/__init__.py
  fog/constants.py
  fog/coordinates.py
  config/records.json
  data/records/Pt_33_FD33_4.npz
  ...
```

The batch engines verify the exact input hashes in the manifest. The single-record
training engine requires `--data` for its parser even when the reference loader
reads the same record via `--reference-root` and `--tag`.

## Primary model (single record)

```bash
python main_text/code/primary/generalized_hopf_deeponet.py \
  --data /path/to/reference_project/data/records/Pt_33_FD33_4.npz \
  --reference-root /path/to/reference_project --tag 'Pt 33_FD33_4' \
  --out /path/to/new_primary_fit --architecture parameters \
  --observation-aggregation endpoint --history-cycles 1 --horizon 100 \
  --epochs 80 --stride 5 --draws 128 --train-fraction 0.65 \
  --validation-within-train 0.2 --normal-only --risk-rule cycle_gap \
  --cycle 1.13 --reference-rule paper_2020 --merge-gap-cycles 0 \
  --event-horizon 1 --alarm-threshold 0.8 --confirm 2
```

The engine's auxiliary objective and immediate training-run risk output are for
one second. Use `multihorizon_events.py` for the subsequent 1/2/3/5-second evaluation.
The chosen checkpoint minimises joint validation loss; validation uses eight paths
with fixed noise seed 777. Training uses four paths. This is not selection by
two-second coverage or Brier score.

## Training controls

Use the same command and options with the engine
`main_text/code/controls/generalized_hopf_deeponet.py`, adding:

- No event loss: `--event-loss-weight 0`.
- Constant future coefficients: `--event-loss-weight 0.2 --constant-coefficients`.

The constant-coefficient control removes the branch and trunk layers and predicts
one history-conditioned coefficient triple. It also changes model capacity.
It is different from freezing one coefficient after fitting the full model.

The MLP training entry point is:

```bash
python main_text/code/controls/train_direct_risk_mlp.py \
  --reference-root /path/to/reference_project \
  --base-run /path/to/primary_training_output \
  --out /path/to/new_mlp_fit --event-horizon 2 --seed 2026
```

`--base-run` is the full primary training-output project with `input_manifest.json`
and per-record `model.pt`, `metrics.json`, `peak_calibration.json`,
`reference_events.csv`, and `network_inputs.npz`. The model-only folder published
here is not a substitute for that project. The MLP uses paired-force history,
peak context and peak age; it selects its checkpoint using validation BCE.

## Additional main-text replays

| Experiment | Archived entry point | Required external layout |
|---|---|---|
| Primary learned/zero diffusion | `code/interventions/run_unified.py` | `--source`: full original checkpoint project; `--reference`: reference project; `--out`: new evaluation directory |
| No-event, constant and MLP comparison | `code/controls/evaluate_2s_controls.py` | `--base-run`, `--no-event-run`, `--constant-run`, `--mlp-run`, `--reference-root`, `--out` |
| Freeze beta | `code/interventions/beta_sensitivity_replay.py` | See `--help`; original source and replay inputs |
| Freeze frequency or sigma | `code/interventions/f_sigma_sensitivity_replay.py` | See `--help`; original source and replay inputs |
| Joint observation/reconstruction alignment | `code/observation_alignment/train_pair.py` | `--archive`: original unified project; `--reference`: reference project; `--out`: new paired fits |
| Evaluate paired alignment fits | `code/observation_alignment/evaluate_retrained.py` | `--archive`, `--train-root`, `--out` |
| Continuous vs filtered inference | `code/interventions/run_continuous.py` | See `--help`; original unified archive and raw records |

The original unified archive includes `source_snapshot/code/`,
`source_snapshot/records/<record>/`, its input manifest, and the saved evaluation
inputs expected by each archived script. Full decoded path arrays are not part of
this release. The self-contained `predict.py` interface does not require this
archive layout and supports all six published checkpoint families.

`code/statistical.py` retains the original Kaplan-Meier fitting, renewal recursion,
peak-age logistic fitting and saved-fit prediction formulas. The selected fit
parameters for all 21 records and four event windows are already available in
`models/statistical/baseline_models.json`.

Exact Monte Carlo reproduction requires the original device, PyTorch version,
batch shape, sample order and RNG state. The original primary GPU run used
PyTorch 2.6.0+cu124 and batches of 32. The ablation replay used batches of 16.
The continuous CPU and paired alignment runs are separate experiments. A new
single-time CPU simulation should not be labelled as the saved primary result.
