# Models evaluated in the main article

This directory corresponds to the author's `Transition_DeepOnet_all_editors (39).pdf`,
**Inverse modelling of stochastic Hopf dynamics for early warning of stepping-cycle anomalies**.
It contains the implemented models from main-text Sections 2 and 4. Models cited only
as prior work in the Introduction (for example DMD and reservoir computing) are not
presented as implementations of this study.

**126 original trained checkpoints are now included: six families, 21 recordings each.**
The files are byte-for-byte copies of the saved checkpoints; no model was retrained or
modified for this release. `MODEL_INDEX.json` lists every path, size and SHA256 hash.
The source PDF, raw force records, full Monte Carlo arrays and appendix-only result
tables/figures are not added here. The repository's older published directories are
retained as historical material and are not newly generated results.

## Model-to-manuscript map

| Model or control | Main-text location | Implementation | Saved parameters |
|---|---|---|---|
| DeepONet-Hopf, 5-s endpoint model | Sections 2.1-2.2; Table 1 | `code/primary/generalized_hopf_deeponet.py`, `ParameterHopfDeepONet` | `models/full/<record>/model.pt`, 21 fits |
| Inter-peak survival with future renewals | Section 2.2; Table 1; Section 4.5 | `code/statistical.py`, `fit_survival`, `survival_risks` | `models/statistical/baseline_models.json` |
| Pure peak clock | Section 2.2; Table 1 | `code/statistical.py`, `predict_baseline(..., method='peak_clock')` | Fixed rule; no trained checkpoint |
| Peak-age logistic | Section 2.2; Table 1 | `code/statistical.py`, `fit_logistic`, `predict_baseline` | Same JSON; separate fits for 1/2/3/5 s |
| Fit-prefix constant | Section 2.2; Table 1 | `code/statistical.py`, `predict_baseline(..., method='fit_constant')` | Same JSON; fitting-prevalence estimates |
| Constant probability 0.5 | Section 2.2; Table 1 | `code/statistical.py`, `predict_baseline(..., method='constant_half')` | Fixed reference; no trained checkpoint |
| Force persistence | Section 2.1; Figure 3 | `code/statistical.py`, `persistence` | Last observed five-sample block mean; no trained checkpoint |
| No auxiliary event loss | Section 2.3.1; Table 3 | `code/controls/generalized_hopf_deeponet.py`, `--event-loss-weight 0` | `models/no_event_loss/<record>/model.pt`, 21 fits |
| Constant future coefficients | Section 2.3.1; Table 3 | Same engine, `--constant-coefficients` | `models/constant_coefficients/<record>/model.pt`, 21 fits |
| Direct risk MLP | Section 2.3.1; Table 3 | `code/controls/train_direct_risk_mlp.py`, `DirectRiskMLP` | `models/direct_risk_mlp/<record>/model.pt`, 21 fits |
| Retrained endpoint/block-mean reconstruction reference | Section 2.3.1, page 10 | `code/observation_alignment/training/` | `models/alignment_endpoint/<record>/model.pt`, 21 fits |
| Retrained block-mean/instantaneous reconstruction model | Section 2.3.1, page 10 | Same alignment engine | `models/alignment_blockmean/<record>/model.pt`, 21 fits |
| Freeze future beta, frequency or sigma | Section 2.3.2 | `code/interventions/*sensitivity_replay.py`; single-time interface in `predict.py` | Reuses `models/full`; no additional fits |
| Zero diffusion | Section 2.3.3; Figure 5 | `code/interventions/frozen_solver.py`; `predict.py --noise-multiplier 0` | Reuses `models/full`; no additional fits |
| Continuous vs reference-normal replay | Section 2.2, pages 8-9 | `code/interventions/run_continuous.py` | Reuses `models/full`; changes evaluation/inference policy |

The full-model ablation row reuses the primary 21 checkpoints but a different Monte
Carlo batch stream. It is **83/91, 166 alarms**, whereas the archived primary stream is
**84/91, 160 alarms**. The alignment endpoint reference is a *newly retrained* model;
it must not be substituted for the original primary checkpoint.

## Verify and run

From the repository root:

```bash
python -m pip install -r requirements.txt
python main_text/verify_models.py
python main_text/verify_models.py --smoke
```

The first check needs only the standard library. The smoke check strictly loads all
126 checkpoints, executes one forward pass per family, checks statistical outputs,
and exercises the shared peak-deadline detector. Its synthetic inputs are software
checks, not additional research results.

A statistical model can be evaluated without raw force data:

```bash
python main_text/predict.py --model survival --record Pt_33_FD33_4 --age 0.59
python main_text/predict.py --model peak_age_logistic --record Pt_33_FD33_4 --age 0.59
```

For a neural model, supply the matching 100-Hz record as an NPZ containing arrays
`t`, `x`, `y`, or a CSV with those headers. Times are in seconds; force units must
match the training data. Do not normalise the file in advance.

```bash
python main_text/predict.py --model full --record Pt_33_FD33_4 \
  --data /path/to/Pt_33_FD33_4.npz --time 89.94 --out /path/to/prediction.json
```

Use `--model no_event_loss`, `constant_coefficients`, `direct_risk_mlp`,
`alignment_endpoint` or `alignment_blockmean` for another trained family.
`--model persistence` returns a force forecast, not a warning probability.
Optional `--out` also saves generated arrays as NPZ; these are local outputs and are
not included in this repository addition.

For fixed-weight interventions, use `--model full --noise-multiplier 0`, or
`--model full --freeze beta`, `--freeze frequency`, or `--freeze sigma`.
Freezing sigma preserves stochastic forcing. Zero diffusion removes it. Running the
single-time interface with identical seed/device/draw count pairs the random draws,
but this fresh single-time stream is **not** the archived batch-32 manuscript stream.
It must not be expected to reproduce the illustrated 114/128 count exactly.

An individual probability is not an issued alarm. Apply the archived
`threshold_warnings(times, risks, threshold=0.8, confirm=2, dt=0.05)` state machine
from `code/primary/generalized_hopf_deeponet.py` to chronological predictions.
Risk must be strictly above 0.80 twice; the second time is the alarm time.

## Training targets and evaluation scope

- Each recording has its own model: approximately 52% fitting, 13% validation,
  35% testing in chronological order. These are not held-out-participant models.
- Neural input is one cycle (1.13 s, 23 paired averages). The event calculation
  additionally uses past peak context and peak age.
- The stochastic models simulate 5 s. Their auxiliary event loss uses **1 s**;
  the exploratory primary evaluation uses **2 s**. The MLP is trained directly on
  two-second labels. No dynamical-parameter labels are used.
- Main labels are algorithmic prolonged-cycle references: accepted peak gap
  greater than 1.5T, T=1.13 s, onset at previous peak +1.695 s, no extra event merging.
- Primary simulated force observations are endpoints; observed loss targets are
  five-sample means. The alignment models change both this readout and the initial
  reconstruction target. They are reported as a separate retraining comparison.
- Primary headline metrics use offline reference-normal screening. The single-time
  interface uses past signal only and does not perform that offline selection.

## Training and archived batch replay

The exact archived engines are retained under `code/`. Their historical defaults
include settings not used by this article, so use the explicit commands in
[`TRAINING_AND_REPLAY.md`](TRAINING_AND_REPLAY.md). `SOURCE_PROVENANCE.json` records
the copied source hashes and the source-PDF hash. The checkpoint files retain
their original training arguments for provenance; archived server paths in those
arguments do not need to exist to use `predict.py`.

The original raw data are identified in the article at
<https://figshare.com/s/a14be7360925639736ba>. `input_manifest.json` records the 21
processed record identifiers and hashes. A reader needs the matching processed
inputs to rerun training or the exact archived batch evaluations. Saved headline
results elsewhere in the repository remain unchanged.

The article's current Code availability sentence says checkpoints are omitted.
That sentence describes the earlier repository and should be updated to note that
the main-text checkpoints are now in `main_text/models/`. The PDF itself is not
changed by this release.
