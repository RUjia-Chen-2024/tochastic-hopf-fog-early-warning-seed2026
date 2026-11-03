# Generalized Hopf + DeepONet experimental model

## Current label protocol: Parakkal Unni et al. (2020)

The default reference rule is `--reference-rule paper_2020 --merge-gap-cycles 0`.
It follows Supplementary section 2 of doi:10.3389/fmedt.2020.581264: use mean-peak
scaling and a 1.5T abnormally long inter-peak gap, with fixed global T=1.13 s.
There is no additional event-gap merging. Paper mode rejects nonzero merging.
The article does not fully specify exact onset/offset boundaries or peak detector
settings. Our explicit boundary convention is onset at the preceding peak +1.5T,
offset at the next detected peak, and a terminal event ending at the recording
boundary without asserting recovery. This replaces the legacy +1.2T onset.
The retained 0.45T peak distance, 0.4 height and 0.08 prominence are implementation
choices, not reported paper requirements. Leading gaps before the first peak are
not labelled, and the extractor requires at least two detected peaks; these are
also retained implementation choices. Each result records these limitations in
`label_policy.json` and `metrics.json`, rather than claiming an exact replication.

Offline labels and decoded-force hard/soft risk use the same onset delay. Paper
cycle mode defaults to one cycle of network input when no history is specified.
The 52/13/35 split, 128 paths, 1.5 s simulation, 1 s event target, normal-origin
screening and immediate >0.7 alarm rule remain the research protocol requested
by the user; the article's RF/NN/NB classifier is not being reproduced.

```bash
python <SERVER_RUN_DIR> --reference-root ${FOG_REFERENCE_ROOT} --out <SERVER_RUN_DIR> --epochs 80 --gpus 0,1,2,3 --cycle-mode --history-cycles 1 --reference-rule paper_2020 --merge-gap-cycles 0
```

Use a fresh output directory. Old weights, references, 90/62-event denominators,
and warning scores do not describe this changed label protocol. All-record labels
can be regenerated without training using `audit_paper_cycle_labels.py`.
Legacy boundaries require `--reference-rule legacy`; merging is then an explicit
optional parameter. Historical results retain their original immutable code.

## One-cycle network history experiment

Add `--history-cycles 1` to the cycle-mode batch or single-record command. This
changes only the observed history supplied to the network. The reference cycle,
label rule, selected merging setting, causal peak calibration, physical risk calculation,
forecast horizon and alarm threshold remain the same. Models are retrained.
At 100 Hz, the input support is exactly [origin-1.13 s, origin], including 114
raw samples. They form 23 chronological input blocks: four samples in the oldest
block and five in each remaining block. No older sample enters the network;
output origins and forecast steps remain at 20 Hz. Paper cycle mode defaults to
this one-cycle history; legacy mode retains 40 blocks when omitted. The causal context used by
the external cycle detector is unchanged and is not fed to the network.
Each result exports network_history.json, network_history_context.csv and the
actual normalized test inputs in network_inputs.npz for verification.

## Historical cycle-abnormality experiment with extra event merging

This historical experiment requires `--reference-rule legacy --merge-gap-cycles 1.5`.
Single-record mode additionally uses `--risk-rule cycle_gap --horizon 30
--event-horizon 1 --normal-only`. The fixed
reference cycle remains T=1.13 s. Unmerged cycle-defined intervals are created
with the original source detector, then intervals with end-to-next-onset gaps
STRICTLY less than 1.5T=1.695 s are merged transitively. An exact 1.5T gap remains
separate. Filled gaps belong to the combined event for training and normal-origin
selection. Both unmerged and merged intervals are exported.

All 21 canonical records are processed, with new weights fitted against the
changed labels. A record with no normal test origin after merging is explicitly
reported as not evaluable and is not assigned fabricated predictions or weights.
The first 65% is still split into 52% fit and 13% validation. The last 35% remains
test data. Models predict 1.5 seconds of force, while the event-probability target
is an onset within the next 1.0 second. The extra time covers the 0.3T difference
between onset assignment (1.2T) and confirmation of an abnormal gap (1.5T).

The Hopf solver additionally returns its 100-Hz internal output. Decoded force
paths are concatenated to observed histories and checked for qualifying steps.
Peak scale is calibrated ONLY from the fit prefix; height=0.4, prominence=0.08
and peak distance=0.45T are frozen. Observed peak contexts use prefix data only.
If the last observed peak occurred at p, a candidate onset is p+1.2T and a missing
step is confirmed only if no valid future peak appears by p+1.5T. Since the 1-s
event horizon is shorter than 1.2T, only that last observed peak can generate an
onset in the target horizon. A predicted no-peak tail shorter than the confirmation
deadline is unknown, not a positive event. No observed past peak is also unknown.

Training uses a piecewise differentiable peak-presence proxy in decoded force:
local maxima have detached locations, while peak height and local prominence
remain differentiable. Hard SciPy peak rules are used for reported risk. The
proxy and hard rule are not identical and risk calibration is not established.
The old latent radius=0.3 and three-future-point occupancy rule are NOT used in
cycle mode; their CLI arguments remain for historical latent-mode experiments.
The immediate alarm threshold remains strictly >0.7, without multi-roll confirmation.
Offline reference peak normalization still follows the original full-record label
algorithm; causal prediction normalization is frozen from the fit prefix, so their
agreement is a validation question rather than an assumed identity.

```bash
python <SERVER_RUN_DIR> --reference-root ${FOG_REFERENCE_ROOT} --out <SERVER_RUN_DIR> --epochs 80 --gpus 0,1,2,3 --cycle-mode --reference-rule legacy --merge-gap-cycles 1.5
```

The following sections describe previous latent-risk experiments where indicated.

## Current version: DeepONet predicts parameters

The default architecture is now `--architecture parameters`. A branch/trunk
DeepONet predicts beta(tau), frequency(tau), sigma(tau), with the history encoder
estimating the initial latent state. Future latent states are generated entirely
by numerical integration of the generalized Hopf SDE. There is no additive neural
drift correction and no direct future-state output head. The earlier implementation
is retained under `--architecture residual`; its historical result files remain
unchanged. Old checkpoints without an architecture field load as residual models.

Parameter curves are constant within each 0.05-s output interval. The same
tamed-Euler integrator with internal steps no larger than 0.01 s is used.
Parameters are transformed into beta=(-0.98,-0.02), frequency=(0.1,3) Hz and
sigma=(0.01,0.4). Keeping beta inside the imposed bistable range permits testing
stochastic escape; it does not test deterministic loss of an attractor or identify
a parameter crossing a bifurcation point. Parameter identifiability is unproven.

Training uses one forecast mixture likelihood + 0.5 * current reconstruction
+ 0.01 * parameter-logit smoothness + 0.2 * auxiliary event loss (when labels
are supplied). The obsolete second-branch and path-consistency columns are zero
for this architecture; they are not independent model comparisons.

Run on one record with the requested normal-origin protocol:

```bash
python <SERVER_RUN_DIR> --data '<SERVER_RUN_DIR> 31/original_state_Pt 31_FD31_1.txt' --reference-root ${FOG_REFERENCE_ROOT} --tag 'Pt 31_FD31_1' --out <SERVER_RUN_DIR> --architecture parameters --normal-only --train-fraction 0.65 --epochs 80
```

Outputs additionally contain parameter_forecasts.csv and parameter_forecasts.png/pdf.
The CSV has one row per origin and forecast horizon, with beta, frequency, sigma,
and sqrt(-beta), the unstable radius of the frozen-parameter deterministic system.
That instantaneous radius is descriptive for time-varying parameters, not an exact
time-dependent basin boundary. Risk retains the three-point radius-0.3 occupancy
criterion inside each forecast. The alarm now issues immediately on risk > 0.7
(--confirm 1, the default), as requested after inspecting the first experiment.
An above-threshold run produces one issuance; a later drop re-arms it. The risk
threshold itself remains 0.7. This is a post-hoc alarm-rule change, not a new
independent validation. risk.csv's parameter columns describe
tau=0; all future parameter values are saved in parameter_forecasts.csv/NPZ.

The model is saved with architecture and normalization metadata. Replot without
retraining using `python normal_warning_protocol.py --results OUTPUT_DIRECTORY`.
To replay existing predictions with immediate alarms, use
`python normal_warning_protocol.py --results OUTPUT_DIRECTORY --confirm 1`.
The replay preserves a before_immediate_alarm snapshot and verifies hashes of the
model, forecasts, parameter CSV, training log and cached training fit. It changes
only alarm decisions, evaluation and plot markers. Subsequent replots use the
effective configuration in metrics.json, retaining original training arguments
inside the unchanged checkpoint.

The sections below describe the earlier residual pilot where indicated.

Input: a whitespace-separated file with exactly three columns: time in seconds,
channel x, channel y. Channel meanings must be checked against source metadata.

```bash
python generalized_hopf_deeponet.py --data original_state.txt --out experiment --epochs 60
```

Server invocation:

```bash
python <SERVER_RUN_DIR> --data '<SERVER_RUN_DIR> 31/original_state_Pt 31_FD31_1.txt' --out <SERVER_RUN_DIR> --epochs 60
```

Dependencies: Python, numpy, torch, matplotlib.

## Structure

A history encoder estimates a latent initial condition and beta, frequency,
noise intensity. The DeepONet branch/trunk product predicts a time-dependent
drift correction to a stochastic generalized Hopf model. A shared linear
decoder maps latent states to the two observed channels. The uncorrected Hopf
paths and corrected joint paths use the same Wiener increments. This makes
their path consistency loss meaningful. The corrected model is more flexible;
its dynamics are not guaranteed to retain the uncorrected Hopf bifurcation diagram.

Generalized Hopf drift: (beta + i*2*pi*f)*z + (1-beta)*|z|^2*z - |z|^4*z.
Beta is restricted to (-0.98,-0.02), so bistability is an imposed assumption,
not a discovered result. The potential parameters and latent scale need
identifiability checks before physical interpretation.

Training uses Monte Carlo mixture likelihood, Hopf-path likelihood, paired-path
consistency, current-state reconstruction and drift-correction regularization.
The finite-sample likelihood is a Monte Carlo approximation. It does not establish
calibrated uncertainty. Validation uses fixed noise. Checkpoints depend on local
trusted pickle data; load only checkpoints generated by this code or trusted sources.

## Data and validation

Non-overlapping five-sample block averages reduce aliasing, with timestamps set
to block ends to preserve availability times. Training/validation/test partitions
are chronological 60/20/20. Windows remain inside each partition. Scaling uses
training observations only. Default history: 40 averaged samples (~2 seconds).
Forecast: 20 samples (~1 second). All plotted test predictions are rolling-origin,
using newly observed history, not a single autonomous long rollout.

## Experimental risk

Risk is the sampled fraction of joint latent paths that enter a radius-0.3 disk
for at least three consecutive prediction steps. Paths already inside the disk
are counted as well: this is future occupancy/entry risk, not strictly first passage
from a known stepping state. The radius, dwell, probability threshold and confirmation
count are user-configurable assumptions, not validated clinical decision rules.
Process intervals omit observation noise and parameter uncertainty.

Optional `--events events.csv` accepts expert onset/offset times:

```csv
onset,offset
12.5,15.1
```

For supervised risk, origins within ongoing events are excluded; other origins
are positive when an onset falls within the forecast horizon. A differentiable
disk-occupancy surrogate supplies an auxiliary event loss. No events are inferred
from force amplitude. Event supervision is not clinical calibration and the current
script does not calculate event sensitivity, false alarms per hour or lead time.
One record may have too few events for meaningful supervised evaluation.

Outputs: model.pt (weights, preprocessing and configuration), metrics.json,
training.csv, risk.csv, predictions.npz and results.png. Predictions are reported
in original channel units. A trigger in risk.csv is experimental, not a validated
FoG warning. Generalization and the contribution of each module require ablations,
multiple records, held-out participants and actual event annotations.

## Reuse existing cycle-reference labels

Add `--reference-root ${FOG_REFERENCE_ROOT} --tag 'Pt 31_FD31_1'`.
This reads the project's canonical NPZ time,x,y rather than the --data text file.
The current default is the paper label rule described above. Add
`--reference-rule legacy --merge-gap-cycles 0` for the historical unmerged rule.
The generated complete intervals are saved as reference_events.csv. Events use
the full record offline: they are algorithmic references, not clinician labels
or causal detector outputs. Defaults in the source detector currently use a
1.13-second reference cycle, 1.5-cycle abnormal gap and onset at 1.2 cycles after
the previous peak. Terminal gaps and nearby-event merging are handled by that
same function.

The new model uses these events for auxiliary supervision. The source project's
original trajectory model did NOT use event labels in training. Current pilot
splits (60/20/20), candidate rule, and risk threshold are also different from the
paper project's 52/13/35 split and SE2/ME2 warning rules. Thus labels match, but
performance is not a replication or directly comparable paper evaluation.

## Normal-origin rolling prediction after a 65% prefix

```bash
python <SERVER_RUN_DIR> --data '<SERVER_RUN_DIR> 31/original_state_Pt 31_FD31_1.txt' --reference-root ${FOG_REFERENCE_ROOT} --tag 'Pt 31_FD31_1' --out <SERVER_RUN_DIR> --normal-only --train-fraction 0.65 --epochs 80
```

The first 65% contains the fit prefix (52%) and validation (13%). The remaining
35% is used for rolling prediction without weight updates. Preprocessing fits
only on the 52% fit prefix. Every new 20-Hz test origin must be outside reference
event intervals, including their boundaries. History may include earlier event
samples because only the origin, not the entire history, is required to be normal.
When the 65% boundary lies inside a reference event, inference waits until the
first normal sample. Prediction continues across later normal periods; it pauses
inside each reference event. A risk > 0.7 now issues immediately. A below-or-equal
value or an excluded-event gap re-arms the issuance latch. The first single
exceedance no longer needs two further rolling confirmations.

Important: this uses OFFLINE reference labels to select normal origins, per the
requested experiment. It is not an autonomous online normal-state detector.

Only warning issuance times and reference onset times are marked, with no
interval shading. An event is covered if an actual warning occurs strictly after
the previous reference event ends and strictly before its onset. There is no
maximum lead-time requirement. An event already in progress at the 65% boundary
is not a new target; subsequent onsets remain in the denominator even if there
are no eligible origins. Warnings without a subsequent observed onset are reported
as unresolved, not automatically as false alarms. Only currently available history
goes into the network; target signals and onset labels are not inference inputs.

Outputs additionally include warning_times.csv, event_warning_validation.csv,
warning_summary.json, train_fit.csv, warning_dashboard.png/pdf and
raw_force_warning.png/pdf. The normal-origin plots show gaps wherever no
prediction was issued. `normal_warning_protocol.py` implements evaluation and
plotting and must be placed beside generalized_hopf_deeponet.py.

The earlier `plot_hopf_warning.py` belongs to the earlier interval-display pilot;
do not use it to replot this normal-only protocol.

## Batch: canonical 21 records / 7 participants

`batch_parameter_hopf.py` reads the full manifest in the supplied reference root.
It verifies the canonical NPZ hashes, takes snapshots of the code, and trains one
independent parameter model per record using fixed settings. GPU worker processes
are launched with CUDA_VISIBLE_DEVICES and no model/optimizer is shared between
records. Failed runs are explicitly retained in status.json and summary.json.

```bash
python <SERVER_RUN_DIR> --reference-root ${FOG_REFERENCE_ROOT} --out <SERVER_RUN_DIR> --epochs 80 --gpus 0,1,2,3 --reference-rule legacy
```

`--reuse-single` accepts only results with matching explicit label configuration
and input identity. Old checkpoints without label metadata are rejected. Each record
is trained from seed 2026, with 80 maximum epochs and validation-based early stopping.
The test data are not used to select an alarm threshold. A new output directory
must be supplied for every new batch; existing status files prevent overwriting.

Each records/Pt_*/ directory contains model weights, risk trace, parameter curves,
reference intervals, event-level before-onset evaluation, and the three sets of
PNG/PDF plots. Top-level outputs: status.json, protocol.json, input_manifest.json,
per_record.csv, per_participant.csv, events.csv, warnings.csv, summary.json and
batch_summary.png/pdf. Pooled event coverage is the total covered / total events;
lead times use the first qualifying warning for each covered event, not all
repeat warnings. Warnings in the terminal normal segment without a later onset
are counted as unresolved. This is within-record conditional evaluation using
offline normal-state selection and algorithmic reference labels, not evaluation
of a model transferred to unseen people.

