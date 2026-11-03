# 1 / 2 / 3 / 5-s cumulative first-onset risks

Open `index.html` or read `REPORT_中文.md` for interpretation. The simulation
horizon remains 5 s. This experiment changes the **event horizon**, using the
same future paths for all four event targets. It does not fit four models.

The source model's 1-s event function rejects windows >=1.695 s. The new
`multihorizon_events.py` checks successive predicted cycles and records the
first NEW onset. Five synthetic behavioral checks cover immediate and later
gaps, regular cycles, already ongoing gaps and recovery followed by a new gap.
Every 1-s probability must exactly match the previous same-seed replay, and
every origin must have nested cumulative risks.

## Execution

Actual server directory:

`<SERVER_RUN_DIR>`

```bash
cd <SERVER_RUN_DIR>
python -u run_multihorizon.py \
  --source <SERVER_RUN_DIR> \
  --reference <SERVER_RUN_DIR> \
  --old-results <SERVER_RUN_DIR> \
  --out results
```

In a fresh extraction, `--old-results one_second_reference` uses the included
exact compatibility-reference CSVs. The original source checkpoints, force data
and model code must still be supplied through `--source` and `--reference`.
`source_hashes.json` verifies those source files. The included
`run_intervention.py` is an exact helper copy from the prior fixed-checkpoint
experiment; it supplies the solver, IO and seed utilities.

Four RTX A6000 GPUs were used. Paths per origin: 128 (one unique deterministic
path at zero noise). Batch size: 32. Recorded base seeds: 20261027, 20261028,
20261029; record offset is 1000 times the zero-based manifest index. Completed
record markers make the run resumable; choose a new output directory for an
entirely fresh run.

Run `python analyze_multihorizon.py` to regenerate figures and report. This
requires NumPy, SciPy, pandas and Matplotlib; inference additionally requires
PyTorch. The cached fit-prefix constant risks let the saved report regenerate
without the original checkpoints. Those constants were estimated from fit-only
normal origins, with event follow-up restricted to the fit prefix.

## Interpretation constraints

The prior saved primary run's exact full paths were not retained. These four
windows therefore all use a reproducible new path ensemble. Its 1-s result is
81/91 events and 149 alarms for seed 20261027; the saved primary stream had
82/91 and 146 alarms. This is Monte Carlo variation, not a new trained model.

The model was trained with a 1-s soft event objective. Longer-window cumulative
risks are a frozen-model extension and require their own calibration assessment.
Per-horizon outcome labels, complete follow-up and alarm-matching windows differ;
raw Brier, alarm precision and coverage cannot be naively ranked across targets.
Same-origin full-5-s follow-up diagnostics and common-1-s alarm coverage are
provided to make these distinctions visible.

The outcomes are cycle-derived reference anomalies, and normal-origin selection
remains retrospective. This analysis does not establish clinical FOG prediction
or identification of a physiological Hopf-bifurcation time.

