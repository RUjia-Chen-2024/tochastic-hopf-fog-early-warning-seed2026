# Latest 5-s endpoint: process-diffusion intervention

Read `REPORT_中文.md` or open `index.html` for the full analysis.

This experiment does not retrain models. The main learned-noise comparator is
the exact saved primary evaluation (82/91 within 1 s; 146 alarms). New outputs
include a zero-diffusion intervention and three learned-diffusion replays using
the same latest 5-s endpoint checkpoints. There is no old 1.5-s checkpoint in
this experiment.

## Server execution

Server experiment directory:

`<SERVER_RUN_DIR>`

```bash
cd <SERVER_RUN_DIR>
python -u run_intervention.py \
  --source <SERVER_RUN_DIR> \
  --reference <SERVER_RUN_DIR> \
  --out results --workers 4 --batch-size 32
python validate_zero_cpu.py
```

Per-record COMPLETE markers make the main script resumable. To perform a fresh
run, choose a new `--out` directory. The CPU-audit script targets `results`.
Checkpoint, data and source hashes are checked against `source_hashes.json`.
The source project is intentionally not duplicated in this output package.

Packages: Python 3, NumPy, SciPy, PyTorch, Matplotlib; pandas for the report.
Server inference used PyTorch 2.6.0+cu124 and four NVIDIA RTX A6000 GPUs.
The separate CPU audit exactly reproduced every zero-diffusion origin risk.

## Reporting

Run `python analyze_results.py` from any environment containing NumPy, pandas
and Matplotlib to regenerate plots, tables, HTML and manuscript snippets from
the saved result CSVs. `manuscript_replacement.tex` contains proposed result,
methods, caption, abstract and discussion replacements. The existing manuscript
files have not been overwritten.

All event targets refer to algorithmically derived cycle anomalies, and origin
selection remains the same retrospective reference-normal screening used in
the primary evaluation. No new clinical annotation or held-out-person training
test is introduced. The fixed-checkpoint contrast must not be described as a
comparison against a separately retrained deterministic model.

