"""Refit the archived training protocol on CPU under two observation definitions."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ARMS = {
    "endpoint_blockrec": ("endpoint", "last_block_mean"),
    "blockmean_instantrec": ("block_mean", "instantaneous"),
}
TRAINING_OPTIONS = (
    "tag", "architecture", "epochs", "stride", "history", "horizon", "history_cycles",
    "draws", "radius", "dwell", "alarm_threshold", "confirm", "train_fraction",
    "validation_within_train", "risk_rule", "cycle", "reference_rule", "merge_gap_cycles",
    "event_horizon",
)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_record(engine, output, archive, reference, record, original, arm):
    aggregation, reconstruction = ARMS[arm]
    destination = output / arm / "records" / record
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(engine), "--data", str(reference / "data" / "records" / f"{record}.npz"),
               "--reference-root", str(reference), "--out", str(destination), "--observation-aggregation", aggregation,
               "--reconstruction-target", reconstruction, "--device", "cpu", "--cpu-threads", "1", "--fit-only"]
    for option in TRAINING_OPTIONS:
        if original[option] is not None:
            command.extend(("--" + option.replace("_", "-"), str(original[option])))
    if original["normal_only"]:
        command.append("--normal-only")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="2026")
    settings = dict(arm=arm, record=record, engine_sha256=file_digest(engine), arguments=command[3:])
    marker = destination / "FIT_COMPLETE.json"
    checkpoint = destination / "model.pt"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["settings"] == settings and saved["checkpoint_sha256"] == file_digest(checkpoint):
            return saved
        raise ValueError(f"Existing completed fit has different configuration: {arm}/{record}")
    started = time.perf_counter()
    with (destination / "fit.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - started
    archived = archive / "source_snapshot" / "records" / record
    for name in ("reference_events.csv", "peak_calibration.json"):
        if file_digest(destination / name) != file_digest(archived / name):
            raise ValueError(f"Reference artifact changed for {arm}/{record}: {name}")
    saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = saved_checkpoint["args"]
    if args["observation_aggregation"] != aggregation or args["reconstruction_target"] != reconstruction:
        raise ValueError(f"Unexpected checkpoint definition for {arm}/{record}")
    completed = dict(settings=settings, checkpoint_sha256=file_digest(checkpoint), elapsed_s=elapsed)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=28)
    options = parser.parse_args()
    archive, reference, out = options.archive.resolve(), options.reference.resolve(), options.out.resolve()
    if archive == out or archive in out.parents or options.workers < 1:
        parser.error("Use a separate output directory and at least one worker")
    engine = Path(__file__).resolve().parent / "training" / "generalized_hopf_deeponet.py"
    manifest = json.loads((archive / "source_snapshot" / "input_manifest.json").read_text())
    records = []
    for row in manifest:
        name = row["tag"].replace(" ", "_")
        data_path = reference / row["record"]
        if file_digest(data_path) != row["record_sha256"]:
            raise ValueError(f"Reference data differs from archived data for {name}")
        checkpoint_path = archive / "source_snapshot" / "records" / name / "model.pt"
        original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["args"]
        if original["observation_aggregation"] != "endpoint" or original["horizon"] != 100 or original["stride"] != 5:
            raise ValueError(f"Unexpected archived training definition for {name}")
        records.append((data_path.stat().st_size, name, original))
    records.sort(key=lambda item: item[0], reverse=True)
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {executor.submit(train_record, engine, out, archive, reference, record, original, arm): (arm, record)
                   for _, record, original in records for arm in ARMS}
        for future in as_completed(futures):
            arm, record = futures[future]
            result = future.result()
            print(f"{arm}/{record} complete in {result['elapsed_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
