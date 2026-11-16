"""Evaluate retrained checkpoints at common test origins with common random draws."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import random
from types import SimpleNamespace

# Use the original test split, causal input construction, and alarm scoring.
from run_continuous import (
    BATCH, CONFIRM, DRAWS, DT, EPS, HORIZON, SEED, THRESHOLD,
    digest, policy_rows, read_csv, read_json, record_names, summarize,
    test_inputs, use_archive_code, write_csv, write_json,
)

import numpy as np
import torch


POLICIES = ("reference_normal", "continuous")
READOUTS = ("endpoint", "block_mean", "persistence")
FORCE_WINDOWS = (1., 2., 3., 5.)
MATCHED_CONFIG = (
    "architecture", "stride", "history", "history_cycles", "horizon", "epochs", "draws",
    "risk_rule", "cycle", "reference_rule", "merge_gap_cycles", "event_horizon",
    "train_fraction", "validation_within_train", "normal_only", "alarm_threshold", "confirm",
)


def checkpoint_path(train_root, name):
    return train_root / "records" / name / "model.pt"


def reference_masks(times, events):
    active = np.zeros(len(times), dtype=bool)
    labels = np.zeros(len(times), dtype=bool)
    for onset, offset in events:
        active |= (times >= onset - EPS) & (times <= offset + EPS)
        labels |= (onset - times > EPS) & (onset - times <= HORIZON + EPS)
    return active, labels


def archived_events(archive, name):
    source = archive / "source_snapshot" / "records" / name
    return [(float(r["onset"]), float(r["offset"])) for r in read_csv(source / "reference_events.csv")]


def verify_checkpoint(checkpoint, archived, name):
    config = checkpoint["args"]
    original = archived["configuration"]
    changed = [key for key in MATCHED_CONFIG if config.get(key) != original[key]]
    if changed:
        raise ValueError(f"Unexpected training protocol changes for {name}: {changed}")
    if config.get("constant_coefficients", False) or config.get("event_loss_weight", .2) != .2:
        raise ValueError(f"Changed dynamics or event objective for {name}")
    readout = config["observation_aggregation"]
    reconstruction = config.get("reconstruction_target", "last_block_mean")
    if (readout, reconstruction) not in {
        ("endpoint", "last_block_mean"), ("block_mean", "instantaneous")
    }:
        raise ValueError(f"Unexpected observation and reconstruction targets for {name}")
    if not np.isclose(float(checkpoint["dt"]), DT, rtol=0, atol=EPS):
        raise ValueError(f"Unexpected time step for {name}")
    return readout, reconstruction


def infer_group(archive, train_root, out, names, code_hashes):
    use_archive_code(archive)
    from cycle_transition import fit_peak_settings, observed_contexts
    from frozen_solver import simulate
    from generalized_hopf_deeponet import make_model
    from multihorizon_events import first_new_onsets

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for name in names:
        source = archive / "source_snapshot" / "records" / name
        raw_path = archive / "reference_snapshot" / "data" / "records" / f"{name}.npz"
        calibration_path = source / "peak_calibration.json"
        checkpoint_file = checkpoint_path(train_root, name)
        checkpoint_hash = digest(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        original_metrics = read_json(source / "metrics.json")
        readout, reconstruction = verify_checkpoint(checkpoint, original_metrics, name)
        args = SimpleNamespace(**checkpoint["args"])
        provenance = dict(record=name, checkpoint_sha256=checkpoint_hash, raw_sha256=digest(raw_path),
                          calibration_sha256=digest(calibration_path), code_sha256=code_hashes,
                          seed=SEED, draws=DRAWS, batch_size=BATCH, torch=str(torch.__version__),
                          training_readout=readout, reconstruction_target=reconstruction)
        path = out / "inference" / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as prior:
                if json.loads(prior["provenance"].item()) != provenance:
                    raise ValueError(f"Existing replay inputs differ for {name}: {path}")
            print(f"{name}: verified existing replay", flush=True)
            continue

        with np.load(raw_path, allow_pickle=False) as record:
            raw = np.column_stack([record[key] for key in ("t", "x", "y")])
        dt = float(checkpoint["dt"])
        mean, scale = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["scale"])
        block_count = len(raw) // args.stride
        blocks = raw[:block_count * args.stride].reshape(block_count, args.stride, 3)
        forces = blocks.mean(1)[:, 1:]
        test_start = int(args.train_fraction * block_count)
        fit_end = int((1 - args.validation_within_train) * test_start)
        np.testing.assert_array_equal(mean, forces[:fit_end].mean(0))
        np.testing.assert_array_equal(scale, forces[:fit_end].std(0).clip(1e-6))
        settings = read_json(calibration_path)
        if fit_peak_settings(raw, fit_end * args.stride, args.cycle, args.reference_rule) != settings:
            raise ValueError(f"Peak calibration changed for {name}")
        ids, times, histories = test_inputs(raw, args, mean, scale)
        if abs(times[0] - original_metrics["split"]["test_boundary"]) > EPS:
            raise ValueError(f"Test boundary changed for {name}")

        # Recreate the float32 standardisation and inverse transform of the
        # archived truth arrays. Retain NaNs at unavailable future blocks.
        targets = ids[:, None] + np.arange(1, args.horizon + 1)
        normalized = ((forces - mean) / scale).astype(np.float32)
        truth = normalized[targets.clip(max=block_count - 1)] * scale + mean
        truth[targets >= block_count] = np.nan

        contexts, ages = observed_contexts(raw, ids, args.stride, settings)
        model = make_model(args, dt).eval()
        model.load_state_dict(checkpoint["state_dict"])
        inputs = torch.as_tensor(histories, dtype=torch.float32)
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        forecasts = {"endpoint": [], "block_mean": []}
        onsets = []
        for start in range(0, len(times), BATCH):
            stop = min(start + BATCH, len(times))
            decoded, _, _, fine_dt = simulate(model, inputs[start:stop], DRAWS, 1.)
            if not np.isclose(fine_dt, .01, rtol=0, atol=EPS):
                raise ValueError(f"Expected five 100 Hz values per prediction block for {name}")
            fine_force = (decoded * scale_tensor + mean_tensor).cpu().numpy()
            onsets.append(first_new_onsets(fine_force, contexts[start:stop], ages[start:stop], settings, fine_dt))
            force_blocks = fine_force.reshape(stop - start, DRAWS, args.horizon, args.stride, 2)
            forecasts["endpoint"].append(force_blocks[:, :, :, -1, :].mean(1))
            forecasts["block_mean"].append(force_blocks.mean(3).mean(1))
        first = np.concatenate(onsets)
        risk = (first <= HORIZON + EPS).mean(1)
        risk[np.isnan(first).any(1)] = np.nan
        if digest(checkpoint_file) != checkpoint_hash:
            raise ValueError(f"Checkpoint changed during replay for {name}")

        temporary = path.with_name(f".{name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, origin_time=times, normalized_history=histories,
                peak_age=ages, risk_2s=risk, first_new_onset_offsets=first,
                endpoint=np.concatenate(forecasts["endpoint"]),
                block_mean=np.concatenate(forecasts["block_mean"]), truth=truth,
                current_force=forces[ids], horizon_seconds=np.arange(1, args.horizon + 1) * dt,
                raw_end=raw[-1, 0], dt=dt, provenance=json.dumps(provenance),
            )
        temporary.replace(path)
        print(f"{name}: evaluated {len(times)} test origins", flush=True)


def force_error_rows(name, policy, times, use, events, data):
    truth = data["truth"]
    horizon = data["horizon_seconds"]
    starts = np.asarray(sorted(start for start, _ in events))
    next_start = np.r_[starts, np.inf][np.searchsorted(starts, times, side="right")]
    crossing = next_start[:, None] <= times[:, None] + horizon[None, :] + EPS
    persistence = np.broadcast_to(data["current_force"][:, None, :], truth.shape)
    predictions = {"endpoint": data["endpoint"], "block_mean": data["block_mean"], "persistence": persistence}
    valid = np.isfinite(truth) & use[:, None, None]
    if any(not np.isfinite(prediction[valid]).all() for prediction in predictions.values()):
        raise ValueError(f"Nonfinite predicted force at a valid target for {name}")
    for readout, prediction in predictions.items():
        error = prediction.astype(np.float64) - truth
        for scope, subset in (("all_targets", valid), ("onset_crossing", valid & crossing[:, :, None])):
            for seconds in FORCE_WINDOWS:
                mask = subset & (horizon <= seconds + EPS)[None, :, None]
                values = error[mask]
                count = values.size
                abs_sum = float(np.abs(values).sum())
                sq_sum = float(np.square(values).sum())
                yield dict(record=name, policy=policy, readout=readout, target_scope=scope,
                           horizon_s=seconds, scalar_targets=count, absolute_error_sum=abs_sum,
                           squared_error_sum=sq_sum, mae=abs_sum / count if count else None,
                           rmse=np.sqrt(sq_sum / count).item() if count else None)


def pooled_force_errors(per_record):
    groups = {}
    for row in per_record:
        key = (row["policy"], row["readout"], row["target_scope"], row["horizon_s"])
        groups.setdefault(key, []).append(row)
    for (policy, readout, scope, horizon), rows in groups.items():
        count = sum(row["scalar_targets"] for row in rows)
        abs_sum = sum(row["absolute_error_sum"] for row in rows)
        sq_sum = sum(row["squared_error_sum"] for row in rows)
        yield dict(policy=policy, readout=readout, target_scope=scope, horizon_s=horizon,
                   records=len(rows), scalar_targets=count, mae=abs_sum / count if count else None,
                   rmse=np.sqrt(sq_sum / count).item() if count else None)


def score(archive, out, names):
    use_archive_code(archive)
    from generalized_hopf_deeponet import threshold_warnings

    counts_by_policy = {policy: [] for policy in POLICIES}
    alarms, event_rows, origin_rows, record_rows, force_rows, audits = [], [], [], [], [], []
    for name in names:
        events = archived_events(archive, name)
        source = archive / "source_snapshot" / "records" / name
        with np.load(out / "inference" / f"{name}.npz", allow_pickle=False) as result:
            data = dict(result)
        times, risk = data["origin_time"], data["risk_2s"]
        dt, raw_end = float(data["dt"]), float(data["raw_end"])
        active, labels = reference_masks(times, events)
        normal = ~active
        complete = times + HORIZON <= raw_end + EPS
        with np.load(source / "network_inputs.npz", allow_pickle=False) as original:
            np.testing.assert_array_equal(times[normal], original["origin_time"])
            np.testing.assert_array_equal(data["normalized_history"][normal], original["inputs"])
        with np.load(source / "predictions.npz", allow_pickle=False) as original:
            np.testing.assert_array_equal(times[normal], original["origin_time"])
            np.testing.assert_array_equal(data["truth"][normal], original["truth"])
        audits.append(json.loads(data["provenance"].item()))
        all_flags = {}
        for policy, use in (("reference_normal", normal), ("continuous", np.ones(len(times), dtype=bool))):
            flags = np.zeros(len(times), dtype=bool)
            flags[use] = threshold_warnings(times[use], risk[use], THRESHOLD, CONFIRM, dt)
            all_flags[policy] = flags
            counts, warnings, refs = policy_rows(name, policy, times, risk, flags, use, active,
                                                 labels, complete, events, float(times[0]), dt)
            counts_by_policy[policy].append(counts)
            alarms.extend(warnings)
            event_rows.extend(refs)
            record_rows.append(dict(record=name, **summarize(policy, [counts], warnings, refs)))
            force_rows.extend(force_error_rows(name, policy, times, use, events, data))
        for i, time in enumerate(times):
            origin_rows.append(dict(record=name, origin_time=float(time), risk_2s=float(risk[i]),
                                    peak_age_s=float(data["peak_age"][i]), reference_active=int(active[i]),
                                    label_2s=int(labels[i]), complete_2s=int(complete[i]),
                                    reference_normal_alarm=int(all_flags["reference_normal"][i]),
                                    continuous_alarm=int(all_flags["continuous"][i])))

    variants = {(r["training_readout"], r["reconstruction_target"]) for r in audits}
    if len(variants) != 1:
        raise ValueError("Mixed retraining configurations in one replay")
    training_readout, reconstruction_target = variants.pop()
    summaries = [summarize(policy, counts_by_policy[policy],
                           [r for r in alarms if r["policy"] == policy],
                           [r for r in event_rows if r["policy"] == policy]) for policy in POLICIES]
    force_summary = list(pooled_force_errors(force_rows))
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "per_record.csv", record_rows)
    write_csv(out / "origins.csv", origin_rows)
    if alarms:
        write_csv(out / "alarms.csv", alarms)
    write_csv(out / "events.csv", event_rows)
    write_csv(out / "force_summary.csv", force_summary)
    write_csv(out / "force_per_record.csv", force_rows)
    write_json(out / "audit.json", audits)
    report = dict(training_readout=training_readout, reconstruction_target=reconstruction_target,
                  backend="cpu", records=len(names), seed=SEED, paths=DRAWS, batch_size=BATCH,
                  forecast_s=5., alarm_horizon_s=HORIZON, threshold=THRESHOLD, confirmation=CONFIRM,
                  warning=summaries, force_5s=[r for r in force_summary if r["horizon_s"] == 5.])
    write_json(out / "comparison.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--record", action="append", help="Select records for a pilot replay")
    args = parser.parse_args()
    archive, train_root, out = args.archive.resolve(), args.train_root.resolve(), args.out.resolve()
    if out == archive or archive in out.parents or out == train_root:
        parser.error("Use a separate replay directory outside the original archive")
    if not 1 <= args.cpu_workers <= (os.cpu_count() or 1):
        parser.error("Invalid CPU worker count")
    names = record_names(archive)
    if args.record:
        if not set(args.record) <= set(names):
            parser.error("Unknown record in --record")
        names = [name for name in names if name in args.record]
    missing = [name for name in names if not checkpoint_path(train_root, name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing retrained checkpoints: {missing}")
    out.joinpath("inference").mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name("run_continuous.py"),
               archive / "frozen_solver.py", archive / "multihorizon_events.py"]
    sources.extend(sorted((archive / "source_snapshot" / "code").glob("*.py")))
    code_hashes = {path.name: digest(path) for path in sources}
    workers = min(args.cpu_workers, len(names))
    if workers == 1:
        infer_group(archive, train_root, out, names, code_hashes)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(infer_group, archive, train_root, out, names[i::workers], code_hashes)
                       for i in range(workers)]
            for future in futures:
                future.result()
    score(archive, out, names)


if __name__ == "__main__":
    main()
