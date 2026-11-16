"""Replay the frozen seed-2026 model at every test time, then score offline."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch


SEED = 2026
BATCH = 32
DRAWS = 128
DT = 0.05
HORIZON = 2.0
THRESHOLD = 0.80
CONFIRM = 2
EPS = 1e-9
POLICIES = ("archived_gated", "gated_same_risk", "continuous")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def use_archive_code(archive):
    sys.path[:0] = [str(archive / "source_snapshot" / "code"), str(archive)]


def record_names(archive):
    manifest = read_json(archive / "source_snapshot" / "input_manifest.json")
    names = [row["tag"].replace(" ", "_") for row in manifest]
    if not names or len(names) != len(set(names)):
        raise ValueError("Record manifest is empty or has duplicate tags")
    return names


def test_inputs(raw, args, mean, scale):
    from history_window import history_specification, network_histories

    count = len(raw) // args.stride
    blocks = raw[: count * args.stride].reshape(count, args.stride, 3)
    block_times = blocks[:, -1, 0]
    if not np.isfinite(raw).all() or not np.allclose(np.diff(block_times), DT, rtol=0, atol=1e-5):
        raise ValueError("Expected the original regularly sampled force record")
    split = int(args.train_fraction * count)
    ids = np.arange(split, count, dtype=int)
    spec = history_specification(raw[:, 0], args.stride, args.history, args.history_cycles, args.cycle)
    if not len(ids) or spec["input_blocks"] != args.history:
        raise ValueError("Unexpected test split or checkpoint history shape")
    normalized_blocks = ((blocks.mean(1)[:, 1:] - mean) / scale).astype(np.float32)
    histories = network_histories(normalized_blocks, raw, ids, args.stride, mean, scale, spec)
    return ids, block_times[ids], histories


def infer_group(archive, out, names, device_name, code_hashes):
    use_archive_code(archive)
    from cycle_transition import observed_contexts
    from frozen_solver import simulate
    from generalized_hopf_deeponet import make_model, threshold_warnings
    from multihorizon_events import first_new_onsets

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    expected_hashes = read_json(archive / "source_hashes.json")
    for name in names:
        source = archive / "source_snapshot" / "records" / name
        checkpoint_path = source / "model.pt"
        raw_path = archive / "reference_snapshot" / "data" / "records" / f"{name}.npz"
        calibration_path = source / "peak_calibration.json"
        checkpoint_hash = digest(checkpoint_path)
        if expected_hashes[f"records/{name}/model.pt"] != checkpoint_hash:
            raise ValueError(f"Checkpoint differs from the archived experiment for {name}")
        provenance = {
            "record": name,
            "checkpoint_sha256": checkpoint_hash,
            "raw_sha256": digest(raw_path),
            "calibration_sha256": digest(calibration_path),
            "code_sha256": code_hashes,
            "seed": SEED,
            "batch_size": BATCH,
            "draws": DRAWS,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "backend": device.type,
            "device_type": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        }
        result_path = out / "inference" / f"{name}.npz"
        if result_path.exists():
            with np.load(result_path, allow_pickle=False) as saved:
                if json.loads(saved["provenance"].item()) != provenance:
                    raise ValueError(f"Existing result has different inputs or code for {name}")
            print(f"{name} verified existing inference", flush=True)
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**checkpoint["args"])
        dt = float(checkpoint["dt"])
        if (args.stride, args.history_cycles, args.horizon, args.observation_aggregation) != (5, 1.0, 100, "endpoint"):
            raise ValueError(f"Expected the frozen one-cycle, 5 s endpoint model for {name}")
        if not np.isclose(dt, DT, rtol=0, atol=EPS):
            raise ValueError(f"Checkpoint step differs from 50 ms for {name}")
        settings = read_json(calibration_path)
        with np.load(raw_path, allow_pickle=False) as data:
            raw = np.column_stack([data[key] for key in ("t", "x", "y")])
        mean = np.asarray(checkpoint["mean"])
        scale = np.asarray(checkpoint["scale"])
        ids, times, histories = test_inputs(raw, args, mean, scale)
        contexts, ages = observed_contexts(raw, ids, args.stride, settings)
        model = make_model(args, dt).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
        inputs = torch.as_tensor(histories, dtype=torch.float32, device=device)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        first_offsets = []
        for start in range(0, len(times), BATCH):
            stop = min(start + BATCH, len(times))
            decoded, _, _, fine_dt = simulate(model, inputs[start:stop], DRAWS, 1.0)
            force = (decoded * scale_tensor + mean_tensor).cpu().numpy()
            first_offsets.append(first_new_onsets(force, contexts[start:stop], ages[start:stop], settings, fine_dt))
        first_offsets = np.concatenate(first_offsets)
        risk = np.mean(first_offsets <= HORIZON + EPS, axis=1)
        risk[np.isnan(first_offsets).any(axis=1)] = np.nan
        flags = np.asarray(threshold_warnings(times, risk, THRESHOLD, CONFIRM, dt), dtype=bool)
        if digest(checkpoint_path) != checkpoint_hash:
            raise ValueError(f"Checkpoint changed while inferring {name}")
        temporary = result_path.with_name(f".{name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, origin_time=times, normalized_history=histories, peak_age=ages,
                risk_2s=risk, continuous_alarm=flags, first_new_onset_offsets=first_offsets,
                raw_end=raw[-1, 0], test_boundary=times[0], dt=dt, provenance=json.dumps(provenance),
            )
        temporary.replace(result_path)
        print(f"{name} {device_name} origins={len(times)} alarms={flags.sum()}", flush=True)


def infer(archive, out, names, devices):
    out.joinpath("inference").mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__), archive / "frozen_solver.py", archive / "multihorizon_events.py"]
    code_files.extend(sorted((archive / "source_snapshot" / "code").glob("*.py")))
    code_hashes = {path.name: digest(path) for path in code_files}
    jobs = [(archive, out, names[index::len(devices)], device, code_hashes)
            for index, device in enumerate(devices) if index < len(names)]
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as executor:
        results = [executor.submit(infer_group, *job) for job in jobs]
        for result in results:
            result.result()


def policy_rows(name, policy, times, risk, flags, use, active, labels, complete, events, boundary, dt):
    from normal_warning_protocol import assess_warnings

    # Preserve the paper's event-matching criterion for all controllers.
    normal = use & ~active
    _, normal_alarms, references = assess_warnings(times[normal], flags[normal], events, boundary)
    normal_alarm_times = np.array([alarm["warning_time"] for alarm in normal_alarms])
    onsets = np.array([start for start, _ in events])
    alarms = []
    for index in np.flatnonzero(use & flags):
        time = times[index]
        later = onsets[onsets > time + EPS]
        lead = float(later[0] - time) if len(later) else None
        matched = not active[index] and lead is not None and lead <= HORIZON + EPS
        match = "matched" if matched else "unmatched" if complete[index] else "tail_censored"
        alarms.append(dict(record=name, policy=policy, warning_time=float(time),
                           reference_active=int(active[index]), complete_2s=int(complete[index]),
                           next_onset=float(later[0]) if len(later) else None, lead_s=lead, match=match))
    event_rows = []
    for ref in references:
        onset = ref["reference_onset"]
        previous = ref["previous_reference_end"]
        previous = -np.inf if previous is None else previous
        leads = onset - normal_alarm_times
        hits = normal_alarm_times[(normal_alarm_times > previous + EPS) & (leads > EPS) & (leads <= HORIZON + EPS)]
        first = float(hits.min()) if len(hits) else None
        event_rows.append(dict(record=name, policy=policy, reference_onset=onset,
                               reference_end=ref["reference_end"], eligible_origins=ref["eligible_origins"],
                               covered=int(first is not None), first_alarm=first,
                               lead_s=onset - first if first is not None else None))
    scored = use & complete & np.isfinite(risk)
    normal_scored = scored & ~active
    active_scored = scored & active
    errors = np.square(risk - labels)
    counts = dict(origins=int(use.sum()), full_test_origins=len(times),
                  gap_resets=int((np.diff(times[use]) > 1.5 * dt).sum()),
                  scored_origins=int(scored.sum()), scored_normal=int(normal_scored.sum()),
                  scored_active=int(active_scored.sum()), unknown_risks=int((use & ~np.isfinite(risk)).sum()),
                  positive_targets=int(labels[scored].sum()), brier_sum=float(errors[scored].sum()),
                  brier_normal_sum=float(errors[normal_scored].sum()), brier_active_sum=float(errors[active_scored].sum()))
    return counts, alarms, event_rows


def summarize(policy, counts, alarms, events):
    totals = {key: sum(row[key] for row in counts) for key in counts[0]}
    matched = sum(row["match"] == "matched" for row in alarms)
    unmatched = sum(row["match"] == "unmatched" for row in alarms)
    tail = sum(row["match"] == "tail_censored" for row in alarms)
    leads = [row["lead_s"] for row in events if row["covered"]]
    seconds = totals["origins"] * DT
    full_seconds = totals["full_test_origins"] * DT

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return dict(policy=policy, records=len(counts), origins=totals["origins"], exposure_s=round(seconds, 6),
                gap_resets=totals["gap_resets"], events=len(events), covered=len(leads),
                event_coverage=ratio(len(leads), len(events)), median_lead_s=float(np.median(leads)) if leads else None,
                alarms=len(alarms), matched=matched, unmatched=unmatched, tail_censored=tail,
                alarms_inside_reference_events=sum(row["reference_active"] for row in alarms),
                unmatched_inside_reference_events=sum(row["reference_active"] and row["match"] == "unmatched" for row in alarms),
                matched_fraction=ratio(matched, matched + unmatched), alarms_per_minute=ratio(60 * len(alarms), seconds),
                unmatched_per_minute=ratio(60 * unmatched, seconds),
                unmatched_per_full_test_minute=ratio(60 * unmatched, full_seconds),
                scored_origins=totals["scored_origins"], unknown_risks=totals["unknown_risks"],
                target_prevalence=ratio(totals["positive_targets"], totals["scored_origins"]),
                brier=ratio(totals["brier_sum"], totals["scored_origins"]),
                scored_normal=totals["scored_normal"], brier_normal=ratio(totals["brier_normal_sum"], totals["scored_normal"]),
                scored_active=totals["scored_active"], brier_active=ratio(totals["brier_active_sum"], totals["scored_active"]))


def score(archive, out, names):
    use_archive_code(archive)
    from generalized_hopf_deeponet import threshold_warnings

    all_counts = {policy: [] for policy in POLICIES}
    all_alarms = []
    all_events = []
    origin_rows = []
    per_record = []
    audits = []
    changes = []
    for name in names:
        source = archive / "source_snapshot" / "records" / name
        stored = archive / "results" / "records" / name
        with np.load(out / "inference" / f"{name}.npz", allow_pickle=False) as data:
            times = data["origin_time"]
            risk = data["risk_2s"]
            continuous_flags = data["continuous_alarm"]
            histories = data["normalized_history"]
            ages = data["peak_age"]
            raw_end, boundary, dt = map(float, (data["raw_end"], data["test_boundary"], data["dt"]))
            provenance = json.loads(data["provenance"].item())
        events = [(float(row["onset"]), float(row["offset"])) for row in read_csv(source / "reference_events.csv")]
        active = np.zeros(len(times), dtype=bool)
        labels = np.zeros(len(times), dtype=bool)
        for start, end in events:
            active |= (times >= start - EPS) & (times <= end + EPS)
            labels |= (start - times > EPS) & (start - times <= HORIZON + EPS)
        normal = ~active
        complete = times + HORIZON <= raw_end + EPS
        full_mask = np.ones(len(times), dtype=bool)
        original_boundary = read_json(source / "metrics.json")["split"]["test_boundary"]
        if abs(boundary - original_boundary) > EPS:
            raise ValueError(f"Test boundary differs from archived evaluation for {name}")
        if read_json(stored / "COMPLETE.json")["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint differs from the original evaluation for {name}")
        with np.load(source / "network_inputs.npz", allow_pickle=False) as original:
            np.testing.assert_array_equal(times[normal], original["origin_time"])
            np.testing.assert_array_equal(histories[normal], original["inputs"])
        archived = [row for row in read_csv(stored / "origins.csv") if row["variant"] == "learned_seed_2026"]
        np.testing.assert_array_equal(times[normal], [float(row["origin_time"]) for row in archived])
        np.testing.assert_array_equal(labels[normal], [int(row["label_2s"]) for row in archived])
        np.testing.assert_array_equal(complete[normal], [int(row["complete_2s"]) for row in archived])
        archived_risk = np.full(len(times), np.nan)
        archived_risk[normal] = [float(row["risk_2s"]) for row in archived]
        archived_flags = np.zeros(len(times), dtype=bool)
        archived_flags[normal] = [int(row["alarm_2s"]) for row in archived]
        np.testing.assert_array_equal(archived_flags[normal], threshold_warnings(times[normal], archived_risk[normal], THRESHOLD, CONFIRM, dt))
        np.testing.assert_array_equal(continuous_flags, threshold_warnings(times, risk, THRESHOLD, CONFIRM, dt))
        gated_flags = np.zeros(len(times), dtype=bool)
        gated_flags[normal] = threshold_warnings(times[normal], risk[normal], THRESHOLD, CONFIRM, dt)
        configurations = (
            ("archived_gated", archived_risk, archived_flags, normal),
            ("gated_same_risk", risk, gated_flags, normal),
            ("continuous", risk, continuous_flags, full_mask),
        )
        for policy, probabilities, flags, use in configurations:
            counts, alarms, event_rows = policy_rows(name, policy, times, probabilities, flags, use,
                                                    active, labels, complete, events, boundary, dt)
            all_counts[policy].append(counts)
            all_alarms.extend(alarms)
            all_events.extend(event_rows)
            per_record.append(dict(record=name, **summarize(policy, [counts], alarms, event_rows)))
        for index, time in enumerate(times):
            origin_rows.append(dict(record=name, origin_time=float(time), peak_age_s=float(ages[index]),
                                    reference_active=int(active[index]), label_2s=int(labels[index]),
                                    complete_2s=int(complete[index]), risk_2s=float(risk[index]),
                                    continuous_alarm=int(continuous_flags[index]),
                                    gated_same_risk_alarm=int(gated_flags[index]),
                                    archived_risk_2s=float(archived_risk[index]) if normal[index] else None,
                                    archived_alarm=int(archived_flags[index]) if normal[index] else None))
            if normal[index] and continuous_flags[index] != gated_flags[index]:
                changes.append(dict(record=name, origin_time=float(time), risk_2s=float(risk[index]),
                                    continuous_alarm=int(continuous_flags[index]), gated_same_risk_alarm=int(gated_flags[index])))
        audits.append(dict(record=name, full_origins=len(times), reference_normal_origins=int(normal.sum()),
                           archived_inputs_match_exactly=True, archived_checkpoint_matches=True,
                           archived_labels_and_flags_match_exactly=True, raw_test_span_s=raw_end-boundary,
                           provenance=provenance))

    summary = [summarize(policy, all_counts[policy],
                         [row for row in all_alarms if row["policy"] == policy],
                         [row for row in all_events if row["policy"] == policy]) for policy in POLICIES]
    saved = next(row for row in read_csv(archive / "results" / "noise_by_horizon.csv")
                 if row["variant"] == "learned_seed_2026" and float(row["horizon_s"]) == HORIZON)
    for key, original_key in [("covered", "covered_within_horizon"), ("events", "events"), ("alarms", "alarms"),
                              ("matched", "matched"), ("unmatched", "unmatched"), ("tail_censored", "tail_censored"),
                              ("median_lead_s", "median_lead_s"), ("brier", "brier"), ("scored_origins", "scored_origins")]:
        if not np.isclose(summary[0][key], float(saved[original_key]), rtol=0, atol=EPS):
            raise ValueError(f"Archived {key} did not reproduce the published result")

    events_by_policy = {policy: {(row["record"], row["reference_onset"]): row for row in all_events
                                 if row["policy"] == policy} for policy in POLICIES}
    if not (events_by_policy[POLICIES[0]].keys() == events_by_policy[POLICIES[1]].keys() == events_by_policy[POLICIES[2]].keys()):
        raise ValueError("Policies have different reference-event denominators")
    pairs = []
    for key, full in events_by_policy["continuous"].items():
        gated = events_by_policy["gated_same_risk"][key]
        old = events_by_policy["archived_gated"][key]
        pairs.append(dict(record=key[0], reference_onset=key[1], continuous_covered=full["covered"],
                          gated_same_risk_covered=gated["covered"], archived_covered=old["covered"],
                          continuous_lead_s=full["lead_s"], gated_same_risk_lead_s=gated["lead_s"],
                          archived_lead_s=old["lead_s"], paired_lead_change_s=full["lead_s"] - gated["lead_s"]
                          if full["covered"] and gated["covered"] else None))
    paired_leads = [row["paired_lead_change_s"] for row in pairs if row["paired_lead_change_s"] is not None]
    paired = dict(both=len(paired_leads), continuous_only=sum(row["continuous_covered"] and not row["gated_same_risk_covered"] for row in pairs),
                  gated_only=sum(row["gated_same_risk_covered"] and not row["continuous_covered"] for row in pairs),
                  neither=sum(not row["continuous_covered"] and not row["gated_same_risk_covered"] for row in pairs),
                  median_lead_change_both_s=float(np.median(paired_leads)) if paired_leads else None,
                  normal_alarms_added=sum(row["continuous_alarm"] for row in changes),
                  normal_alarms_removed=sum(row["gated_same_risk_alarm"] for row in changes))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_record.csv", per_record)
    write_csv(out / "origins.csv", origin_rows)
    write_csv(out / "alarms.csv", all_alarms)
    write_csv(out / "events.csv", all_events)
    write_csv(out / "paired_events.csv", pairs)
    if changes:
        write_csv(out / "normal_alarm_changes.csv", changes)
    write_json(out / "audit.json", audits)
    backends = {row["provenance"]["backend"] for row in audits}
    if len(backends) != 1:
        raise ValueError("Results must come from a single inference backend")
    report = dict(records=len(names), inference_backend=backends.pop(),
                  full_test_raw_span_s=sum(row["raw_test_span_s"] for row in audits),
                  forecast_s=5.0, event_horizon_s=HORIZON, update_step_s=DT, risk_threshold=THRESHOLD,
                  confirmation_count=CONFIRM, paths=DRAWS, seed=SEED, batch_size=BATCH,
                  paired_controller_effect=paired, summary=summary)
    write_json(out / "comparison.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Original seed-2026 run with the frozen snapshots")
    parser.add_argument("--out", required=True, type=Path, help="Independent output directory for this replay")
    phases = parser.add_subparsers(dest="phase", required=True)
    inference = phases.add_parser("infer", help="Replay all test origins without reading reference-event labels")
    devices_group = inference.add_mutually_exclusive_group()
    devices_group.add_argument("--gpus", help="Comma-separated CUDA device IDs, default 0,1,2,3")
    devices_group.add_argument("--cpu-workers", type=int, help="Use this number of one-thread CPU processes")
    phases.add_parser("score", help="Compare continuous and retrospectively gated policies")
    args = parser.parse_args()
    archive, out = args.archive.resolve(), args.out.resolve()
    if out == archive or archive in out.parents:
        parser.error("Use an independent output directory outside the archived experiment")
    names = record_names(archive)
    if args.phase == "infer":
        if args.cpu_workers is not None:
            if not 1 <= args.cpu_workers <= (os.cpu_count() or 1):
                parser.error("CPU worker count must be between 1 and the number of available CPUs")
            devices = ["cpu"] * args.cpu_workers
        else:
            try:
                gpus = [int(item) for item in (args.gpus or "0,1,2,3").split(",")]
            except ValueError:
                parser.error("GPU IDs must be comma-separated integers")
            if not torch.cuda.is_available():
                parser.error("CUDA initialization failed. Use --cpu-workers for a CPU replay.")
            if len(set(gpus)) != len(gpus) or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in gpus):
                parser.error("GPU IDs must be unique and available")
            devices = [f"cuda:{gpu}" for gpu in gpus]
        infer(archive, out, names, devices)
    else:
        score(archive, out, names)


if __name__ == "__main__":
    main()
