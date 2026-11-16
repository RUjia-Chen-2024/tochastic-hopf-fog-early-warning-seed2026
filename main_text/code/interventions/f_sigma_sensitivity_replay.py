"""Paired post-fit replay freezing frequency or noise intensity at each origin.

The stochastic Hopf paths use paired Brownian increments. For a sigma freeze,
the same dW is multiplied by the original sigma(t) and by sigma(0), respectively.
This isolates the sigma schedule while preserving the random-number realization.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

SEED = 2026
HORIZONS = (1.0, 2.0, 3.0, 5.0)


def drift(z, beta, frequency):
    r2 = z.square().sum(-1, keepdim=True)
    radial = beta + (1.0 - beta) * r2 - r2.square()
    rotation = torch.stack((-z[..., 1], z[..., 0]), -1)
    return radial * z + 2.0 * math.pi * frequency * rotation


@torch.no_grad()
def paired_paths(model, z0, curve, current, draws, device, freeze):
    """Return baseline and counterfactual paths with common Brownian increments."""
    batch, horizon, _ = curve.shape
    substeps = max(1, int(math.ceil(model.dt / 0.01)))
    ds = model.dt / substeps
    z_base = z0[:, None, :].expand(-1, draws, -1).clone()
    z_cf = z_base.clone()
    nsteps = horizon * substeps
    base_path = torch.empty((batch, draws, nsteps, 2), device=device)
    cf_path = torch.empty_like(base_path)

    fixed_frequency = current[:, 1, None, None]
    fixed_sigma = current[:, 2, None, None]
    step = 0
    for k in range(horizon):
        beta = curve[:, k, 0, None, None]
        frequency = curve[:, k, 1, None, None]
        sigma = curve[:, k, 2, None, None]
        cf_frequency = fixed_frequency if freeze == "frequency" else frequency
        cf_sigma = fixed_sigma if freeze == "sigma" else sigma
        for _ in range(substeps):
            d_w = torch.randn_like(z_base) * math.sqrt(ds)
            d_base = drift(z_base, beta, frequency)
            d_cf = drift(z_cf, beta, cf_frequency)
            z_base = z_base + ds * d_base / (
                1.0 + ds * torch.linalg.vector_norm(d_base, dim=-1, keepdim=True)
            ) + d_w * sigma
            z_cf = z_cf + ds * d_cf / (
                1.0 + ds * torch.linalg.vector_norm(d_cf, dim=-1, keepdim=True)
            ) + d_w * cf_sigma
            base_path[:, :, step] = model.decoder(z_base)
            cf_path[:, :, step] = model.decoder(z_cf)
            step += 1
    return base_path, cf_path, ds


def first_onsets_and_first_peaks(force_paths, contexts, ages, settings, fine_dt):
    """Apply the reference accepted-peak rule; return first gap onset and peak."""
    paths = np.asarray(force_paths)
    batch, draws, steps, _ = paths.shape
    end = steps * fine_dt
    signal = np.abs(paths[..., 0] - paths[..., 1]) / settings["peak_scale"]
    first_onset = np.full((batch, draws), np.inf, dtype=np.float64)
    first_peak = np.full((batch, draws), np.inf, dtype=np.float64)
    gap = settings["abnormal_gap"]
    if abs(gap - settings["onset_delay"]) >= 1e-9:
        raise ValueError("abnormal_gap and onset_delay must match")
    for b, age in enumerate(ages):
        if not np.isfinite(age):
            first_onset[b] = np.nan
            first_peak[b] = np.nan
            continue
        history_length = len(contexts[b])
        minimum = 0.45 * settings["cycle"] - float(age)
        for m in range(draws):
            sequence = np.r_[contexts[b], signal[b, m]]
            peaks, _ = find_peaks(
                sequence, distance=settings["distance_samples"],
                height=settings["height"], prominence=settings["prominence"]
            )
            times = (peaks - history_length + 1) * fine_dt
            future = times[(peaks >= history_length) & (times >= minimum - 1e-9)]
            if len(future):
                first_peak[b, m] = float(future[0])
            last = -float(age)
            for next_peak in np.r_[future, np.inf]:
                deadline = last + gap
                if deadline > end + 1e-9:
                    break
                if deadline > 1e-9 and next_peak > deadline + 1e-9:
                    first_onset[b, m] = deadline
                    break
                if not np.isfinite(next_peak):
                    break
                last = float(next_peak)
    return first_onset, first_peak


def pooled_warning_summary(rows, manifests, base, ref, risk_key, flag_key):
    """Pool alarms, event coverage, exact pooled lead, and Brier score."""
    from beta_sensitivity_replay import load_raw, onset_targets, read_rows, warning_summary

    alarms = covered = events_n = scored = 0
    minutes = 0.0
    brier_terms, leads = [], []
    by_record = []
    for item in manifests:
        record = item["tag"].replace(" ", "_")
        rr = [r for r in rows if r["record"] == record]
        if not rr:
            continue
        src = base / "records" / record
        meta = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        raw = load_raw(ref / "data" / "records" / f"{record}.npz")
        events = [(float(x["onset"]), float(x["offset"]))
                  for x in read_rows(src / "reference_events.csv")]
        times = np.asarray([float(r["origin_time"]) for r in rr])
        risk = np.asarray([float(r[risk_key]) for r in rr])
        flags = np.asarray([int(r[flag_key]) for r in rr])
        end_time = float(raw[-1, 0])
        dt = float(meta["dt"])
        record_stats, _ = warning_summary(
            times, risk, events, float(meta["split"]["test_boundary"]), end_time, dt
        )
        by_record.append({"record": record, **record_stats})
        alarms += int(record_stats["alarms"])
        covered += int(record_stats["covered_events"])
        events_n += int(record_stats["events"])
        minutes += len(times) * dt / 60.0
        labels = onset_targets(times, events, 2.0)
        complete = times + 2.0 <= end_time + 1e-9
        score = np.isfinite(risk) & complete
        brier_terms.extend(((risk[score] - labels[score]) ** 2).tolist())
        scored += int(score.sum())
        alarm_times = times[flags == 1]
        boundary = float(meta["split"]["test_boundary"])
        for j, (onset, _) in enumerate(events):
            if onset < boundary - 1e-9:
                continue
            previous = events[j - 1][1] if j else -np.inf
            hits = [t for t in alarm_times
                    if previous < t < onset - 1e-9 and onset - t <= 2.0 + 1e-9]
            if hits:
                leads.append(float(onset - min(hits)))
    return {
        "records": len(by_record), "origins": len(rows), "test_events": events_n,
        "covered_events": covered,
        "event_coverage": covered / events_n if events_n else None,
        "alarms": alarms,
        "alarms_per_normal_minute": alarms / minutes if minutes else None,
        "pooled_median_lead_within_2s": float(np.median(leads)) if leads else None,
        "brier_2s": float(np.mean(brier_terms)) if brier_terms else None,
        "scored_origins": scored,
    }, by_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", choices=("frequency", "sigma"), required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--draws", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--records", help="Optional comma-separated record folder names for a smoke run")
    args = ap.parse_args()

    base, ref = Path(args.base_run), Path(args.reference_root)
    evaluation, out = Path(args.evaluation), Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    source_code = Path(__file__).resolve().parent
    sys.path.insert(0, str(ref))
    sys.path.insert(0, str(source_code))
    sys.path.append(str(base / "code"))
    from generalized_hopf_deeponet import ParameterHopfDeepONet
    from cycle_transition import observed_contexts
    from multihorizon_events import cumulative_risks, first_new_onsets
    from beta_sensitivity_replay import (
        load_raw, model_outputs, onset_targets, read_rows, warning_summary, write_rows
    )

    device = torch.device(args.device)
    manifest = json.loads((base / "input_manifest.json").read_text(encoding="utf-8"))
    if args.records:
        keep = set(args.records.split(","))
        manifest = [m for m in manifest if m["tag"].replace(" ", "_") in keep]
    if not manifest:
        raise ValueError("No records selected")
    origin_rows, replay_rows = [], []
    verified_baseline = {}
    pair_peak_abs, pair_peak_signed = [], []
    pair_onset_abs, pair_onset_signed = [], []
    peak_presence_mismatch = path_status = path_status_changed = 0
    path_status_counts = {h: {"baseline": 0, "frozen": 0, "changed": 0, "valid": 0}
                          for h in HORIZONS}

    for item in manifest:
        tag = item["tag"]
        record = tag.replace(" ", "_")
        src = base / "records" / record
        if not (src / "model.pt").exists():
            print(f"SKIP {record}: no fitted checkpoint", flush=True)
            continue
        meta = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        ck = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
        cfg = ck.get("args", meta["configuration"])
        if cfg.get("architecture") != "parameters":
            print(f"SKIP {record}: not a parameter model", flush=True)
            continue
        model = ParameterHopfDeepONet(
            int(cfg["history"]), int(cfg["horizon"]), float(ck["dt"]),
            cfg.get("observation_aggregation", "endpoint"),
            bool(cfg.get("constant_coefficients", False))
        ).to(device).eval()
        model.load_state_dict(ck["state_dict"])
        saved = np.load(src / "network_inputs.npz")
        times = saved["origin_time"].astype(float)
        inputs = torch.as_tensor(saved["inputs"], dtype=torch.float32, device=device)
        if len(times) != len(inputs):
            raise ValueError(f"{record}: origins and histories differ")
        raw = load_raw(ref / "data" / "records" / f"{record}.npz")
        stride = int(cfg["stride"])
        blocks = raw[:len(raw) // stride * stride].reshape(-1, stride, 3)
        block_times = blocks[:, -1, 0]
        ids = np.searchsorted(block_times, times)
        if np.any(ids >= len(block_times)) or not np.allclose(block_times[ids], times,
                                                                 atol=1e-8, rtol=0):
            raise ValueError(f"{record}: saved origins do not align to raw signal")
        settings = json.loads((src / "peak_calibration.json").read_text(encoding="utf-8"))
        contexts, ages = observed_contexts(raw, ids, stride, settings)
        events = [(float(r["onset"]), float(r["offset"]))
                  for r in read_rows(src / "reference_events.csv")]
        boundary = float(meta["split"]["test_boundary"])
        end_time, dt = float(raw[-1, 0]), float(ck["dt"])
        mean = torch.as_tensor(ck["mean"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(ck["scale"], dtype=torch.float32, device=device)
        with torch.no_grad():
            z0, current, curve = model_outputs(model, inputs)

        # Match the previously run protocol: same seed, record order and batch shape.
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        risk_base = {h: [] for h in HORIZONS}
        risk_frozen = {h: [] for h in HORIZONS}
        local_first_peaks_base, local_first_peaks_frozen = [], []
        local_first_onsets_base, local_first_onsets_frozen = [], []
        local_path_counts = {h: {"baseline": 0, "frozen": 0, "changed": 0, "valid": 0}
                             for h in HORIZONS}

        for lo in range(0, len(inputs), args.batch_size):
            hi = min(lo + args.batch_size, len(inputs))
            base_path, frozen_path, fine_dt = paired_paths(
                model, z0[lo:hi], curve[lo:hi], current[lo:hi], args.draws, device,
                args.freeze
            )
            base_path = (base_path * scale + mean).detach().cpu().numpy()
            frozen_path = (frozen_path * scale + mean).detach().cpu().numpy()
            first_base, peak_base = first_onsets_and_first_peaks(
                base_path, contexts[lo:hi], ages[lo:hi], settings, fine_dt
            )
            first_frozen, peak_frozen = first_onsets_and_first_peaks(
                frozen_path, contexts[lo:hi], ages[lo:hi], settings, fine_dt
            )
            # Check our combined event/peak extraction against the canonical event code.
            if record not in verified_baseline:
                canonical = first_new_onsets(base_path, contexts[lo:hi], ages[lo:hi],
                                             settings, fine_dt)
                if not np.allclose(first_base, canonical, equal_nan=True):
                    raise AssertionError(f"{record}: reference event extraction mismatch")
                verified_baseline[record] = True
            cumulative_base = cumulative_risks(first_base)
            cumulative_frozen = cumulative_risks(first_frozen)
            for horizon_idx, h in enumerate(HORIZONS):
                risk_base[h].extend(cumulative_base[:, horizon_idx].tolist())
                risk_frozen[h].extend(cumulative_frozen[:, horizon_idx].tolist())
                valid = (~np.isnan(first_base) & ~np.isnan(first_frozen))
                base_event = first_base <= h + 1e-9
                frozen_event = first_frozen <= h + 1e-9
                local_path_counts[h]["baseline"] += int(base_event[valid].sum())
                local_path_counts[h]["frozen"] += int(frozen_event[valid].sum())
                local_path_counts[h]["changed"] += int((base_event[valid] != frozen_event[valid]).sum())
                local_path_counts[h]["valid"] += int(valid.sum())
            local_first_peaks_base.append(peak_base)
            local_first_peaks_frozen.append(peak_frozen)
            local_first_onsets_base.append(first_base)
            local_first_onsets_frozen.append(first_frozen)
            del base_path, frozen_path

        first_peak_base = np.concatenate(local_first_peaks_base, axis=0)
        first_peak_frozen = np.concatenate(local_first_peaks_frozen, axis=0)
        first_onset_base = np.concatenate(local_first_onsets_base, axis=0)
        first_onset_frozen = np.concatenate(local_first_onsets_frozen, axis=0)
        risk_base = {h: np.asarray(v, dtype=float) for h, v in risk_base.items()}
        risk_frozen = {h: np.asarray(v, dtype=float) for h, v in risk_frozen.items()}

        for h in HORIZONS:
            path_status_counts[h]["baseline"] += local_path_counts[h]["baseline"]
            path_status_counts[h]["frozen"] += local_path_counts[h]["frozen"]
            path_status_counts[h]["changed"] += local_path_counts[h]["changed"]
            path_status_counts[h]["valid"] += local_path_counts[h]["valid"]
        both_peak = np.isfinite(first_peak_base) & np.isfinite(first_peak_frozen)
        only_peak = np.isfinite(first_peak_base) ^ np.isfinite(first_peak_frozen)
        peak_presence_mismatch += int(only_peak.sum())
        if both_peak.any():
            signed = first_peak_frozen[both_peak] - first_peak_base[both_peak]
            pair_peak_signed.extend(signed.tolist())
            pair_peak_abs.extend(np.abs(signed).tolist())
        both_onset = np.isfinite(first_onset_base) & np.isfinite(first_onset_frozen)
        if both_onset.any():
            signed = first_onset_frozen[both_onset] - first_onset_base[both_onset]
            pair_onset_signed.extend(signed.tolist())
            pair_onset_abs.extend(np.abs(signed).tolist())

        labels = onset_targets(times, events, 2.0)
        record_rows = []
        for i, t in enumerate(times):
            row = {
                "record": record, "origin_time": float(t),
                "future_2s_event": int(labels[i]),
                "beta0": float(current[i, 0].detach().cpu()),
                "frequency0_hz": float(current[i, 1].detach().cpu()),
                "sigma0": float(current[i, 2].detach().cpu()),
                "first_peak_base_count": int(np.isfinite(first_peak_base[i]).sum()),
                "first_peak_frozen_count": int(np.isfinite(first_peak_frozen[i]).sum()),
                "first_peak_presence_mismatch_count": int(only_peak[i].sum()),
                "median_abs_first_peak_shift_s": (
                    float(np.median(np.abs(first_peak_frozen[i, both_peak[i]] -
                                           first_peak_base[i, both_peak[i]])))
                    if both_peak[i].any() else None
                ),
            }
            for h in HORIZONS:
                tag_h = str(int(h))
                rb, rf = float(risk_base[h][i]), float(risk_frozen[h][i])
                row[f"risk_{tag_h}s_baseline"] = rb
                row[f"risk_{tag_h}s_frozen"] = rf
                row[f"risk_{tag_h}s_delta"] = rf - rb
            record_rows.append(row)
        origin_rows.extend(record_rows)

        # Paired alarm flags are evaluated at the requested 2-s event horizon.
        stats_base, flags_base = warning_summary(times, risk_base[2.0], events, boundary,
                                                 end_time, dt)
        stats_frozen, flags_frozen = warning_summary(times, risk_frozen[2.0], events,
                                                     boundary, end_time, dt)
        replay_rows.extend([
            {"record": record, "scenario": "fitted parameter curve", **stats_base},
            {"record": record, "scenario": f"{args.freeze}(s) fixed at origin value",
             **stats_frozen},
        ])
        local_flags_changed = int(np.sum(flags_base != flags_frozen))
        ref_files = list(evaluation.glob(f"{record}_Full DeepONet*origins.csv"))
        baseline_mae = None
        if ref_files:
            ref_rows = read_rows(ref_files[0])
            ref_map = {round(float(r["origin_time"]), 8): float(r["risk"]) for r in ref_rows}
            paired = [(risk_base[2.0][i], ref_map[round(float(t), 8)])
                      for i, t in enumerate(times) if round(float(t), 8) in ref_map]
            if paired:
                baseline_mae = float(np.mean([abs(a - b) for a, b in paired]))
        flags_key = {
            "alarm_baseline": flags_base,
            "alarm_frozen": flags_frozen,
        }
        for i, row in enumerate(record_rows):
            row["alarm_baseline"] = int(flags_key["alarm_baseline"][i])
            row["alarm_frozen"] = int(flags_key["alarm_frozen"][i])
        # update the appended rows with alarm columns
        origin_rows[-len(record_rows):] = record_rows
        print(f"DONE {record}: origins={len(times)}, alarm_flags_changed={local_flags_changed}, "
              f"baseline_risk_MAE_vs_seed2026={baseline_mae}", flush=True)

    if not origin_rows:
        raise RuntimeError("No fitted parameter records were processed")

    freeze_label = "frequency" if args.freeze == "frequency" else "sigma"
    risk_rows = []
    for h in HORIZONS:
        baseline = np.asarray([float(r[f"risk_{int(h)}s_baseline"]) for r in origin_rows])
        frozen = np.asarray([float(r[f"risk_{int(h)}s_frozen"]) for r in origin_rows])
        valid = np.isfinite(baseline) & np.isfinite(frozen)
        delta = np.abs(frozen[valid] - baseline[valid])
        risk_rows.append({
            "horizon_s": h, "origins": int(valid.sum()),
            "risk_changed_origins": int(np.sum(delta > 0)),
            "risk_changed_fraction": float(np.mean(delta > 0)),
            "mean_absolute_risk_change": float(np.mean(delta)),
            "median_absolute_risk_change": float(np.median(delta)),
            "q90_absolute_risk_change": float(np.quantile(delta, .9)),
            "q95_absolute_risk_change": float(np.quantile(delta, .95)),
            "max_absolute_risk_change": float(np.max(delta)),
            "baseline_mean_risk": float(np.mean(baseline[valid])),
            "frozen_mean_risk": float(np.mean(frozen[valid])),
            "baseline_paths_with_event": int(path_status_counts[h]["baseline"]),
            "frozen_paths_with_event": int(path_status_counts[h]["frozen"]),
            "paired_path_event_status_changed": int(path_status_counts[h]["changed"]),
            "paired_path_event_status_changed_fraction": float(
                path_status_counts[h]["changed"] / path_status_counts[h]["valid"]
            ),
            "valid_paired_paths": int(path_status_counts[h]["valid"]),
        })

    alarms_changed = sum(int(r["alarm_baseline"] != r["alarm_frozen"]) for r in origin_rows)
    baseline_summary, baseline_record = pooled_warning_summary(
        origin_rows, manifest, base, ref, "risk_2s_baseline", "alarm_baseline"
    )
    frozen_summary, frozen_record = pooled_warning_summary(
        origin_rows, manifest, base, ref, "risk_2s_frozen", "alarm_frozen"
    )
    comparison = [
        {"scenario": "fitted parameter curve", **baseline_summary,
         "alarm_origin_flags_changed_vs_baseline": 0},
        {"scenario": f"{freeze_label}(s) fixed at each origin's {freeze_label}(0)",
         **frozen_summary, "alarm_origin_flags_changed_vs_baseline": alarms_changed},
    ]

    finite_peak_shift = np.asarray(pair_peak_abs, dtype=float)
    signed_peak_shift = np.asarray(pair_peak_signed, dtype=float)
    finite_onset_shift = np.asarray(pair_onset_abs, dtype=float)
    signed_onset_shift = np.asarray(pair_onset_signed, dtype=float)
    timing = [{
        "paired_paths_with_first_accepted_peak_in_both": int(len(finite_peak_shift)),
        "paired_paths_with_peak_presence_mismatch": int(peak_presence_mismatch),
        "median_abs_first_peak_shift_s": float(np.median(finite_peak_shift)) if len(finite_peak_shift) else None,
        "q90_abs_first_peak_shift_s": float(np.quantile(finite_peak_shift, .9)) if len(finite_peak_shift) else None,
        "q95_abs_first_peak_shift_s": float(np.quantile(finite_peak_shift, .95)) if len(finite_peak_shift) else None,
        "median_signed_first_peak_shift_frozen_minus_baseline_s": float(np.median(signed_peak_shift)) if len(signed_peak_shift) else None,
        "paired_paths_with_first_onset_in_both": int(len(finite_onset_shift)),
        "median_abs_first_onset_shift_s": float(np.median(finite_onset_shift)) if len(finite_onset_shift) else None,
        "q90_abs_first_onset_shift_s": float(np.quantile(finite_onset_shift, .9)) if len(finite_onset_shift) else None,
        "median_signed_first_onset_shift_frozen_minus_baseline_s": float(np.median(signed_onset_shift)) if len(signed_onset_shift) else None,
    }]

    write_rows(out / "origin_replay.csv", origin_rows)
    write_rows(out / "metrics_by_record.csv", replay_rows)
    write_rows(out / "summary.csv", comparison)
    write_rows(out / "risk_and_path_events_by_horizon.csv", risk_rows)
    write_rows(out / "peak_timing_summary.csv", timing)
    write_rows(out / "warning_metrics_by_record.csv", baseline_record + frozen_record)
    protocol = {
        "seed": SEED, "records": len(manifest), "history_cycles": 1.0,
        "simulator_horizon_seconds": 5.0, "paths_per_origin": args.draws,
        "event_probability_horizons_seconds": list(HORIZONS),
        "primary_alarm_horizon_seconds": 2.0, "alarm_threshold": 0.8,
        "alarm_confirmation": "two consecutive 0.05-s origins",
        "intervention": f"Freeze {freeze_label}(s) to the current origin value {freeze_label}(0) across that forecast's full 5-s rollout; retain initial state and other parameter curves.",
        "paired_noise": "Use the same standard Brownian increments dW for baseline and counterfactual. For sigma intervention, multiply that shared dW by sigma(t) in the baseline and sigma(0) in the counterfactual.",
        "peak_timing": "First future accepted peak is detected with the same peak calibration, height, prominence, distance, and cycle-age filter used in the event label.",
        "interpretation_limit": "Post-fit paired sensitivity analysis; not a refit, parameter identifiability test, or physiological causal claim.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"freeze": freeze_label, "summary": comparison,
                      "risk_by_horizon": risk_rows, "peak_timing": timing}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
