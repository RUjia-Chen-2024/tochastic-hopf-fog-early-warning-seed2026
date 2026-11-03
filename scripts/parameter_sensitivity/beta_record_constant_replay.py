"""Hold beta at a fit-prefix median per record during all test rollouts."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_raw(path):
    with np.load(path) as z:
        return np.column_stack([z["t"], z["x"], z["y"]]).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--draws", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    base, ref, evaluation, out = map(Path, (a.base_run, a.reference_root, a.evaluation, a.out))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    source_code = Path(__file__).resolve().parent
    sys.path.insert(0, str(ref))
    sys.path.insert(0, str(source_code))
    sys.path.append(str(base / "code"))
    from generalized_hopf_deeponet import ParameterHopfDeepONet
    from history_window import history_specification, network_histories
    from cycle_transition import observed_contexts
    from multihorizon_events import first_new_onsets, cumulative_risks
    from beta_sensitivity_replay import model_outputs, paired_paths, warning_summary, onset_targets

    device = torch.device(a.device)
    manifest = json.loads((base / "input_manifest.json").read_text(encoding="utf-8"))
    record_summaries, origin_rows = [], []
    for item in manifest:
        tag = item["tag"]
        record = tag.replace(" ", "_")
        src = base / "records" / record
        if not (src / "model.pt").exists():
            continue
        meta = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        ck = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
        cfg = ck.get("args", meta["configuration"])
        model = ParameterHopfDeepONet(
            int(cfg["history"]), int(cfg["horizon"]), float(ck["dt"]),
            cfg.get("observation_aggregation", "endpoint"),
            bool(cfg.get("constant_coefficients", False))
        ).to(device).eval()
        model.load_state_dict(ck["state_dict"])

        raw = load_raw(ref / "data" / "records" / f"{record}.npz")
        stride = int(cfg["stride"])
        blocks = raw[:len(raw) // stride * stride].reshape(-1, stride, 3)
        block_averages = blocks.mean(axis=1)
        block_averages[:, 0] = blocks[:, -1, 0]
        t, v = block_averages[:, 0], block_averages[:, 1:]
        n = len(t)
        b2 = int(.65 * n)
        b1 = int(.8 * b2)
        horizon = int(cfg["horizon"])
        history = int(cfg["history"])
        mean_np, scale_np = np.asarray(ck["mean"]), np.asarray(ck["scale"])
        normalized = ((v - mean_np) / scale_np).astype("float32")

        # Use only fitting-prefix input histories to establish beta* for this record.
        fit_spec = history_specification(
            raw[:, 0], stride, history, cfg.get("history_cycles"), 1.13
        )
        fit_ids = np.arange(max(0, history - 1), b1 - horizon, 2, dtype=int)
        if not len(fit_ids):
            raise ValueError(f"{record}: no training-prefix origins for beta*")
        fit_inputs_np = network_histories(
            normalized, raw, fit_ids, stride, mean_np, scale_np, fit_spec
        )
        fit_inputs = torch.as_tensor(fit_inputs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            fit_beta = []
            for lo in range(0, len(fit_inputs), 512):
                _, fit_current, _ = model_outputs(model, fit_inputs[lo:lo + 512])
                fit_beta.extend(fit_current[:, 0].detach().cpu().numpy().tolist())
        beta_star = float(np.median(fit_beta))

        saved = np.load(src / "network_inputs.npz")
        inputs = torch.as_tensor(saved["inputs"], dtype=torch.float32, device=device)
        times = saved["origin_time"].astype(float)
        block_times = blocks[:, -1, 0]
        ids = np.searchsorted(block_times, times)
        if np.any(ids >= len(block_times)) or not np.allclose(block_times[ids], times, atol=1e-8, rtol=0):
            raise ValueError(f"{record}: test origins do not align")
        settings = json.loads((src / "peak_calibration.json").read_text(encoding="utf-8"))
        contexts, ages = observed_contexts(raw, ids, stride, settings)
        events = [(float(r["onset"]), float(r["offset"]))
                  for r in read_rows(src / "reference_events.csv")]
        boundary, end_time, dt = float(meta["split"]["test_boundary"]), float(raw[-1, 0]), float(ck["dt"])
        mean = torch.as_tensor(ck["mean"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(ck["scale"], dtype=torch.float32, device=device)
        with torch.no_grad():
            z0, current, curve = model_outputs(model, inputs)

        # Seed and batch shape match the prior evaluation. Both scenarios use identical
        # per-substep Gaussian increments; beta* is fixed across all test origins in
        # this record and throughout every 5-s rollout.
        torch.manual_seed(2026)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(2026)
        risk_tv, risk_star = [], []
        for lo in range(0, len(inputs), a.batch_size):
            hi = min(lo + a.batch_size, len(inputs))
            fixed = torch.full((hi - lo,), beta_star, dtype=z0.dtype, device=device)
            ptv, pstar, fine_dt = paired_paths(
                model, z0[lo:hi], current[lo:hi], curve[lo:hi],
                a.draws, device, fixed_beta=fixed
            )
            ptv = (ptv * scale + mean).detach().cpu().numpy()
            pstar = (pstar * scale + mean).detach().cpu().numpy()
            first_tv = first_new_onsets(ptv, contexts[lo:hi], ages[lo:hi], settings, fine_dt)
            first_star = first_new_onsets(pstar, contexts[lo:hi], ages[lo:hi], settings, fine_dt)
            risk_tv.extend(cumulative_risks(first_tv)[:, 1].tolist())
            risk_star.extend(cumulative_risks(first_star)[:, 1].tolist())

        risk_tv, risk_star = np.asarray(risk_tv), np.asarray(risk_star)
        stats_tv, flags_tv = warning_summary(times, risk_tv, events, boundary, end_time, dt)
        stats_star, flags_star = warning_summary(times, risk_star, events, boundary, end_time, dt)
        record_summaries.extend([
            {"record": record, "scenario": "fitted beta(t)", "beta_star_fit_prefix": beta_star,
             "fit_origins": int(len(fit_beta)), **stats_tv},
            {"record": record, "scenario": "one beta* from fit prefix", "beta_star_fit_prefix": beta_star,
             "fit_origins": int(len(fit_beta)), **stats_star},
        ])
        reference_files = list(evaluation.glob(f"{record}_Full DeepONet*origins.csv"))
        risk_mae = None
        if reference_files:
            reference_risk = {round(float(r["origin_time"]), 8): float(r["risk"])
                              for r in read_rows(reference_files[0])}
            common = [(risk_tv[i], reference_risk[round(float(t), 8)])
                      for i, t in enumerate(times) if round(float(t), 8) in reference_risk]
            if common:
                risk_mae = float(np.mean([abs(x - y) for x, y in common]))
        for i, t in enumerate(times):
            origin_rows.append({
                "record": record, "origin_time": float(t), "future_2s_event": int(
                    onset_targets(np.asarray([t]), events, 2.0)[0]),
                "beta0": float(current[i, 0].detach().cpu()), "beta_star_fit_prefix": beta_star,
                "risk_fitted_beta_curve": float(risk_tv[i]),
                "risk_one_beta_star": float(risk_star[i]),
                "risk_change_beta_star_minus_curve": float(risk_star[i] - risk_tv[i]),
                "alarm_fitted_beta_curve": int(flags_tv[i]),
                "alarm_one_beta_star": int(flags_star[i]),
            })
        print(
            f"DONE {record}: beta_star={beta_star:.6f}, fit_origins={len(fit_beta)}, "
            f"test_origins={len(times)}, baseline_risk_MAE={risk_mae}",
            flush=True
        )

    write_rows(out / "beta_star_by_record.csv", record_summaries)
    write_rows(out / "origin_beta_star_replay.csv", origin_rows)
    # Pooled event coverage, alarm burden and Brier for the two paired scenarios.
    total_rows = []
    for scenario in ("fitted beta(t)", "one beta* from fit prefix"):
        selected = [r for r in record_summaries if r["scenario"] == scenario]
        total_events = sum(int(r["events"]) for r in selected)
        covered = sum(int(r["covered_events"]) for r in selected)
        alarms = sum(int(r["alarms"]) for r in selected)
        minutes = sum(int(r["origins"]) * float(
            json.loads((base / "records" / r["record"] / "metrics.json").read_text())["dt"]
        ) / 60.0 for r in selected)
        scored = sum(int(r["scored_origins"]) for r in selected)
        brier = sum(float(r["brier_2s"]) * int(r["scored_origins"]) for r in selected) / scored
        leads = [float(r["median_lead_within_2s"]) for r in selected
                 if r["median_lead_within_2s"] is not None]
        total_rows.append({
            "scenario": scenario, "records": len(selected), "test_events": total_events,
            "covered_events": covered, "event_coverage": covered / total_events,
            "alarms": alarms, "alarms_per_normal_minute": alarms / minutes,
            "median_of_record_median_leads_s": float(np.median(leads)),
            "brier_2s": brier, "scored_origins": scored,
            "alarm_origin_flags_changed": int(sum(
                int(r["alarm_fitted_beta_curve"]) != int(r["alarm_one_beta_star"])
                for r in origin_rows
            )) if scenario == "one beta* from fit prefix" else 0,
        })
    write_rows(out / "beta_star_replay_summary.csv", total_rows)
    protocol = {
        "seed": 2026, "network_history_cycles": 1.0, "simulator_horizon_seconds": 5.0,
        "paths": a.draws, "event_window_seconds": 2.0, "alarm_threshold": 0.8,
        "alarm_confirmation": "two consecutive 0.05-s origins",
        "beta_star": "For each record, median inferred beta(0) over model input histories ending in that record's fit prefix only; this scalar is then used for every test origin and all 5 s of each rollout.",
        "paired_controls": "Same z0, frequency(t), sigma(t), decoder and Gaussian increments; only beta differs.",
        "interpretation_limit": "Post-fit intervention. It tests dependence on record-specific, history-varying beta values; it does not establish physiological causality or unique parameter identification."
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("SUMMARY")
    for row in total_rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
