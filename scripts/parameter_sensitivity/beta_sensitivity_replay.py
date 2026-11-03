"""Paired beta-only sensitivity replay for the fitted full DeepONet-Hopf models.

For every normal test origin, hold z0, frequency(s), sigma(s), decoder and
Brownian increments fixed; compare the fitted beta(tau) with beta(0) held
constant over the same 5-s rollout. This is a post-fit sensitivity analysis,
not a refit or causal-identification experiment.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAM_RANGES = {
    "beta": (-0.98, -0.02),
    "frequency_hz": (0.1, 3.0),
    "sigma": (0.01, 0.4),
}
SEED = 2026


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


def onset_targets(times, events, horizon=2.0):
    return np.asarray(
        [int(any(t + 1e-9 < s <= t + horizon + 1e-9 for s, _ in events))
         for t in times], dtype=int
    )


def model_outputs(model, x):
    h = model.encoder(x.flatten(1))
    z0 = 1.5 * torch.tanh(model.initial(h))
    p0 = model.param_head(h)
    if model.constant_coefficients:
        curve = model.physical_parameters(p0)[:, None, :].expand(
            -1, model.horizon, -1
        )
    else:
        branch = model.branch(h).reshape(len(x), 3, 32)
        trunk = model.trunk(model.tau).reshape(model.horizon, 3, 32)
        delta = torch.einsum("bcp,tcp->btc", branch, trunk) / np.sqrt(32)
        logits = p0[:, None, :] + model.tau[None, :, :] * delta
        curve = model.physical_parameters(logits)
    current = model.physical_parameters(p0)
    return z0, current, curve


def drift(z, beta, frequency):
    r2 = z.square().sum(-1, keepdim=True)
    radial = beta + (1.0 - beta) * r2 - r2.square()
    rotation = torch.stack((-z[..., 1], z[..., 0]), -1)
    return radial * z + 2.0 * math.pi * frequency * rotation


@torch.no_grad()
def paired_paths(model, z0, current, curve, draws, device, fixed_beta=None):
    """Generate paired time-varying-beta and beta(0)-fixed force paths."""
    batch, horizon, _ = curve.shape
    substeps = max(1, int(math.ceil(model.dt / 0.01)))
    ds = model.dt / substeps
    z_tv = z0[:, None, :].expand(-1, draws, -1).clone()
    z_fx = z_tv.clone()
    nsteps = horizon * substeps
    path_tv = torch.empty((batch, draws, nsteps, 2), device=device)
    path_fx = torch.empty_like(path_tv)
    if fixed_beta is None:
        beta0 = current[:, 0, None, None]
    else:
        beta0 = torch.as_tensor(fixed_beta, dtype=z0.dtype, device=device).reshape(
            batch, 1, 1
        )
    step = 0
    for k in range(horizon):
        beta_tv = curve[:, k, 0, None, None]
        freq = curve[:, k, 1, None, None]
        sigma = curve[:, k, 2, None, None]
        for _ in range(substeps):
            # One draw is shared by both counterfactual trajectories.
            noise = torch.randn_like(z_tv) * math.sqrt(ds) * sigma
            d_tv = drift(z_tv, beta_tv, freq)
            d_fx = drift(z_fx, beta0, freq)
            z_tv = z_tv + ds * d_tv / (
                1.0 + ds * torch.linalg.vector_norm(d_tv, dim=-1, keepdim=True)
            ) + noise
            z_fx = z_fx + ds * d_fx / (
                1.0 + ds * torch.linalg.vector_norm(d_fx, dim=-1, keepdim=True)
            ) + noise
            path_tv[:, :, step] = model.decoder(z_tv)
            path_fx[:, :, step] = model.decoder(z_fx)
            step += 1
    return path_tv, path_fx, ds


def warning_summary(times, risk, events, boundary, end_time, dt):
    from generalized_hopf_deeponet import threshold_warnings
    labels = onset_targets(times, events, 2.0)
    flags = np.asarray(threshold_warnings(times, risk, 0.8, 2, dt), dtype=int)
    alarm_times = times[flags == 1]
    event_rows = []
    for j, (onset, offset) in enumerate(events):
        if onset < boundary - 1e-9:
            continue
        previous_end = events[j - 1][1] if j else -np.inf
        hits = [t for t in alarm_times
                if previous_end < t < onset - 1e-9 and onset - t <= 2.0 + 1e-9]
        event_rows.append({
            "reference_onset": float(onset),
            "covered_within_2s": int(bool(hits)),
            "lead_seconds": float(onset - min(hits)) if hits else None,
        })
    complete = times + 2.0 <= end_time + 1e-9
    score = np.isfinite(risk) & complete
    leads = [r["lead_seconds"] for r in event_rows if r["lead_seconds"] is not None]
    return {
        "origins": int(len(times)),
        "events": int(len(event_rows)),
        "covered_events": int(sum(r["covered_within_2s"] for r in event_rows)),
        "event_coverage": (float(sum(r["covered_within_2s"] for r in event_rows) /
                                  len(event_rows)) if event_rows else None),
        "alarms": int(len(alarm_times)),
        "alarms_per_normal_minute": float(len(alarm_times) / (len(times) * dt / 60.0))
        if len(times) else None,
        "median_lead_within_2s": float(np.median(leads)) if leads else None,
        "brier_2s": float(np.mean((risk[score] - labels[score]) ** 2)) if score.any() else None,
        "scored_origins": int(score.sum()),
        "mean_risk": float(np.nanmean(risk)) if np.isfinite(risk).any() else None,
    }, flags


def qstats(values, prefix):
    a = np.asarray(values, dtype=float)
    return {
        prefix + "_median": float(np.nanmedian(a)),
        prefix + "_q10": float(np.nanquantile(a, .10)),
        prefix + "_q90": float(np.nanquantile(a, .90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--records", help="Optional comma-separated record folders for a smoke run")
    ap.add_argument("--draws", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    base, ref, out = Path(a.base_run), Path(a.reference_root), Path(a.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    source_code = Path(__file__).resolve().parent
    sys.path.insert(0, str(ref))
    sys.path.insert(0, str(source_code))
    sys.path.append(str(base / "code"))
    from generalized_hopf_deeponet import ParameterHopfDeepONet
    from cycle_transition import observed_contexts
    from multihorizon_events import first_new_onsets, cumulative_risks

    device = torch.device(a.device)
    manifest = json.loads((base / "input_manifest.json").read_text(encoding="utf-8"))
    if a.records:
        keep = set(a.records.split(","))
        manifest = [m for m in manifest if m["tag"].replace(" ", "_") in keep]
    if not manifest:
        raise ValueError("No records selected")

    per_record, origin_rows, basin_rows, replay_rows = [], [], [], []
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
            raise ValueError(f"{record}: origins and histories differ in length")
        raw = load_raw(ref / "data" / "records" / f"{record}.npz")
        stride = int(cfg["stride"])
        blocks = raw[:len(raw) // stride * stride].reshape(-1, stride, 3)
        block_times = blocks[:, -1, 0]
        ids = np.searchsorted(block_times, times)
        if np.any(ids >= len(block_times)) or not np.allclose(
                block_times[ids], times, atol=1e-8, rtol=0):
            raise ValueError(f"{record}: saved origins do not align to raw signal")
        settings = json.loads((src / "peak_calibration.json").read_text(encoding="utf-8"))
        contexts, ages = observed_contexts(raw, ids, stride, settings)
        event_rows = read_rows(src / "reference_events.csv")
        events = [(float(r["onset"]), float(r["offset"])) for r in event_rows]
        boundary = float(meta["split"]["test_boundary"])
        end_time = float(raw[-1, 0])
        dt = float(ck["dt"])
        mean = torch.as_tensor(ck["mean"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(ck["scale"], dtype=torch.float32, device=device)

        with torch.no_grad():
            z0, current, curve = model_outputs(model, inputs)
        radii = torch.linalg.vector_norm(z0, dim=-1).detach().cpu().numpy()
        beta0 = current[:, 0].detach().cpu().numpy()
        ru0 = np.sqrt(-beta0)
        labels = onset_targets(times, events, 2.0)
        curve_np = curve.detach().cpu().numpy()
        future_delta = curve_np[:, -1, :] - curve_np[:, 0, :]
        future_range = curve_np.max(axis=1) - curve_np.min(axis=1)

        ranges = {}
        for j, name in enumerate(("beta", "frequency_hz", "sigma")):
            lo, hi = PARAM_RANGES[name]
            values = curve_np[:, :, j]
            near = (values <= lo + .01 * (hi - lo)) | (values >= hi - .01 * (hi - lo))
            ranges[name] = float(near.mean())
        p_row = {"record": record, "participant": tag.split("_FD")[0].replace("Pt ", "Pt_")}
        p_row.update(qstats(beta0, "beta0"))
        p_row.update(qstats(current[:, 1].detach().cpu().numpy(), "frequency0_hz"))
        p_row.update(qstats(current[:, 2].detach().cpu().numpy(), "sigma0"))
        p_row.update(qstats(future_delta[:, 0], "beta_end_minus_start"))
        p_row.update(qstats(future_range[:, 0], "beta_5s_range"))
        p_row.update(qstats(future_range[:, 1], "frequency_5s_range_hz"))
        p_row.update(qstats(future_range[:, 2], "sigma_5s_range"))
        p_row.update({
            "beta_near_1pct_either_bound_fraction": ranges["beta"],
            "frequency_near_1pct_either_bound_fraction": ranges["frequency_hz"],
            "sigma_near_1pct_either_bound_fraction": ranges["sigma"],
            "origins": int(len(times)),
            "r0_below_ru_fraction": float(np.mean(radii < ru0)),
        })
        per_record.append(p_row)

        # Use the same seed, record order, batch size and per-substep RNG calls
        # as the original 2-s replay. The intervention shares each Gaussian draw.
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        risk_tv, risk_fx = [], []
        for lo in range(0, len(inputs), a.batch_size):
            hi = min(lo + a.batch_size, len(inputs))
            ptv, pfx, fine_dt = paired_paths(
                model, z0[lo:hi], current[lo:hi], curve[lo:hi], a.draws, device
            )
            ptv = (ptv * scale + mean).detach().cpu().numpy()
            pfx = (pfx * scale + mean).detach().cpu().numpy()
            first_tv = first_new_onsets(ptv, contexts[lo:hi], ages[lo:hi], settings, fine_dt)
            first_fx = first_new_onsets(pfx, contexts[lo:hi], ages[lo:hi], settings, fine_dt)
            risk_tv.extend(cumulative_risks(first_tv)[:, 1].tolist())
            risk_fx.extend(cumulative_risks(first_fx)[:, 1].tolist())
            del ptv, pfx
        risk_tv = np.asarray(risk_tv, dtype=float)
        risk_fx = np.asarray(risk_fx, dtype=float)
        stats_tv, flags_tv = warning_summary(times, risk_tv, events, boundary, end_time, dt)
        stats_fx, flags_fx = warning_summary(times, risk_fx, events, boundary, end_time, dt)
        replay_rows.extend([
            {"record": record, "scenario": "fitted beta(t)", **stats_tv},
            {"record": record, "scenario": "beta fixed at beta(0)", **stats_fx},
        ])

        # Compare the paired beta(t) replay with the already reported seed-2026
        # full-model risks. An exact match is a useful protocol verification.
        ref_files = list((Path(a.out).parent / "evaluation").glob(
            f"{record}_Full DeepONet*origins.csv"
        ))
        ref_mae = None
        if ref_files:
            ref_rows = read_rows(ref_files[0])
            ref_map = {round(float(r["origin_time"]), 8): float(r["risk"]) for r in ref_rows}
            shared = [(risk_tv[i], ref_map[round(float(t), 8)])
                      for i, t in enumerate(times) if round(float(t), 8) in ref_map]
            if shared:
                ref_mae = float(np.mean([abs(x - y) for x, y in shared]))

        for i, t in enumerate(times):
            origin_rows.append({
                "record": record, "origin_time": float(t), "peak_age": float(ages[i]),
                "future_2s_event": int(labels[i]), "r0": float(radii[i]),
                "beta0": float(beta0[i]), "ru0_sqrt_minus_beta": float(ru0[i]),
                "frequency0_hz": float(current[i, 1].detach().cpu()),
                "sigma0": float(current[i, 2].detach().cpu()),
                "r0_minus_ru0": float(radii[i] - ru0[i]),
                "inside_fixed_parameter_basin": int(radii[i] < ru0[i]),
                "beta_end_minus_start_5s": float(future_delta[i, 0]),
                "beta_range_5s": float(future_range[i, 0]),
                "risk_beta_time_varying": float(risk_tv[i]),
                "risk_beta_fixed_at_origin": float(risk_fx[i]),
                "risk_change_fixed_minus_varying": float(risk_fx[i] - risk_tv[i]),
                "alarm_beta_time_varying": int(flags_tv[i]),
                "alarm_beta_fixed_at_origin": int(flags_fx[i]),
            })
        for positive in (0, 1):
            chosen = labels == positive
            if chosen.any():
                basin_rows.append({
                    "record": record,
                    "future_2s_event": positive,
                    "origins": int(chosen.sum()),
                    "inside_fraction": float(np.mean(radii[chosen] < ru0[chosen])),
                    "median_r0": float(np.median(radii[chosen])),
                    "median_ru0": float(np.median(ru0[chosen])),
                    "median_r0_over_ru0": float(np.median(radii[chosen] / ru0[chosen])),
                    "mean_risk_beta_time_varying": float(np.mean(risk_tv[chosen])),
                    "mean_risk_beta_fixed": float(np.mean(risk_fx[chosen])),
                })
        print(
            f"DONE {record}: origins={len(times)}, inside={np.mean(radii < ru0):.3f}, "
            f"beta_replay_MAE_vs_seed2026={ref_mae}",
            flush=True
        )

    if not per_record:
        raise RuntimeError("No records processed")
    write_rows(out / "parameter_by_record.csv", per_record)
    write_rows(out / "origin_diagnostics.csv", origin_rows)
    write_rows(out / "basin_by_event_label.csv", basin_rows)
    write_rows(out / "beta_replay_by_record.csv", replay_rows)

    overall = []
    for scenario, risk_key in (("fitted beta(t)", "risk_beta_time_varying"),
                               ("beta fixed at beta(0)", "risk_beta_fixed_at_origin")):
        # The per-origin risk table permits pooled Brier and aggregate alert metrics.
        for record in sorted({r["record"] for r in origin_rows}):
            rr = [r for r in origin_rows if r["record"] == record]
            src_meta = json.loads((base / "records" / record / "metrics.json").read_text())
            src_raw = load_raw(ref / "data" / "records" / f"{record}.npz")
            events = [(float(x["onset"]), float(x["offset"])) for x in
                      read_rows(base / "records" / record / "reference_events.csv")]
            boundary = float(src_meta["split"]["test_boundary"])
            dt = float(src_meta["dt"])
            times = np.asarray([float(r["origin_time"]) for r in rr])
            risk = np.asarray([float(r[risk_key]) for r in rr])
            labels = onset_targets(times, events, 2.0)
            # Aggregate separately below; record rows are useful for paired checks.
            summary, _ = warning_summary(times, risk, events, boundary, float(src_raw[-1, 0]), dt)
            overall.append({"scenario": scenario, "record": record, **summary})
    write_rows(out / "beta_replay_metrics_record.csv", overall)

    # Aggregate all scored origins and event opportunities by scenario.
    total_rows = []
    for scenario, risk_key in (("fitted beta(t)", "risk_beta_time_varying"),
                               ("beta fixed at beta(0)", "risk_beta_fixed_at_origin")):
        alarms = covered = events_n = scored = 0
        minutes = 0.0
        brier_terms, leads = [], []
        for record in sorted({r["record"] for r in origin_rows}):
            rr = [r for r in origin_rows if r["record"] == record]
            metrics = [x for x in overall if x["record"] == record and x["scenario"] == scenario][0]
            alarms += int(metrics["alarms"])
            covered += int(metrics["covered_events"])
            events_n += int(metrics["events"])
            minutes += int(metrics["origins"]) * float(
                json.loads((base / "records" / record / "metrics.json").read_text())["dt"]
            ) / 60.0
            # event leads and Brier are recomputed at record level for exact pooling.
            src_meta = json.loads((base / "records" / record / "metrics.json").read_text())
            src_raw = load_raw(ref / "data" / "records" / f"{record}.npz")
            events = [(float(x["onset"]), float(x["offset"])) for x in
                      read_rows(base / "records" / record / "reference_events.csv")]
            times = np.asarray([float(r["origin_time"]) for r in rr])
            risk = np.asarray([float(r[risk_key]) for r in rr])
            labels = onset_targets(times, events, 2.0)
            complete = times + 2.0 <= float(src_raw[-1, 0]) + 1e-9
            score = np.isfinite(risk) & complete
            brier_terms.extend(((risk[score] - labels[score]) ** 2).tolist())
            scored += int(score.sum())
            flags = np.asarray([int(r["alarm_beta_time_varying"] if risk_key ==
                                     "risk_beta_time_varying" else r["alarm_beta_fixed_at_origin"])
                                for r in rr], dtype=int)
            alarm_times = times[flags == 1]
            for j, (onset, _) in enumerate(events):
                if onset < float(src_meta["split"]["test_boundary"]) - 1e-9:
                    continue
                previous = events[j - 1][1] if j else -np.inf
                hits = [t for t in alarm_times
                        if previous < t < onset - 1e-9 and onset - t <= 2.0 + 1e-9]
                if hits:
                    leads.append(float(onset - min(hits)))
        total_rows.append({
            "scenario": scenario, "records": len({r["record"] for r in origin_rows}),
            "origins": len(origin_rows), "test_events": events_n,
            "covered_events": covered,
            "event_coverage": float(covered / events_n) if events_n else None,
            "alarms": alarms,
            "alarms_per_normal_minute": float(alarms / minutes) if minutes else None,
            "median_lead_within_2s": float(np.median(leads)) if leads else None,
            "brier_2s": float(np.mean(brier_terms)) if brier_terms else None,
            "scored_origins": scored,
        })
    write_rows(out / "beta_replay_summary.csv", total_rows)

    # Association of the fixed-parameter basin indicator with the algorithmic
    # future-event label; this does not imply a causal or physiological boundary.
    basin_pooled = []
    for positive in (0, 1):
        rr = [r for r in origin_rows if int(r["future_2s_event"]) == positive]
        if rr:
            inside = np.asarray([int(r["inside_fixed_parameter_basin"]) for r in rr])
            basin_pooled.append({
                "future_2s_event": positive, "origins": len(rr),
                "inside_fraction": float(inside.mean()),
                "median_r0": float(np.median([r["r0"] for r in rr])),
                "median_ru0": float(np.median([r["ru0_sqrt_minus_beta"] for r in rr])),
                "median_r0_over_ru0": float(np.median(
                    [r["r0"] / r["ru0_sqrt_minus_beta"] for r in rr])),
            })
    write_rows(out / "basin_summary.csv", basin_pooled)

    # Parameter distributions across record-specific test-origin inferences.
    tags = [r["record"] for r in per_record]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, column, title, color in zip(
        axes, ("beta0_median", "frequency0_hz_median", "sigma0_median"),
        ("Inferred beta at the origin", "Inferred frequency at the origin (Hz)",
         "Inferred sigma at the origin"), ("#3465a4", "#2a9d8f", "#e76f51")
    ):
        vals = []
        for row in per_record:
            low, high = PARAM_RANGES["beta" if column.startswith("beta") else
                                     "frequency_hz" if column.startswith("frequency") else "sigma"]
            # Re-read values from per-origin diagnostics to show within-record spread.
            part = [float(x[column.replace("_median", "")]) for x in origin_rows
                    if x["record"] == row["record"] and column.replace("_median", "") in x]
            if not part:
                # Use saved origin-level values where parameter columns are named by parameter.
                key = {"beta0_median": "beta0", "frequency0_hz_median": "frequency0_hz",
                       "sigma0_median": "sigma0"}[column]
                part = [float(x[key]) for x in origin_rows if x["record"] == row["record"]]
            vals.append(part)
        ax.boxplot(vals, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": color, "alpha": .35},
                   medianprops={"color": "#202a35", "linewidth": 1.5})
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xticks(np.arange(1, len(tags) + 1), tags, rotation=55, ha="right")
    fig.suptitle("Fitted full-model parameters across normal test origins")
    fig.tight_layout()
    fig.savefig(out / "parameter_distributions.png", dpi=180)
    fig.savefig(out / "parameter_distributions.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.asarray([r["ru0_sqrt_minus_beta"] for r in origin_rows])
    y = np.asarray([r["r0"] for r in origin_rows])
    c = np.asarray([int(r["future_2s_event"]) for r in origin_rows])
    for label, color, name in ((0, "#3478bf", "No reference onset in 2 s"),
                               (1, "#d64b40", "Reference onset in 2 s")):
        keep = c == label
        axes[0].scatter(x[keep], y[keep], s=8, alpha=.22, color=color, label=name,
                        rasterized=True)
    lim = max(float(np.nanquantile(x, .995)), float(np.nanquantile(y, .995)))
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="r0 = sqrt(-beta0)")
    axes[0].set(xlabel="Unstable radius sqrt(-beta0)", ylabel="Inferred initial radius r0",
                title="Initial state vs. frozen-parameter separatrix")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    rtv = np.asarray([r["risk_beta_time_varying"] for r in origin_rows])
    rfx = np.asarray([r["risk_beta_fixed_at_origin"] for r in origin_rows])
    axes[1].scatter(rtv, rfx, s=8, alpha=.2, color="#5c4d7d", rasterized=True)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(xlabel="P(event within 2 s), fitted beta(t)",
                ylabel="P(event within 2 s), beta fixed at beta(0)",
                title="Paired beta-only replay")
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "initial_state_and_beta_replay.png", dpi=180)
    fig.savefig(out / "initial_state_and_beta_replay.pdf")
    plt.close(fig)

    meta = {
        "seed": SEED, "records_processed": len(per_record),
        "protocol": "1T history, 5-s simulator, 128 paths, 2-s event probability, threshold >0.80 on two consecutive 0.05-s origins",
        "intervention": "Within each origin, hold beta at that origin's inferred beta(0) for all 5 s; preserve z0, frequency curve, sigma curve, decoder and paired Brownian increments.",
        "interpretation_limit": "Post-fit sensitivity only; the noise simulation is stochastic and beta is not shown to be uniquely identified or physiologically causal.",
        "outputs": ["parameter_by_record.csv", "origin_diagnostics.csv", "basin_by_event_label.csv",
                    "beta_replay_by_record.csv", "beta_replay_metrics_record.csv",
                    "beta_replay_summary.csv", "basin_summary.csv",
                    "parameter_distributions.png", "initial_state_and_beta_replay.png"]
    }
    (out / "protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print("BETA REPLAY SUMMARY", flush=True)
    for row in total_rows:
        print(json.dumps(row), flush=True)
    print("BASIN SUMMARY", flush=True)
    for row in basin_pooled:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
