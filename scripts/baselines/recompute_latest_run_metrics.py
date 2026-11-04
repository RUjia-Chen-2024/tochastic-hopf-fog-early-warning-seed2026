"""Recalculate publication metrics for the user's latest 5 s server run.

The script validates per-origin alignment, reproduces the server's threshold+
confirmation rule, and reports both the server's uncapped pre-onset coverage
and a secondary 1 s alarm-association metric. It does not retrain or simulate.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "analysis" / "latest_server_run_20261027"
BASE = ROOT / "analysis" / "baseline_comparison"
OUT = RUN / "recomputed_manuscript_metrics.json"


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def flags(times, risks, threshold, confirm=2, dt=0.05):
    streak = 0
    active = False
    last = None
    result = []
    for time, risk in zip(times, risks):
        if last is not None and time - last > 1.5 * dt + 1e-9:
            streak, active = 0, False
        above = math.isfinite(risk) and risk > threshold
        streak = streak + 1 if above else 0
        if not above:
            active = False
        issued = streak >= confirm and not active
        if issued:
            active = True
        result.append(issued)
        last = time
    return result


def event_metrics(alarms, event_rows, boundary):
    eligible = [e for e in event_rows if float(e["reference_onset"]) >= boundary - 1e-9]
    any_leads, one_sec_leads = [], []
    for index, event in enumerate(eligible):
        onset = float(event["reference_onset"])
        previous_end = event["previous_reference_end"]
        previous_end = float(previous_end) if previous_end not in (None, "") else -math.inf
        hits = [a for a in alarms if a >= boundary - 1e-9 and a > previous_end + 1e-9 and a < onset - 1e-9]
        if hits:
            any_leads.append(onset - min(hits))
        timely = [a for a in hits if 0 < onset - a <= 1.0 + 1e-9]
        if timely:
            one_sec_leads.append(onset - min(timely))
    return {
        "events": len(eligible),
        "covered_any_pre_onset": len(any_leads),
        "coverage_any_pre_onset": len(any_leads) / len(eligible) if eligible else None,
        "covered_within_1s": len(one_sec_leads),
        "coverage_within_1s": len(one_sec_leads) / len(eligible) if eligible else None,
        "median_first_warning_lead_any_s": float(np.median(any_leads)) if any_leads else None,
        "median_lead_within_1s_s": float(np.median(one_sec_leads)) if one_sec_leads else None,
        "leads_any_s": any_leads,
    }


def alarm_association(alarms, event_rows, boundary):
    onsets = [float(e["reference_onset"]) for e in event_rows if float(e["reference_onset"]) >= boundary - 1e-9]
    leads = []
    for t in alarms:
        future = [s - t for s in onsets if s > t + 1e-9]
        if future:
            leads.append(min(future))
    within = sum(0 < lead <= 1.0 + 1e-9 for lead in leads)
    return {"alarms_followed_by_onset": len(leads), "alarms_followed_within_1s": within,
            "alarm_match_fraction_within_1s": within / len(alarms) if alarms else None,
            "tail_alarms": len(alarms) - len(leads)}


def main():
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cfg = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    event_rows = rows(RUN / "events.csv")
    by_record_events = defaultdict(list)
    for e in event_rows:
        by_record_events[e["tag"]].append(e)

    baseline_origins = rows(BASE / "origins.csv")
    baseline_by = defaultdict(dict)
    for r in baseline_origins:
        if r["method"] not in {"age_survival", "age_logistic_ungated", "clock_gate"}:
            continue
        baseline_by[r["record"]].setdefault(r["method"], {}).setdefault("times", []).append(float(r["origin_time"]))
        baseline_by[r["record"]].setdefault(r["method"], {}).setdefault("risk", []).append(float(r["risk"]))
        baseline_by[r["record"]].setdefault(r["method"], {}).setdefault("labels", []).append(int(r["label"]))
        baseline_by[r["record"]].setdefault(r["method"], {}).setdefault("scored", []).append(int(r["scored"]))

    model_rows = []
    all_risk = []
    all_y = []
    all_score_mask = []
    all_force_abs = []
    all_force_sq = []
    all_persistence_abs = []
    all_persistence_sq = []
    all_interval_hit = []
    all_crossing_abs = []
    all_crossing_sq = []
    all_crossing_persistence_abs = []
    all_crossing_persistence_sq = []
    improved_records = []
    record_stats = []
    methods = {
        "DeepONet-Hopf (latest 5 s endpoint)": {"by_record": {}, "risk_scores": [], "labels": [], "score_mask": []},
        "Inter-peak survival": {"by_record": {}, "risk_scores": [], "labels": [], "score_mask": []},
        "Peak-age logistic": {"by_record": {}, "risk_scores": [], "labels": [], "score_mask": []},
        "Pure peak clock": {"by_record": {}, "risk_scores": [], "labels": [], "score_mask": []},
    }
    name_map = {
        "age_survival": "Inter-peak survival",
        "age_logistic_ungated": "Peak-age logistic",
        "clock_gate": "Pure peak clock",
    }

    per_record = rows(RUN / "per_record.csv")
    for pr in per_record:
        if "tag" not in pr:
            pr["tag"] = json.loads((RUN / "records" / pr["record"] / "metrics.json").read_text())["configuration"]["tag"]
        record = pr["tag"].replace(" ", "_")
        folder = RUN / "records" / record
        risk_rows = rows(folder / "risk.csv")
        times = [float(r["origin_time"]) for r in risk_rows]
        risks = [float(r["transition_probability"]) for r in risk_rows]
        trigger_flags = [r["experimental_trigger"] in ("1", "True", "true") for r in risk_rows]
        reproduced = flags(times, risks, float(protocol["alarm_threshold"]), int(protocol["confirm"]))
        assert reproduced == trigger_flags, f"Latest-run trigger mismatch: {record}"
        alarms = [t for t, hit in zip(times, trigger_flags) if hit]
        validation = rows(folder / "event_warning_validation.csv")
        boundary = float(times[0])
        em = event_metrics(alarms, validation, boundary)
        saved_warned = sum(r["warned_before"] == "True" for r in validation)
        assert saved_warned == em["covered_any_pre_onset"], f"Warning coverage mismatch: {record}"
        model_rows.append({"record": record, "participant": pr["participant"], "alarms": len(alarms), **em, **alarm_association(alarms, by_record_events[pr["tag"]], boundary)})
        model_event_metric = em

        # Verify common origin times and event labels against cached baseline data.
        base_ref = baseline_by[record]["age_survival"]
        assert len(times) == len(base_ref["times"])
        assert np.allclose(times, base_ref["times"], atol=1e-9, rtol=0)
        labels = [int(float(r["future_onset_label"])) for r in risk_rows]
        assert labels == base_ref["labels"], f"Target-label mismatch: {record}"
        scored = base_ref["scored"]
        assert len(scored) == len(risk_rows)
        methods["DeepONet-Hopf (latest 5 s endpoint)"]["risk_scores"].extend(r for r, s in zip(risks, scored) if s)
        methods["DeepONet-Hopf (latest 5 s endpoint)"]["labels"].extend(y for y, s in zip(labels, scored) if s)
        methods["DeepONet-Hopf (latest 5 s endpoint)"]["score_mask"].extend([1] * sum(scored))
        methods["DeepONet-Hopf (latest 5 s endpoint)"]["by_record"][record] = {"times": times, "risks": risks, "alarms": alarms, "scored": scored, "event_metrics": model_event_metric, "alarm_association": alarm_association(alarms, by_record_events[pr["tag"]], boundary)}

        # Recompute the stronger baselines at the same >0.8 / two-origin rule.
        for method_key, display in name_map.items():
            bm = baseline_by[record][method_key]
            assert np.allclose(times, bm["times"], atol=1e-9, rtol=0)
            assert labels == bm["labels"] and scored == bm["scored"]
            bf = flags(bm["times"], bm["risk"], float(protocol["alarm_threshold"]), int(protocol["confirm"]))
            ba = [t for t, hit in zip(bm["times"], bf) if hit]
            bem = event_metrics(ba, validation, boundary)
            methods[display]["by_record"][record] = {"times": bm["times"], "risks": bm["risk"], "alarms": ba, "scored": scored, "event_metrics": bem, "alarm_association": alarm_association(ba, by_record_events[pr["tag"]], boundary)}
            methods[display]["risk_scores"].extend(r for r, s in zip(bm["risk"], scored) if s)
            methods[display]["labels"].extend(y for y, s in zip(labels, scored) if s)
            methods[display]["score_mask"].extend([1] * sum(scored))

        # Forecast error distribution from the actual saved five-second ensemble.
        with np.load(folder / "predictions.npz") as z:
            truth = z["truth"].astype(float)
            mean = z["mean"].astype(float)
            q05 = z["q05"].astype(float)
            q95 = z["q95"].astype(float)
            origin = z["origin_time"].astype(float)
            horizon = z["horizon_seconds"].astype(float)
            valid = np.isfinite(truth)
            err = mean - truth
            all_force_abs.append(np.abs(err[valid]))
            all_force_sq.append(np.square(err[valid]))
            all_interval_hit.append(((truth >= q05) & (truth <= q95))[valid])
            event_times = np.asarray([float(e["reference_onset"]) for e in by_record_events[pr["tag"]]], dtype=float)
            target_t = origin[:, None] + horizon[None, :]
            crossing = np.zeros(target_t.shape, dtype=bool)
            for s in event_times:
                crossing |= (origin[:, None] < s) & (s <= target_t + 1e-9)
            crossing3 = np.broadcast_to(crossing[:, :, None], truth.shape) & valid
            raw_path = RUN / "raw_records" / f"{record}.npz"
            with np.load(raw_path) as raw:
                raw_x, raw_y = raw["x"].astype(float), raw["y"].astype(float)
            context = rows(folder / "network_history_context.csv")
            end_indices = [int(float(r["raw_end_index"])) for r in context]
            assert len(end_indices) == len(origin)
            current = np.asarray([
                [raw_x[max(0, end - 4):end + 1].mean(), raw_y[max(0, end - 4):end + 1].mean()]
                for end in end_indices
            ])
            persistence = np.broadcast_to(current[:, None, :], truth.shape)
            perr = persistence - truth
            all_persistence_abs.append(np.abs(perr[valid]))
            all_persistence_sq.append(np.square(perr[valid]))
            all_crossing_abs.append(np.abs(err[crossing3]))
            all_crossing_sq.append(np.square(err[crossing3]))
            all_crossing_persistence_abs.append(np.abs(perr[crossing3]))
            all_crossing_persistence_sq.append(np.square(perr[crossing3]))
            rec_mae = float(np.abs(err[valid]).mean())
            persistence_mae = float(np.abs(perr[valid]).mean())
            improved_records.append(rec_mae < persistence_mae)
            record_stats.append({
                "record": record, "participant": pr["participant"],
                "valid_force_scalars": int(valid.sum()),
                "mae_force": rec_mae,
                "persistence_mae_force": persistence_mae,
                "rmse_force": float(np.sqrt(np.square(err[valid]).mean())),
                "onset_crossing_scalars": int(crossing3.sum()),
                "onset_crossing_mae": float(np.abs(err[crossing3]).mean()) if crossing3.any() else None,
                "onset_crossing_persistence_mae": float(np.abs(perr[crossing3]).mean()) if crossing3.any() else None,
                "interval_coverage_90": float(((truth >= q05) & (truth <= q95))[valid].mean()),
            })

    total_origins = sum(int(r["normal_origins"]) for r in per_record)
    normal_minutes = total_origins * 0.05 / 60.0
    summary = {
        "primary_run": str(RUN),
        "protocol": {
            "architecture": protocol["architecture"],
            "observation_aggregation": protocol["observation_aggregation"],
            "simulation_horizon_seconds": protocol["simulation_horizon_seconds"],
            "event_probability_horizon_seconds": protocol["event_horizon"],
            "network_history_seconds": protocol["network_history"]["duration_seconds"],
            "paths": protocol["draws"], "threshold_strictly_greater_than": protocol["alarm_threshold"],
            "consecutive_origins": protocol["confirm"], "reference_rule": protocol["reference_rule"],
            "merge_gap_cycles": protocol["merge_gap_cycles"],
            "test_split": "last 35%; model weights fit within first 65% (52% fit/13% validation)",
        },
        "records": len(per_record), "participants": len(set(r["participant"] for r in per_record)),
        "reference_events": len(event_rows), "normal_origins": total_origins,
        "normal_exposure_minutes": normal_minutes, "scored_probability_origins": sum(methods[next(iter(methods))]["score_mask"]),
        "warning_evaluation_note": "Code-reported event coverage has no maximum lead: any alarm after the previous reference event ends and before the next onset counts. The within-1-s association is reported separately.",
        "methods": {}, "per_record": model_rows,
        "force_forecast": {
            "scalar_targets": int(sum(len(a) for a in all_force_abs)),
            "pooled_mae": float(np.concatenate(all_force_abs).mean()),
            "pooled_rmse": float(np.sqrt(np.concatenate(all_force_sq).mean())),
            "persistence_pooled_mae": float(np.concatenate(all_persistence_abs).mean()),
            "persistence_pooled_rmse": float(np.sqrt(np.concatenate(all_persistence_sq).mean())),
            "records_lower_mae_than_persistence": int(sum(improved_records)),
            "process_interval_coverage_90": float(np.concatenate(all_interval_hit).mean()),
            "onset_crossing_scalar_targets": int(sum(r["onset_crossing_scalars"] for r in record_stats)),
            "onset_crossing_mae": float(np.concatenate(all_crossing_abs).mean()),
            "onset_crossing_rmse": float(np.sqrt(np.concatenate(all_crossing_sq).mean())),
            "onset_crossing_persistence_mae": float(np.concatenate(all_crossing_persistence_abs).mean()),
            "onset_crossing_persistence_rmse": float(np.sqrt(np.concatenate(all_crossing_persistence_sq).mean())),
            "per_record": record_stats,
        },
    }

    for name, data in methods.items():
        covered_any = covered_1s = alarm_count = scored_count = alarm_followed = alarm_within_1s = tail_alarms = 0
        leads_any, leads_1s = [], []
        for r in per_record:
            record = r["tag"].replace(" ", "_")
            m = data["by_record"][record]
            v = rows(RUN / "records" / record / "event_warning_validation.csv")
            em = event_metrics(m["alarms"], v, min(m["times"]))
            covered_any += em["covered_any_pre_onset"]
            covered_1s += em["covered_within_1s"]
            leads_any.extend(em["leads_any_s"])
            alarm_count += len(m["alarms"])
            aa = m["alarm_association"]
            alarm_followed += aa["alarms_followed_by_onset"]
            alarm_within_1s += aa["alarms_followed_within_1s"]
            tail_alarms += aa["tail_alarms"]
            scored_count += sum(m["scored"])
        y = np.asarray(data["labels"], dtype=float)
        risk = np.asarray(data["risk_scores"], dtype=float)
        brier = float(np.mean(np.square(risk - y)))
        pos = y == 1
        neg = y == 0
        calibration_bins = []
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            keep = (risk >= lo) & (risk < hi if b < 9 else risk <= hi)
            if np.any(keep):
                calibration_bins.append({"bin": b, "n": int(keep.sum()), "mean_probability": float(risk[keep].mean()), "event_fraction": float(y[keep].mean())})
        summary["methods"][name] = {
            "covered_any_pre_onset": covered_any, "coverage_any_pre_onset": covered_any / len(event_rows),
            "covered_within_1s": covered_1s, "coverage_within_1s": covered_1s / len(event_rows),
            "alarms": alarm_count, "alarms_per_sampled_normal_minute": alarm_count / normal_minutes,
            "alarms_followed_within_1s": alarm_within_1s,
            "alarm_match_fraction_within_1s": alarm_within_1s / alarm_count if alarm_count else None,
            "alarms_followed_by_later_onset": alarm_followed, "tail_alarms": tail_alarms,
            "median_first_warning_lead_any_s": float(np.median(leads_any)) if leads_any else None,
            "brier_1s_event_probability": brier if name != "Pure peak clock" else None,
            "positive_origins": int(pos.sum()), "negative_origins": int(neg.sum()),
            "positive_brier_component": float(np.square(risk[pos] - y[pos]).mean()) if np.any(pos) else None,
            "negative_brier_component": float(np.square(risk[neg] - y[neg]).mean()) if np.any(neg) else None,
            "reliability_bins": calibration_bins,
            "scored_origins": scored_count,
        }

    # Participant-cluster bootstrap for descriptive uncertainty at the frozen
    # checkpoint and threshold. Each draw retains every recording from a person.
    rng = np.random.default_rng(20261031)
    participants = sorted({r["participant"] for r in per_record})
    record_participant = {r["tag"].replace(" ", "_"): r["participant"] for r in per_record}
    draws = rng.integers(0, len(participants), size=(5000, len(participants)))
    bootstrap = {name: {metric: [] for metric in ("coverage_any", "coverage_1s", "alarm_rate", "brier")} for name in methods}
    for draw in draws:
        sampled = [participants[i] for i in draw]
        for name, data in methods.items():
            ev, cov, cov1, alarms_n, normal_n, brier_sum, nscore = 0, 0, 0, 0, 0, 0.0, 0
            for person in sampled:
                for record, person0 in record_participant.items():
                    if person0 != person:
                        continue
                    m = data["by_record"][record]
                    em = m["event_metrics"]
                    ev += em["events"]
                    cov += em["covered_any_pre_onset"]
                    cov1 += em["covered_within_1s"]
                    alarms_n += len(m["alarms"])
                    normal_n += len(m["times"])
                    ys = np.asarray(baseline_by[record]["age_survival"]["labels"], dtype=float)
                    sc = np.asarray(m["scored"], dtype=bool)
                    rr = np.asarray(m["risks"], dtype=float)
                    brier_sum += float(np.square(rr[sc] - ys[sc]).sum())
                    nscore += int(sc.sum())
            bootstrap[name]["coverage_any"].append(cov / ev if ev else np.nan)
            bootstrap[name]["coverage_1s"].append(cov1 / ev if ev else np.nan)
            bootstrap[name]["alarm_rate"].append(alarms_n / (normal_n * 0.05 / 60.0) if normal_n else np.nan)
            bootstrap[name]["brier"].append(brier_sum / nscore if nscore else np.nan)

    def ci(a):
        return [float(x) for x in np.nanpercentile(np.asarray(a), [2.5, 97.5])]
    summary["cluster_bootstrap"] = {"seed": 20261031, "resamples": 5000, "clusters": 7, "scope": "frozen risk streams and fixed threshold; participant-level resampling", "methods": {}, "paired_differences_vs_survival": {}}
    for name in methods:
        summary["cluster_bootstrap"]["methods"][name] = {metric: ci(vals) for metric, vals in bootstrap[name].items()}
    main_name = "DeepONet-Hopf (latest 5 s endpoint)"
    surv_name = "Inter-peak survival"
    for metric in ("coverage_any", "coverage_1s", "alarm_rate", "brier"):
        diff = np.asarray(bootstrap[main_name][metric]) - np.asarray(bootstrap[surv_name][metric])
        field = {"coverage_any": "coverage_any_pre_onset", "coverage_1s": "coverage_within_1s", "alarm_rate": "alarms_per_sampled_normal_minute", "brier": "brier_1s_event_probability"}[metric]
        observed = summary["methods"][main_name][field] - summary["methods"][surv_name][field]
        summary["cluster_bootstrap"]["paired_differences_vs_survival"][metric] = {"estimate": observed, "observed_difference": observed, "bootstrap_mean": float(np.mean(diff)), "ci95": ci(diff)}

    (RUN / "latest_run_metrics.csv").write_text(
        "method,covered_any_pre_onset,coverage_any_pre_onset,covered_within_1s,coverage_within_1s,alarms,alarms_per_sampled_normal_minute,median_first_warning_lead_any_s,brier_1s_event_probability,scored_origins\n"
        + "".join(
            ",".join([
                json.dumps(name), str(x["covered_any_pre_onset"]), f'{x["coverage_any_pre_onset"]:.9f}',
                str(x["covered_within_1s"]), f'{x["coverage_within_1s"]:.9f}', str(x["alarms"]),
                f'{x["alarms_per_sampled_normal_minute"]:.6f}', f'{x["median_first_warning_lead_any_s"]:.6f}' if x["median_first_warning_lead_any_s"] is not None else "",
                f'{x["brier_1s_event_probability"]:.9f}' if x["brier_1s_event_probability"] is not None else "", str(x["scored_origins"])
            ]) + "\n" for name, x in summary["methods"].items()
        ), encoding="utf-8")
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "methods": summary["methods"], "force_forecast": {k:v for k,v in summary["force_forecast"].items() if k != "per_record"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
