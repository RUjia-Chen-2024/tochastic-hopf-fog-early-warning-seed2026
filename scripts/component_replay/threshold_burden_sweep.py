"""Post-hoc 0.80-rule threshold sweep from saved 2-s probabilities.

Uses the same one-alarm-per-above-threshold-run, two-consecutive-origin rule as
the main evaluation. This is descriptive operating-point analysis; it must not
be used to select a threshold and then claim held-out test performance.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def threshold_warnings(times, risks, threshold, confirm=2, dt=0.05):
    """Match generalized_hopf_deeponet.threshold_warnings exactly."""
    streak = 0
    active = False
    last_time = None
    flags = []
    for time, risk in zip(times, risks):
        if last_time is not None and time - last_time > 1.5 * dt:
            streak = 0
            active = False
        above = risk > threshold
        streak = streak + 1 if above else 0
        if not above:
            active = False
        issued = streak >= confirm and not active
        if issued:
            active = True
        flags.append(int(issued))
        last_time = time
    return np.asarray(flags, dtype=int)


def main():
    apath = Path(__file__).resolve().parents[1] / "evaluation"
    input_rows = []
    for path in sorted(apath.glob("*_origins.csv")):
        input_rows.extend(read_csv(path))
    per_record = defaultdict(list)
    for r in input_rows:
        per_record[(r["record"], r["method"])].append(r)

    event_rows = read_csv(apath / "events.csv")
    event_groups = defaultdict(list)
    for r in event_rows:
        event_groups[(r["record"], r["method"])].append(r)

    methods = list(dict.fromkeys(r["method"] for r in input_rows))
    thresholds = sorted(set(
        [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.01)]
        + [127.0 / 128.0, 1.0]
    ))
    rows = []
    for method in methods:
        data = [r for r in input_rows if r["method"] == method]
        total_origins = len(data)
        dt = float(np.median([
            float(b["origin_time"]) - float(a["origin_time"])
            for group in per_record.values() if group and group[0]["method"] == method
            for a, b in zip(group[:-1], group[1:])
        ]))
        normal_minutes = total_origins * dt / 60.0
        test_events = sum(len(event_groups[(record, method)])
                          for record, m in per_record if m == method)

        for threshold in thresholds:
            alarms = []
            covered = 0
            for (record, m), group in per_record.items():
                if m != method:
                    continue
                group = sorted(group, key=lambda r: float(r["origin_time"]))
                times = [float(r["origin_time"]) for r in group]
                risks = [float(r["risk"]) for r in group]
                flags = threshold_warnings(times, risks, threshold, confirm=2, dt=dt)
                for r, flag in zip(group, flags):
                    if flag:
                        alarms.append((record, float(r["origin_time"]), int(r["complete_2s"])))
                record_events = event_groups[(record, method)]
                for e in record_events:
                    onset = float(e["reference_onset"])
                    previous_end = (float(e["previous_reference_end"])
                                    if e["previous_reference_end"] else -np.inf)
                    if any(a_record == record and previous_end < t < onset and onset - t <= 2.0 + 1e-9
                           for a_record, t, _ in alarms if a_record == record):
                        covered += 1

            matched = unmatched = tail = 0
            for record, time, complete in alarms:
                if not complete:
                    tail += 1
                    continue
                evs = event_groups[(record, method)]
                future = [float(e["reference_onset"]) for e in evs if float(e["reference_onset"]) > time + 1e-9]
                previous = max([float(e["reference_end"]) for e in evs if float(e["reference_onset"]) <= time + 1e-9], default=-np.inf)
                onset = min(future) if future else None
                if onset is not None and previous < time and onset - time <= 2.0 + 1e-9:
                    matched += 1
                else:
                    unmatched += 1
            rows.append(dict(method=method, threshold=threshold, events=test_events,
                             covered_events=covered, event_coverage=covered / test_events if test_events else None,
                             alarms=len(alarms), alarms_per_normal_minute=len(alarms) / normal_minutes if normal_minutes else None,
                             matched_alarms=matched, unmatched_alarms=unmatched, tail_censored_alarms=tail,
                             normal_test_minutes=normal_minutes))

    # Verify that threshold=.80 reproduces the primary report's counts exactly.
    primary = {r["method"]: r for r in read_csv(apath / "summary.csv")}
    for r in rows:
        if abs(r["threshold"] - 0.80) < 1e-9:
            p = primary[r["method"]]
            assert r["covered_events"] == int(p["covered_within_2s"]), (r, p)
            assert r["alarms"] == int(p["alarms"]), (r, p)

    write_csv(apath / "threshold_sweep.csv", rows)
    mlp = primary["Direct risk MLP"]
    target = float(mlp["alarms_per_normal_minute"])
    burden_brackets = []
    for method in methods:
        candidates = [r for r in rows if r["method"] == method]
        lower = max((r for r in candidates if float(r["alarms_per_normal_minute"]) <= target),
                    key=lambda r: float(r["alarms_per_normal_minute"]), default=None)
        upper = min((r for r in candidates if float(r["alarms_per_normal_minute"]) >= target),
                    key=lambda r: float(r["alarms_per_normal_minute"]), default=None)
        row = dict(reference_burden_method="Direct risk MLP at threshold 0.80",
                   target_alarms_per_minute=target, method=method)
        for side, r in (("below", lower), ("above", upper)):
            row.update({f"{side}_{k}": v for k, v in {
                "threshold": r["threshold"] if r else None,
                "alarms_per_normal_minute": r["alarms_per_normal_minute"] if r else None,
                "covered_events": r["covered_events"] if r else None,
                "events": r["events"] if r else None,
                "event_coverage": r["event_coverage"] if r else None,
                "alarms": r["alarms"] if r else None,
            }.items()})
        burden_brackets.append(row)
    write_csv(apath / "burden_match_brackets.csv", burden_brackets)
    print(json.dumps(burden_brackets, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["#2878B5", "#D17C2F", "#299A82", "#9157A4"]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for i, method in enumerate(methods):
            group = [r for r in rows if r["method"] == method]
            ax.plot([r["alarms_per_normal_minute"] for r in group],
                    [r["event_coverage"] for r in group], marker="o", markersize=2.5,
                    linewidth=1.6, color=colors[i], label=method)
            p = primary[method]
            ax.scatter([float(p["alarms_per_normal_minute"])], [float(p["event_coverage"])],
                       marker="*", s=130, edgecolors="black", linewidths=.4, color=colors[i], zorder=5)
        ax.set(xlabel="Alarm burden (alarms per normal test minute)", ylabel="Event coverage within 2 s",
               title="Coverage–burden curves (threshold sweep; star = threshold 0.80)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.22)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(apath / "coverage_burden_curve.png", dpi=190)
        fig.savefig(apath / "coverage_burden_curve.pdf")
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; CSV sweep still generated")


if __name__ == "__main__":
    main()
