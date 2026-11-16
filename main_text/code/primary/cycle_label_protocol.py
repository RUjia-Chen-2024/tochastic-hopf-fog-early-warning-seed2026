"""Published cycle-gap criterion plus explicitly recorded boundary conventions.

Parakkal Unni et al. (2020), doi:10.3389/fmedt.2020.581264,
Supplementary section 2: mean-peak scaling; abnormal inter-peak gap 1.5 times
the global mean cycle; no additional minimum separation between FoG events.
Exact peak detector and interval endpoint conventions were not published.
"""
from pathlib import Path
import sys
import numpy as np


def label_specification(cycle=1.13, reference_rule="paper_2020", merge_gap_cycles=0.):
    cycle = float(cycle)
    merge_gap_cycles = float(merge_gap_cycles)
    if not np.isfinite(cycle) or cycle <= 0:
        raise ValueError("The fixed global cycle must be finite and positive")
    if not np.isfinite(merge_gap_cycles) or merge_gap_cycles < 0:
        raise ValueError("merge-gap-cycles must be finite and non-negative")
    if reference_rule not in ("paper_2020", "legacy"):
        raise ValueError("Unknown reference rule: " + reference_rule)
    if reference_rule == "paper_2020" and merge_gap_cycles != 0:
        raise ValueError("paper_2020 has no inter-event gap merging; use merge-gap-cycles 0")
    onset_factor = 1.5 if reference_rule == "paper_2020" else 1.2
    return {
        "reference_rule": reference_rule,
        "paper_doi": "10.3389/fmedt.2020.581264",
        "paper_section": "Supplementary material, section 2 Labelling, pages 1-2",
        "cycle_seconds": cycle,
        "cycle_source": "Fixed published approximate global mean, not a per-record or previous-three-cycle estimate",
        "abnormal_gap_factor": 1.5,
        "abnormal_gap_seconds": 1.5 * cycle,
        "onset_factor": onset_factor,
        "onset_delay_seconds": onset_factor * cycle,
        "onset_convention": "Threshold crossing at previous peak + 1.5T" if reference_rule == "paper_2020" else "Legacy backdating to previous peak + 1.2T",
        "offset_convention": "Next detected peak; terminal event ends at recording end without asserting recovery",
        "merge_gap_cycles": merge_gap_cycles,
        "merge_gap_seconds": merge_gap_cycles * cycle,
        "extra_inter_event_merging": merge_gap_cycles > 0,
        "scaling": "Mean coarse-peak force difference, rather than maximum peak",
        "unpublished_implementation_choices": {
            "onset_and_offset_conventions": "Explicit operational choices; not fully specified by the article",
            "peak_distance_cycles": .45,
            "normalized_peak_height": .4,
            "normalized_peak_prominence": .08,
            "coarse_prominence_std_factor": .15,
            "initial_gap": "Do not infer a leading event before the first detected peak",
            "minimum_detected_peaks": 2,
            "threshold_equality": "A gap equal to 1.5T is not abnormal; strict excess with 1e-9 s numerical tolerance",
        },
        "reference_type": "Offline cycle-defined algorithmic reference; no independent video re-annotation performed",
    }


def reference_intervals(raw, reference_root, cycle=1.13, reference_rule="paper_2020"):
    """Keep the historical peak extractor, with a separate documented boundary rule.

    Legacy intervals reveal the exact same detected peaks: their right endpoints
    are unchanged and the new onset is shifted from 1.2T to the 1.5T threshold.
    This avoids copying or mutating the archived source project's detector.
    """
    specification = label_specification(cycle, reference_rule)
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError("Expected finite time,x,y samples")
    if np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError("Time must be increasing")
    sys.path.insert(0, str(Path(reference_root).resolve()))
    from fog.coordinates import detect_switch_by_step_cycle
    fs = 1 / np.median(np.diff(raw[:, 0]))
    # Explicitly override both the old 1.2T boundary and every event-gap merge.
    events = detect_switch_by_step_cycle(
        raw[:, 1], raw[:, 2], raw[:, 0], fs,
        global_mean_cycle=cycle,
        abnormal_factor=specification["abnormal_gap_factor"],
        onset_tolerance_ratio=specification["onset_factor"] - 1,
        merge_gap_ratio=0.,
    )
    # Apply tolerance only at the strict abnormal-gap decision, not by bridging
    # separate events. For the paper rule, interval length is excess over 1.5T.
    if reference_rule == "paper_2020":
        events = [(s, e) for s, e in events if e - s > 1e-9]
    return events
