from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import RobustScaler,StandardScaler
from .constants import *

ROOT=Path(__file__).resolve().parents[1]
def load_state_only_data_for_tag(tag):
    path=ROOT/'data/records'/(str(tag).replace(' ','_')+'.npz')
    with np.load(path) as z:
        t,x,y=z['t'],z['x'],z['y']
    if not np.all(np.diff(t)>0):raise ValueError('Time must be strictly increasing')
    return dict(t_orig=t,x_orig=x,y_orig=y,state_file=str(path))

def positive_median(x, fallback=1.0):
    a = np.asarray(x, float)
    a = a[np.isfinite(a) & (a > EPS)]
    if len(a) == 0:
        return float(fallback)
    return max(float(np.median(a)), EPS)

def causal_robust_amplitude(d, fs):
    """Trailing 0.5*(Q90-Q10), then a trailing mean."""
    d = np.asarray(d, float)
    n = max(3, int(round(AMP_WIN_SEC * fs)))
    s = pd.Series(d)
    q90 = s.rolling(n, min_periods=1).quantile(0.90)
    q10 = s.rolling(n, min_periods=1).quantile(0.10)
    amp = 0.5 * (q90 - q10)
    ns = max(1, int(round(SMOOTH_SEC * fs)))
    amp = amp.rolling(ns, min_periods=1).mean().to_numpy(float)
    return np.clip(np.nan_to_num(amp, nan=0.0), 0.0, None)

def causal_local_reference(signal, fs, fallback=1.0):
    """
    Strictly-past local median.
    The current 0.8-s feature window is excluded by shifting the baseline.
    """
    signal = np.asarray(signal, float)
    shift_n = max(1, int(round(LOCAL_EXCLUDE_SEC * fs)))
    win_n = max(3, int(round(LOCAL_BASELINE_SEC * fs)))
    minp = max(3, int(round(0.50 * win_n)))
    ref = (
        pd.Series(signal)
        .shift(shift_n)
        .rolling(win_n, min_periods=minp)
        .median()
        .to_numpy(float)
    )
    bad = (~np.isfinite(ref)) | (ref <= EPS)
    ref[bad] = max(float(fallback), EPS)
    return ref

def compute_se2_me2(x, y, fs, calibration_stop):
    """
    Build the same causal retention coordinates used in the transition-aware
    alarm study, with calibration constants fitted only on [:calibration_stop].

    SE2 ~ normalized robust oscillation amplitude.
    ME2 ~ normalized derivative-energy retention.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) != len(y):
        raise ValueError("x/y length mismatch")

    n = len(x)
    calibration_stop = int(np.clip(calibration_stop, 10, n))
    d = x - y

    # ---------------- SE2 ----------------
    amp = causal_robust_amplitude(d, fs)
    amp_global = positive_median(amp[:calibration_stop], fallback=1.0)
    amp_local = causal_local_reference(amp, fs, fallback=amp_global)

    amp_global_ratio = np.clip(
        amp / max(amp_global, EPS), 0.0, RETENTION_CLIP_MAX
    )
    amp_local_ratio = np.clip(
        amp / np.maximum(amp_local, EPS), 0.0, RETENTION_CLIP_MAX
    )
    se2 = np.sqrt(amp_global_ratio * amp_local_ratio)

    # ---------------- ME2 ----------------
    v = np.zeros(n, float)
    if n > 1:
        v[1:] = np.diff(d) * float(fs)
    v2 = np.square(v)

    cal_v2 = v2[:calibration_stop]
    finite = cal_v2[np.isfinite(cal_v2)]
    if len(finite) == 0:
        cap = 1.0
    else:
        cap = float(np.quantile(finite, ME_WINSOR_Q))
        if not np.isfinite(cap) or cap <= EPS:
            cap = max(float(np.nanmax(finite)), 1.0)

    v2_cap = np.minimum(np.nan_to_num(v2, nan=0.0), cap)
    n_energy = max(3, int(round(ENERGY_WIN_SEC * fs)))
    energy = (
        pd.Series(v2_cap)
        .rolling(n_energy, min_periods=1)
        .mean()
    )
    ns = max(1, int(round(SMOOTH_SEC * fs)))
    energy = (
        energy
        .rolling(ns, min_periods=1)
        .mean()
        .to_numpy(float)
    )
    energy = np.clip(np.nan_to_num(energy, nan=0.0), 0.0, None)

    energy_global = positive_median(
        energy[:calibration_stop], fallback=1.0
    )
    energy_local = causal_local_reference(
        energy, fs, fallback=energy_global
    )

    energy_global_ratio = np.clip(
        energy / max(energy_global, EPS), 0.0, RETENTION_CLIP_MAX
    )
    energy_local_ratio = np.clip(
        energy / np.maximum(energy_local, EPS),
        0.0,
        RETENTION_CLIP_MAX,
    )
    me2 = np.sqrt(energy_global_ratio * energy_local_ratio)

    return {
        "SE2": np.clip(np.nan_to_num(se2, nan=0.0), 0.0, RETENTION_CLIP_MAX),
        "ME2": np.clip(np.nan_to_num(me2, nan=0.0), 0.0, RETENTION_CLIP_MAX),
        "amp": amp,
        "energy": energy,
        "amp_global_ref": float(amp_global),
        "energy_global_ref": float(energy_global),
        "energy_winsor_cap": float(cap),
    }

def causal_log_slope(v, fs, lookback_sec=LOG_SLOPE_LOOKBACK_SEC):
    """
    Backward log slope:
        [log v(t) - log v(t-tau)] / tau
    """
    v = np.asarray(v, float)
    k = max(1, int(round(float(lookback_sec) * fs)))
    z = np.log(np.maximum(v, EPS))
    out = np.zeros(len(v), float)

    if len(v) <= k:
        return out

    out[k:] = (z[k:] - z[:-k]) / (k / float(fs))
    # Startup samples are outside scored regions; keep the first valid value
    # rather than reading any future scored sample.
    out[:k] = out[k]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def detect_switch_by_step_cycle(
    x_raw,
    y_raw,
    t_seq,
    fs,
    global_mean_cycle=GT_REF_CYCLE_SEC,
    abnormal_factor=GT_ABNORMAL_FACTOR,
    onset_tolerance_ratio=GT_ONSET_TOLERANCE_RATIO,
    peak_height_ratio=0.40,
    peak_prominence_ratio=0.08,
    min_peak_distance_ratio=0.45,
    merge_gap_ratio=0.25,
):
    """
    Offline algorithmic cycle reference.  This is used ONLY for evaluation.
    """
    x_raw = np.asarray(x_raw, float)
    y_raw = np.asarray(y_raw, float)
    t_seq = np.asarray(t_seq, float)

    T_ref = float(global_mean_cycle)
    stepping_force = np.abs(x_raw - y_raw)

    min_distance = max(
        int(round(min_peak_distance_ratio * T_ref * fs)), 2
    )
    coarse_prominence = max(
        0.15 * float(np.std(stepping_force)), 1e-8
    )
    coarse_peaks, _ = find_peaks(
        stepping_force,
        distance=min_distance,
        prominence=coarse_prominence,
    )

    if len(coarse_peaks) < 2:
        return []

    mean_peak = max(
        float(np.mean(stepping_force[coarse_peaks])),
        1e-8,
    )
    stepping_force_norm = stepping_force / mean_peak

    peaks, _ = find_peaks(
        stepping_force_norm,
        distance=min_distance,
        height=peak_height_ratio,
        prominence=peak_prominence_ratio,
    )
    if len(peaks) < 2:
        return []

    peak_times = t_seq[peaks]
    intervals = np.diff(peak_times)

    abnormal_threshold = abnormal_factor * T_ref
    onset_delay = (1.0 + onset_tolerance_ratio) * T_ref

    fog_intervals = []
    for i, interval in enumerate(intervals):
        if interval > abnormal_threshold:
            s = float(peak_times[i] + onset_delay)
            e = float(peak_times[i + 1])
            s = min(max(s, float(peak_times[i])), e)
            if e > s:
                fog_intervals.append((s, e))

    # terminal interval
    last_peak_t = float(peak_times[-1])
    if float(t_seq[-1] - last_peak_t) > abnormal_threshold:
        s = last_peak_t + onset_delay
        if s < float(t_seq[-1]):
            fog_intervals.append((float(s), float(t_seq[-1])))

    # merge very close reference intervals
    if len(fog_intervals) > 1:
        merge_gap = merge_gap_ratio * T_ref
        merged = [list(fog_intervals[0])]
        for s, e in fog_intervals[1:]:
            if s - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        fog_intervals = [(float(s), float(e)) for s, e in merged]

    return fog_intervals

def build_state_only_features(indicator_dict, fs):
    se2 = np.asarray(indicator_dict["SE2"], float)
    me2 = np.asarray(indicator_dict["ME2"], float)

    dlog_se2 = causal_log_slope(se2, fs)
    dlog_me2 = causal_log_slope(me2, fs)

    features = np.column_stack([
        se2,
        dlog_se2,
        me2,
        dlog_me2,
    ])
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    targets = np.column_stack([se2, me2])

    return {
        "features_raw": features,
        "targets_raw": targets,
        "SE2": se2,
        "ME2": me2,
        "dlog_SE2": dlog_se2,
        "dlog_ME2": dlog_me2,
    }

def _participant_from_tag(tag):
    m = re.search(
        r"Pt\s*[_ ]?(\d+)",
        str(tag),
        flags=re.IGNORECASE,
    )
    if m:
        return f"Pt {m.group(1)}"

    # Fallback: group by prefix before first underscore.
    return str(tag).split("_")[0]

def prepare_record(tag):
    data = load_state_only_data_for_tag(tag)
    t = np.asarray(data["t_orig"], float)
    x = np.asarray(data["x_orig"], float)
    y = np.asarray(data["y_orig"], float)

    if len(t) < 500:
        raise ValueError("recording too short")

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("bad time grid")

    fs = 1.0 / dt
    max_time = MAX_DT * dt

    split_idx = int(TRAIN_RATIO * len(t))
    sel_idx = int((1.0 - VAL_RATIO_WITHIN_TRAIN) * split_idx)

    if sel_idx < HISTORY_WINDOW + MAX_DT + 10:
        raise ValueError("selection-train too short")

    gt_all = detect_switch_by_step_cycle(x, y, t, fs)

    ind = compute_se2_me2(
        x, y, fs,
        calibration_stop=sel_idx,
    )
    built = build_state_only_features(ind, fs)

    feature_scaler = RobustScaler().fit(
        built["features_raw"][:sel_idx]
    )
    feats_norm = feature_scaler.transform(
        built["features_raw"]
    )

    target_scaler = StandardScaler().fit(
        built["targets_raw"][:sel_idx]
    )
    targets_norm = target_scaler.transform(
        built["targets_raw"]
    )

    se2 = built["SE2"]
    me2 = built["ME2"]
    dlog_se2 = built["dlog_SE2"]
    dlog_me2 = built["dlog_ME2"]

    se_th = float(np.quantile(se2[:split_idx], SE2_ENTRY_Q))
    me_th = float(np.quantile(me2[:split_idx], ME2_ENTRY_Q))
    se_slope_th = float(np.quantile(dlog_se2[:split_idx], TREND_Q))
    me_slope_th = float(np.quantile(dlog_me2[:split_idx], TREND_Q))

    test_start_idx = split_idx + HISTORY_WINDOW
    if test_start_idx >= len(t):
        raise ValueError("test segment too short")

    test_start_time = float(t[test_start_idx])
    gt_test = [
        (float(s), float(e))
        for s, e in gt_all
        if s >= test_start_time and e > s
    ]

    return {
        "tag": tag,
        "participant": _participant_from_tag(tag),
        "t": t,
        "x": x,
        "y": y,
        "dt": dt,
        "fs": fs,
        "max_time": max_time,
        "split_idx": split_idx,
        "sel_idx": sel_idx,
        "test_start_idx": test_start_idx,
        "test_start_time": test_start_time,
        "gt_test": gt_test,
        "gt_all": gt_all,
        "built": built,
        "features_norm": feats_norm,
        "targets_norm": targets_norm,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "se2": se2,
        "me2": me2,
        "dlog_se2": dlog_se2,
        "dlog_me2": dlog_me2,
        "se_th": se_th,
        "me_th": me_th,
        "se_slope_th": se_slope_th,
        "me_slope_th": me_slope_th,
    }
