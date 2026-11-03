"""First NEW cycle-gap onset on a simulated 5-s force path.

Extends the frozen one-second rule to later simulated stepping cycles.
Observed peaks remain fixed; future accepted peaks use the same SciPy detector,
normalization, observed waveform context and minimum-distance rule as before.
An already overdue observed gap is not counted as a new onset after time zero.
"""
import numpy as np
from scipy.signal import find_peaks

HORIZONS = np.array([1., 2., 3., 5.])


def first_new_onsets(force_paths, contexts, ages, settings, fine_dt):
    paths = np.asarray(force_paths)
    batch, draws, steps, _ = paths.shape
    end = steps * fine_dt
    signal = np.abs(paths[..., 0] - paths[..., 1]) / settings['peak_scale']
    first = np.full((batch, draws), np.inf, dtype=np.float64)
    gap = settings['abnormal_gap']
    assert abs(gap - settings['onset_delay']) < 1e-9
    for b, age in enumerate(ages):
        if not np.isfinite(age):
            first[b] = np.nan
            continue
        history_length = len(contexts[b])
        minimum = .45 * settings['cycle'] - float(age)
        for m in range(draws):
            sequence = np.r_[contexts[b], signal[b, m]]
            peaks, _ = find_peaks(sequence, distance=settings['distance_samples'],
                                 height=settings['height'], prominence=settings['prominence'])
            times = (peaks - history_length + 1) * fine_dt
            future = times[(peaks >= history_length) & (times >= minimum - 1e-9)]
            last = -float(age)
            for next_peak in np.r_[future, np.inf]:
                deadline = last + gap
                if deadline > end + 1e-9:
                    break
                if deadline > 1e-9 and next_peak > deadline + 1e-9:
                    first[b, m] = deadline
                    break
                # deadline <= 0 means this observed gap was already overdue.
                # A predicted accepted peak must terminate it before a new gap.
                if not np.isfinite(next_peak):
                    break
                last = float(next_peak)
    return first


def cumulative_risks(first, horizons=HORIZONS):
    risks = np.stack([(first <= horizon + 1e-9).mean(1) for horizon in horizons], axis=1)
    risks[np.isnan(first).any(1)] = np.nan
    return risks


def synthetic_checks():
    dt = .01
    settings = dict(peak_scale=1., distance_samples=51, height=.4, prominence=.08,
                    cycle=1.13, abnormal_gap=1.695, onset_delay=1.695)
    def outcome(age, peaks):
        context = np.zeros((1, 227))
        at = 226 - round(age / dt)
        if at >= 0:
            context[0, at] = 1.
        force = np.zeros((1, 1, 500, 2))
        for t in peaks:
            force[0, 0, round(t / dt) - 1, 0] = 1.
        first = first_new_onsets(force, context, np.array([age]), settings, dt)
        return first[0, 0], cumulative_risks(first)[0]
    onset, p = outcome(.8, [])
    assert abs(onset - .895) < 1e-8 and np.array_equal(p, [1, 1, 1, 1])
    onset, p = outcome(.2, [.5])
    assert abs(onset - 2.195) < 1e-8 and np.array_equal(p, [0, 0, 1, 1])
    onset, p = outcome(.2, [1., 2., 3., 4.])
    assert np.isinf(onset) and np.array_equal(p, [0, 0, 0, 0])
    onset, p = outcome(2., [])
    assert np.isinf(onset) and np.array_equal(p, [0, 0, 0, 0])
    onset, p = outcome(2., [.5])
    assert abs(onset - 2.195) < 1e-8 and np.array_equal(p, [0, 0, 1, 1])
    print('Five behavioral checks passed: immediate gap, later gap, regular cycles, ongoing gap, recovery then new gap.')


if __name__ == '__main__':
    synthetic_checks()
