"""Causal network input windows; cycle duration does not alter reference labels."""
import numpy as np


def history_specification(raw_times, stride, history=40, history_cycles=None, cycle=1.13):
    raw_dt = float(np.median(np.diff(raw_times)))
    if history_cycles is None:
        samples = int(history) * int(stride)
        mode = 'fixed_blocks'
        requested = None
    else:
        if not np.isfinite(history_cycles) or history_cycles <= 0:
            raise ValueError('history_cycles must be positive and finite')
        if not np.allclose(np.diff(raw_times), raw_dt, rtol=.01, atol=1e-5):
            raise ValueError('Cycle-duration history requires regular raw timestamps')
        requested = float(history_cycles * cycle)
        # Both endpoints are included: at 100 Hz, [t-1.13,t] has 114 samples.
        samples = int(round(requested / raw_dt)) + 1
        mode = 'cycle_duration'
    blocks = int(np.ceil(samples / stride))
    if samples < 2 or blocks < 1:
        raise ValueError('History must contain at least two raw samples')
    return {'mode': mode, 'history_cycles': history_cycles, 'cycle_seconds': float(cycle),
            'requested_duration_seconds': requested,
            'duration_seconds': float((samples - 1) * raw_dt),
            'raw_dt': raw_dt, 'raw_samples': samples, 'input_blocks': blocks,
            'oldest_block_samples': int(samples - (blocks - 1) * stride),
            'stride': int(stride),
            'policy': 'End at origin; trim oldest block to requested duration; average remaining blocks as before'}


def network_histories(x, raw, ids, stride, mean, scale, specification):
    """Return normalized inputs, with no raw sample outside each history window."""
    length = specification['input_blocks']
    if specification['mode'] == 'fixed_blocks':
        return np.stack([x[i-length+1:i+1] for i in ids]).astype('float32')
    samples = specification['raw_samples']
    oldest = specification['oldest_block_samples']
    result = []
    for i in ids:
        stop = (int(i) + 1) * stride
        start = stop - samples
        if start < 0 or stop > len(raw):
            raise ValueError('Insufficient causal raw history')
        signal = raw[start:stop, 1:]
        first = signal[:oldest].mean(0, keepdims=True)
        rest = signal[oldest:].reshape(length-1, stride, 2).mean(1)
        result.append((np.concatenate((first, rest)) - mean) / scale)
    return np.asarray(result, dtype='float32')


def history_context_rows(raw, ids, specification):
    rows = []
    for i in ids:
        end = (int(i) + 1) * specification['stride'] - 1
        start = end - specification['raw_samples'] + 1
        rows.append({'origin_time': float(raw[end, 0]), 'history_start_time': float(raw[start, 0]),
                     'history_end_time': float(raw[end, 0]),
                     'duration_seconds': float(raw[end, 0] - raw[start, 0]),
                     'raw_start_index': start, 'raw_end_index': end,
                     'raw_samples': specification['raw_samples'], 'input_blocks': specification['input_blocks'],
                     'oldest_block_samples': specification['oldest_block_samples']})
    return rows
