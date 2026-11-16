"""Load a published main-text model and predict at one causal observation time.

This convenience entry point does not recreate the archived batched random stream.
Use the archived replay scripts and original inputs for exact manuscript replays.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FAMILIES = ('full', 'no_event_loss', 'constant_coefficients', 'direct_risk_mlp',
            'alignment_endpoint', 'alignment_blockmean')
BASELINES = ('survival', 'peak_age_logistic', 'peak_clock', 'fit_constant', 'constant_half')


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model(family, record, device='cpu'):
    if family not in FAMILIES:
        raise ValueError('Unknown trained family: ' + family)
    path = ROOT / 'models' / family / record / 'model.pt'
    if not path.is_file():
        raise ValueError('Record not present in the published model index: ' + record)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if family == 'direct_risk_mlp':
        directory = ROOT / 'code' / 'controls'
        sys.path.insert(0, str(directory))
        module = import_file('_main_mlp', directory / 'train_direct_risk_mlp.py')
        model = module.DirectRiskMLP(int(checkpoint['n_features']))
    else:
        directory = (ROOT / 'code' / 'primary' if family == 'full' else
                     ROOT / 'code' / 'observation_alignment' / 'training'
                     if family.startswith('alignment_') else ROOT / 'code' / 'controls')
        sys.path.insert(0, str(directory))
        module = import_file('_main_' + family, directory / 'generalized_hopf_deeponet.py')
        model = module.make_model(SimpleNamespace(**checkpoint['args']), checkpoint['dt'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint, module


def raw_data(path):
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            raw = np.column_stack([data[k] for k in ('t', 'x', 'y')]).astype(float)
    else:
        data = np.genfromtxt(path, delimiter=',', names=True)
        raw = np.column_stack([data[k] for k in ('t', 'x', 'y')]).astype(float)
    if len(raw) < 114 or not np.isfinite(raw).all() or np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError('Supply at least 114 finite, chronologically ordered t,x,y samples')
    if not np.allclose(np.diff(raw[:, 0]), .01, rtol=.01, atol=1e-6):
        raise ValueError('The published models require regularly sampled 100-Hz input')
    return raw


def prepare_input(raw, time, checkpoint):
    module = import_file('_main_history', ROOT / 'code' / 'primary' / 'history_window.py')
    count = len(raw) // 5
    blocks = raw[:count*5].reshape(count, 5, 3)
    times = blocks[:, -1, 0]
    index = int(np.argmin(abs(times-time)))
    if abs(times[index]-time) > 1e-7:
        raise ValueError('Use an available block endpoint, e.g. 89.94 for Pt_33_FD33_4')
    spec = module.history_specification(raw[:, 0], 5, 23, 1., 1.13)
    x = ((blocks[:, :, 1:].mean(1)-checkpoint['mean'])/checkpoint['scale']).astype('float32')
    history = module.network_histories(x, raw, [index], 5, checkpoint['mean'], checkpoint['scale'], spec)
    return index, spec, history, blocks[index, :, 1:].mean(0)


def finite_number(x):
    value = float(x)
    return value if np.isfinite(value) else None


@torch.no_grad()
def prediction(args):
    torch.set_num_threads(1)
    if args.model in BASELINES and args.age is not None:
        age = np.array([args.age], dtype=float)
        index = spec = history = raw = None
        base = None
    else:
        if args.data is None or args.time is None:
            raise ValueError('Provide --data and --time; statistical baselines also accept --age')
        raw = raw_data(args.data)
        # MLP features use the original full-model force normalisation.
        family = args.model if args.model in FAMILIES and args.model != 'direct_risk_mlp' else 'full'
        model, checkpoint, engine = load_model(family, args.record, args.device)
        base = checkpoint
        index, spec, history, last_block = prepare_input(raw, args.time, checkpoint)
        settings = json.loads((ROOT/'records'/args.record/'peak_calibration.json').read_text())
        cycle = import_file('_main_cycle', ROOT/'code'/'primary'/'cycle_transition.py')
        contexts, age = cycle.observed_contexts(raw, [index], 5, settings)

    result = dict(model=args.model, record=args.record, prediction_time_s=args.time,
                  peak_age_s=finite_number(age[0]), seed=args.seed,
                  event_windows_s=[1, 2, 3, 5], warning_threshold=.80, confirmation=2)
    arrays = {}
    if args.model in BASELINES:
        models = json.loads((ROOT/'models'/'statistical'/'baseline_models.json').read_text())
        statistical = import_file('_main_statistical', ROOT/'code'/'statistical.py')
        risk = [statistical.predict_baseline(models[args.record], age, float(h), args.model)[0]
                for h in (1, 2, 3, 5)]
        result['risk'] = [finite_number(p) for p in risk]
    elif args.model == 'persistence':
        arrays['force_mean'] = np.repeat(last_block[None, :], 100, axis=0)
        result.update(force_shape=[100, 2], event_windows_s=[],
                      note='Persistence is a force-forecast comparator; no event risk is assigned.')
    elif args.model == 'direct_risk_mlp':
        model, checkpoint, engine = load_model(args.model, args.record, args.device)
        features, _ = engine.feature_matrix(raw, np.array([index]), 5, base['mean'], base['scale'], spec, settings, 1.13)
        features = ((features-checkpoint['feature_mean'])/checkpoint['feature_scale']).astype('float32')
        risk = torch.sigmoid(model(torch.tensor(features, device=args.device))).item()
        result.update(event_windows_s=[2], risk=[risk], draws=0)
    else:
        physical = model.physical_parameters
        inputs = torch.tensor(history, dtype=torch.float32, device=args.device)
        state = model.encoder(inputs.flatten(1))
        current = physical(model.param_head(state))
        if args.freeze != 'none' or args.noise_multiplier != 1.:
            def intervention(logits):
                parameters = physical(logits).clone()
                if parameters.ndim == 3 and args.freeze != 'none':
                    column = {'beta': 0, 'frequency': 1, 'sigma': 2}[args.freeze]
                    parameters[:, :, column] = current[:, None, column]
                parameters[..., 2] *= args.noise_multiplier
                return parameters
            model.physical_parameters = intervention
        torch.manual_seed(args.seed)
        generated = model(inputs, draws=args.draws, return_fine=True)
        scale = torch.tensor(checkpoint['scale'], dtype=torch.float32, device=args.device)
        mean = torch.tensor(checkpoint['mean'], dtype=torch.float32, device=args.device)
        paths = (generated['fine_joint']*scale+mean).cpu().numpy()
        events = import_file('_main_events', ROOT/'code'/'interventions'/'multihorizon_events.py')
        first = events.first_new_onsets(paths, contexts, age, settings, generated['fine_dt'])
        risk = events.cumulative_risks(first)[0]
        result.update(risk=[finite_number(p) for p in risk], draws=args.draws,
                      noise_multiplier=args.noise_multiplier, freeze=args.freeze,
                      note='A fresh single-time simulation, not the archived manuscript batch stream.')
        arrays.update(force_paths=paths[0], parameters=generated['params'][0].cpu().numpy(),
                      first_onsets=first[0], fine_dt=np.array(generated['fine_dt']))
    if 'risk' in result:
        result['note_alarm'] = 'One prediction cannot issue a two-update alarm; apply the shared state machine to a chronological risk stream.'
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=FAMILIES+BASELINES+('persistence',), default='full')
    parser.add_argument('--record', required=True)
    parser.add_argument('--data', type=Path, help='100-Hz NPZ with t,x,y arrays, or CSV with t,x,y headers')
    parser.add_argument('--time', type=float)
    parser.add_argument('--age', type=float, help='Statistical baselines only: elapsed seconds since the last accepted peak')
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--noise-multiplier', type=float, default=1.)
    parser.add_argument('--freeze', choices=('none','beta','frequency','sigma'), default='none')
    parser.add_argument('--out', type=Path, help='Optional JSON result; decoded arrays saved beside it as NPZ')
    args = parser.parse_args()
    if args.draws < 1 or not np.isfinite(args.noise_multiplier) or args.noise_multiplier < 0:
        parser.error('draws must be positive and noise multiplier finite and non-negative')
    if args.age is not None and (not np.isfinite(args.age) or args.age < 0):
        parser.error('age must be finite and non-negative')
    if args.model not in FAMILIES or args.model == 'direct_risk_mlp':
        if args.freeze != 'none' or args.noise_multiplier != 1.:
            parser.error('Noise and coefficient interventions apply only to a dynamical model')
    result, arrays = prediction(args)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text+'\n', encoding='utf8')
        if arrays:
            np.savez_compressed(args.out.with_suffix('.npz'), **arrays)
    print(text)


if __name__ == '__main__':
    main()
