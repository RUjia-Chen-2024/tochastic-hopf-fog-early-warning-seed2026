"""Check published checkpoint hashes; optionally load all models and run smoke forecasts."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Requires PyTorch and model dependencies')
    args = parser.parse_args()
    index = json.loads((ROOT/'MODEL_INDEX.json').read_text())
    counts = Counter(row['family'] for row in index)
    assert len(counts) == 6 and set(counts.values()) == {21}
    for row in index:
        path = ROOT/row['path']
        assert path.stat().st_size == row['bytes'], row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
    fits = json.loads((ROOT/'models/statistical/baseline_models.json').read_text())
    assert len(fits) == 21
    for record, values in fits.items():
        assert set(values['logistic']) == {'1','2','3','5'}
        assert set(values['constant']) == {'1','2','3','5'}
        assert values['survival']['kind'] == 'Kaplan_Meier'
    print('PASS: 126 original checkpoint hashes; 21 saved statistical model sets.')
    if not args.smoke:
        return
    import numpy as np
    import torch
    from predict import load_model, import_file
    torch.set_num_threads(1)
    first = set()
    for row in index:
        family = row['family']
        model, checkpoint, _ = load_model(family,row['record'])
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if family in first:
            continue
        first.add(family)
        torch.manual_seed(2026)
        with torch.no_grad():
            if family == 'direct_risk_mlp':
                output = torch.sigmoid(model(torch.zeros(1,checkpoint['n_features'])))
                assert output.shape == (1,) and torch.isfinite(output).all()
            else:
                output = model(torch.zeros(1,23,2),draws=2,return_fine=True)
                assert output['fine_joint'].shape == (1,2,500,2)
                assert output['params'].shape == (1,100,3)
                assert torch.isfinite(output['fine_joint']).all()
                if family == 'constant_coefficients':
                    assert torch.equal(output['params'][:,0],output['params'][:,-1])
        print('PASS: strict state loading and smoke forward:',family)
    detector = import_file('_verify_event_rule',ROOT/'code/interventions/multihorizon_events.py')
    detector.synthetic_checks()
    stats = import_file('_verify_statistical',ROOT/'code/statistical.py')
    for record,spec in fits.items():
        ages=np.arange(0,5,.05)
        for method in ('survival','peak_clock','peak_age_logistic','fit_constant','constant_half'):
            for h in (1.,2.,3.,5.):
                p=stats.predict_baseline(spec,ages,h,method)
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    print('PASS: all 126 strict loads, all baseline records/windows, and event-rule checks.')


if __name__ == '__main__':
    main()
