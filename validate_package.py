from pathlib import Path
import csv, json, math

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT/'protocol'/'canonical_protocol.json').read_text(encoding='utf-8'))
assert protocol['seeds']['primary_evaluation_seed'] == 2026
assert protocol['seeds']['component_replay_seed'] == 2026
assert protocol['forecast']['simulation_horizon_seconds'] == 5.0
assert protocol['forecast']['primary_event_horizon_seconds'] == 2.0
assert protocol['forecast']['paths'] == 128
assert protocol['history']['cycles'] == 1.0
assert protocol['alarm']['risk_threshold'] == 0.8 and protocol['alarm']['consecutive_origins'] == 2

def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

noise = rows(ROOT/'results'/'primary_seed2026'/'noise_by_horizon.csv')
noise2 = {r['variant']:r for r in noise if float(r['horizon_s']) == 2}
assert int(noise2['zero_noise']['covered_within_horizon']) == 83
assert int(noise2['zero_noise']['alarms']) == 302
assert math.isclose(float(noise2['zero_noise']['median_lead_s']), 1.295, abs_tol=1e-9)
assert int(noise2['learned_seed_2026']['covered_within_horizon']) == 84
assert int(noise2['learned_seed_2026']['alarms']) == 160
assert math.isclose(float(noise2['learned_seed_2026']['median_lead_s']), .68, abs_tol=1e-9)

controls = rows(ROOT/'results'/'component_replay_seed2026'/'evaluation'/'summary.csv')
full = next(r for r in controls if r['method'].startswith('Full DeepONet'))
assert int(full['covered_within_2s']) == 83 and int(full['alarms']) == 166
assert math.isclose(float(full['median_lead_within_2s']), .685, abs_tol=1e-9)

frozen = rows(ROOT/'results'/'parameter_sensitivity_seed2026'/'frequency_and_sigma'/'parameter_replay_summary.csv')
assert any(r['scenario'].startswith('sigma(s) fixed') and int(r['alarms']) == 166 for r in frozen)
decoder_rows = rows(ROOT/'results'/'decoder_limit_cycle_audit_20261030'/'decoder_invertibility_all21.csv')
decoder_summary = json.loads((ROOT/'results'/'decoder_limit_cycle_audit_20261030'/'summary.json').read_text(encoding='utf-8'))
assert len(decoder_rows) == 21
assert all(float(r['det_A_raw']) != 0 and float(r['A_singular_min']) > 0 for r in decoder_rows)
assert decoder_summary['all_21_decoder_matrices_invertible_numerically'] is True
assert math.isclose(float(decoder_summary['condition_number_raw_A']['median']), 68.99181543577131, rel_tol=1e-9)
assert math.isclose(float(decoder_summary['condition_number_raw_A']['max']), 434.5888330802463, rel_tol=1e-9)
streams = json.loads((ROOT/'protocol'/'experiment_streams.json').read_text(encoding='utf-8'))
assert streams['legacy_excluded']['seed'] == 20261027
assert streams['primary_evaluation']['seed'] == streams['component_replay']['seed'] == 2026
print('PASS: canonical protocol, seed-2026 result streams, noise contrast, component baseline, sigma-freeze distinction, and all-21 decoder audit verified.')
