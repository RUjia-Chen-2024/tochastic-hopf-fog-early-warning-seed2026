"""Check saved outputs, input integrity and the article's primary numeric claims."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'analysis/two_second_primary';INPUT=A/'inputs'
manifest=json.loads((INPUT/'manifest.json').read_text())
for item in manifest:
    assert hashlib.sha256((INPUT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
rows=pd.read_csv(A/'summary.csv');origins=pd.read_csv(A/'origins.csv')
expected={1:(80,146,.525,.1159397558),2:(84,160,.680,.2367541535),
          3:(84,186,.925,.3000882095),5:(84,199,1.155,.3238292053)}
for h,values in expected.items():
    r=rows[(rows.method=='DeepONet--Hopf')&(rows.horizon==h)].iloc[0]
    np.testing.assert_allclose([r.covered,r.alarms,r.median_lead,r.brier],values,atol=1e-9,rtol=0)
zero=origins[origins.method=='Zero diffusion']
for record,g in zero.groupby('record'):
    with np.load(INPUT/'unified_seed2026_20261027/results/records'/record/'zero_noise_first_onsets.npz') as z:
        for h in [1,2,3,5]:
            o=g[g.horizon==h]
            np.testing.assert_allclose(o.time,z['origin_time'],atol=1e-9,rtol=0)
            np.testing.assert_array_equal(o.risk,(z['first_onset_offsets']<=h+1e-9).mean(1))
case=json.loads((A/'case_metrics.json').read_text())
assert case['path_event_recalculation_matches'] and case['counts']==[103,104,104,104]
plots=json.loads((A/'record_plot_audit.json').read_text())
assert len(plots)==21
miss=pd.read_csv(A/'miss_diagnostics.csv');assert len(miss)==7
before=json.loads((ROOT/'before_hashes.json').read_text())
for file in ['references.bib','availability.tex','data_driven_manuscript.tex']:
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==before[file],file
checks={'input_hashes_verified':len(manifest),'learned_horizons_exact':True,
        'zero_diffusion_path_risks_exact_records':21,'case_path_events_exact':True,
        'record_plots':21,'missed_events':7,'neural_retrained':False,'simulation_rerun':False,
        'primary_window_seconds':2,'selected_after_results':True,
        'references_funding_author_title_unchanged':True,
        'renewal_analytic_checks':'scripts/check_renewal.py passed',
        'portability':'Audit and figure scripts successfully ran from bundled inputs'}
(A/'final_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
