"""Summarise existing warnings within the one-second event window; no fitting or simulation."""
from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, json, math
import numpy as np
from recompute_latest_run_metrics import flags, rows

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=ROOT)
    args=parser.parse_args(); source=args.data_root
    run=source/'analysis/latest_server_run_20261027'
    prior=json.loads((run/'recomputed_manuscript_metrics.json').read_text())
    baseline=defaultdict(dict)
    for row in rows(source/'analysis/baseline_comparison/origins.csv'):
        if row['method'] in ('age_survival','age_logistic_ungated','clock_gate'):
            baseline[row['record']].setdefault(row['method'],[]).append(row)
    methods={'DeepONet--Hopf':None,'Inter-peak survival':'age_survival',
             'Pure peak clock':'clock_gate','Peak-age logistic':'age_logistic_ungated'}
    prior_keys={'DeepONet--Hopf':'DeepONet-Hopf (latest 5 s endpoint)',
                'Inter-peak survival':'Inter-peak survival','Pure peak clock':'Pure peak clock',
                'Peak-age logistic':'Peak-age logistic'}
    all_events=[]; per_record=[]; scored=defaultdict(list); alarm_counts=defaultdict(int)
    input_hashes={}
    def hashed_rows(p):
        input_hashes[str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return rows(p)
    baseline_file=source/'analysis/baseline_comparison/origins.csv'
    input_hashes[str(baseline_file.relative_to(source))]=hashlib.sha256(baseline_file.read_bytes()).hexdigest()
    for rec in prior['per_record']:
        name=rec['record'];folder=run/'records'/name
        risk=hashed_rows(folder/'risk.csv');events=hashed_rows(folder/'event_warning_validation.csv')
        times=np.array([float(r['origin_time']) for r in risk]);boundary=times[0]
        events=[e for e in events if float(e['reference_onset'])>=boundary-1e-9]
        base_ref=baseline[name]['age_survival']
        assert np.allclose(times,[float(r['origin_time']) for r in base_ref],atol=1e-9,rtol=0)
        for method,key in methods.items():
            source_rows=risk if key is None else baseline[name][key]
            vals=[float(r['transition_probability' if key is None else 'risk']) for r in source_rows]
            if key is not None:assert np.allclose(times,[float(r['origin_time']) for r in source_rows],atol=1e-9,rtol=0)
            alarm_flags=flags(times,vals,.8,2,.05)
            if key is None:
                assert alarm_flags==[r['experimental_trigger'] in ('1','True','true') for r in risk]
            alarms=[float(t) for t,f in zip(times,alarm_flags) if f];alarm_counts[method]+=len(alarms)
            for r,base,p in zip(source_rows,base_ref,vals):
                label=int(float(r['future_onset_label' if key is None else 'label']))
                assert label==int(base['label'])
                if int(base['scored']):scored[method].append((p-label)**2)
            local=[]
            for event in events:
                onset=float(event['reference_onset']);end=float(event['previous_reference_end']) if event['previous_reference_end'] else -math.inf
                any_hits=[t for t in alarms if t> end+1e-9 and t<onset-1e-9]
                timely=[t for t in any_hits if 0<onset-t<=1+1e-9]
                lead_any=onset-min(any_hits) if any_hits else None
                lead_1=onset-min(timely) if timely else None
                e={'method':method,'record':name,'participant':rec['participant'],
                   'onset_s':onset,'previous_event_end_s':end,
                   'first_alarm_any_s':min(any_hits) if any_hits else None,'lead_any_s':lead_any,
                   'first_alarm_within_1s_s':min(timely) if timely else None,'lead_within_1s_s':lead_1,
                   'alarm_count_within_1s':len(timely),
                   'eligible_origins_within_1s':int(sum((times>end+1e-9)&(times<onset-1e-9)&(onset-times<=1+1e-9)))}
                local.append(e);all_events.append(e)
            a=[e['lead_any_s'] for e in local if e['lead_any_s'] is not None]
            b=[e['lead_within_1s_s'] for e in local if e['lead_within_1s_s'] is not None]
            per_record.append({'method':method,'record':name,'participant':rec['participant'],
                'events':len(local),'covered_within_1s':len(b),'covered_any':len(a),'alarms':len(alarms),
                'median_lead_within_1s_s':float(np.median(b)) if b else None,
                'median_lead_any_s':float(np.median(a)) if a else None,
                'normal_origins':len(times)})
    output={'primary_model':'latest 5-s endpoint','threshold':.8,'confirm':2,
            'primary_event_window_s':1.0,'simulation_horizon_s':5.0,'methods':{},'per_record':per_record,
            'lead_definition':'For each covered event, use its first qualifying alarm within [onset-1s, onset), including exactly 1 s and excluding the onset. The alarm must follow the previous event end. Pool event leads before taking the median.',
            'event_matching_tolerance_s':1e-9,'normal_exposure_minutes':prior['normal_exposure_minutes'],
            'new_training':False,'new_simulation':False,'retrospective_metric_reorganisation':True,
            'input_sha256':input_hashes}
    for method in methods:
        es=[e for e in all_events if e['method']==method]
        a=[e['lead_any_s'] for e in es if e['lead_any_s'] is not None]
        b=[e['lead_within_1s_s'] for e in es if e['lead_within_1s_s'] is not None]
        p=prior['methods'][prior_keys[method]]
        assert len(es)==91 and len(a)==p['covered_any_pre_onset'] and len(b)==p['covered_within_1s']
        assert alarm_counts[method]==p['alarms'] and np.isclose(np.median(a),p['median_first_warning_lead_any_s'])
        assert len(scored[method])==8234
        if p['brier_1s_event_probability'] is not None:
            assert np.isclose(np.mean(scored[method]),p['brier_1s_event_probability'])
        output['methods'][method]={'events':91,'covered_within_1s':len(b),'coverage_within_1s':len(b)/91,
            'alarms':alarm_counts[method],'alarms_per_normal_minute':alarm_counts[method]/prior['normal_exposure_minutes'],
            'alarms_followed_within_1s':p['alarms_followed_within_1s'],
            'median_lead_within_1s_s':float(np.median(b)),'mean_lead_within_1s_s':float(np.mean(b)),
            'lead_within_1s_s':b,'covered_any':len(a),'median_lead_any_s':float(np.median(a)),'lead_any_s':a,
            'brier':p['brier_1s_event_probability']}
    out=ROOT/'analysis/primary_within1s';out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for filename,data in [('events.csv',all_events),('per_record.csv',per_record)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps({k:{x:y for x,y in v.items() if not isinstance(y,list)} for k,v in output['methods'].items()},indent=2))

if __name__=='__main__':main()
