"""Evaluate full and ablated predictors on the same 2-s cycle-event task."""
import argparse, csv, json, math, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from generalized_hopf_deeponet import make_model, threshold_warnings
from cycle_transition import observed_contexts
from multihorizon_events import first_new_onsets, cumulative_risks


def read_rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write_rows(path, data):
    if not data: return
    fields=list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def load_raw(path):
    with np.load(path) as d: return np.column_stack([d['t'],d['x'],d['y']]).astype(float)

def targets(times, events, horizon):
    return np.asarray([int(any(t+1e-9 < s <= t+horizon+1e-9 for s,e in events)) for t in times],dtype=int)

def eval_rows(record,method,times,risk,labels,events,boundary,end_time,dt,output):
    flags=np.asarray(threshold_warnings(times,risk,.8,2,dt),dtype=int)
    finite=np.isfinite(risk);complete=times+2.0<=end_time+1e-9
    score=finite&complete
    alarms=[]
    for t,p in zip(times[flags==1],risk[flags==1]):
        future=[(s,e) for s,e in events if s>t+1e-9]
        onset=future[0][0] if future else None
        previous=max([e for s,e in events if e<t+1e-9],default=-np.inf)
        lead=onset-float(t) if onset is not None else None
        category='matched' if onset is not None and previous < t and lead <= 2.+1e-9 else ('tail_censored' if t+2>end_time+1e-9 else 'unmatched')
        alarms.append(dict(record=record,method=method,warning_time=float(t),risk=float(p),next_onset=onset,
                           lead_seconds=lead,category=category))
    event_rows=[]
    for i,(s,e) in enumerate(events):
        if s<boundary-1e-9: continue
        previous=events[i-1][1] if i else -np.inf
        hits=[a['warning_time'] for a in alarms if previous<a['warning_time']<s-1e-9 and s-a['warning_time']<=2.+1e-9]
        any_hits=[a['warning_time'] for a in alarms if previous<a['warning_time']<s-1e-9]
        event_rows.append(dict(record=record,method=method,reference_onset=float(s),reference_end=float(e),
             previous_reference_end=None if not np.isfinite(previous) else float(previous),
             covered_within_2s=int(bool(hits)),lead_within_2s=(s-min(hits) if hits else None),
             covered_any_lead=int(bool(any_hits)),lead_any=(s-min(any_hits) if any_hits else None)))
    rr=[dict(record=record,method=method,origin_time=float(t),risk=float(p),label=int(y),
             complete_2s=int(c),scored_2s=int(q),alarm=int(f))
        for t,p,y,c,q,f in zip(times,risk,labels,complete,score,flags)]
    write_rows(output/f'{record}_{method}_origins.csv',rr)
    return alarms,event_rows,dict(record=record,method=method,origins=len(times),scored_origins=int(score.sum()),
       events=len(event_rows),covered_2s=int(sum(r['covered_within_2s'] for r in event_rows)),
       covered_any=int(sum(r['covered_any_lead'] for r in event_rows)),alarms=len(alarms),
       matched=int(sum(a['category']=='matched' for a in alarms)),unmatched=int(sum(a['category']=='unmatched' for a in alarms)),
       tail_censored=int(sum(a['category']=='tail_censored' for a in alarms)),
       normal_minutes=len(times)*dt/60.,brier=float(np.mean((risk[score]-labels[score])**2)) if score.any() else None,
       mean_risk=float(np.nanmean(risk)),target_prevalence=float(labels[score].mean()) if score.any() else None,
       median_lead_2s=float(np.median([r['lead_within_2s'] for r in event_rows if r['lead_within_2s'] is not None])) if any(r['lead_within_2s'] is not None for r in event_rows) else None,
       first_onsets_covered=sum(1 for r in event_rows if r['covered_within_2s']))

@torch.no_grad()
def simulate_deeponet(source_record,reference,record,method,device,seed,draws=128,batch_size=16):
    meta=json.loads((source_record/'metrics.json').read_text(encoding='utf-8'))
    ck=torch.load(source_record/'model.pt',map_location='cpu',weights_only=False)
    args=SimpleNamespace(**ck.get('args',meta['configuration']))
    assert int(args.horizon)==100 and args.observation_aggregation=='endpoint'
    assert float(getattr(args,'history_cycles',1.0))==1.0, 'Not the requested 1T run'
    model=make_model(args,float(ck['dt'])).to(device).eval();model.load_state_dict(ck['state_dict'])
    saved=np.load(source_record/'network_inputs.npz'); inputs=torch.tensor(saved['inputs'],dtype=torch.float32,device=device)
    times=saved['origin_time'].astype(float); raw=load_raw(reference/'data'/'records'/(record+'.npz'))
    stride=int(args.stride); blocks=raw[:len(raw)//stride*stride].reshape(-1,stride,3)
    bt=blocks[:,-1,0]; ids=np.searchsorted(bt,times)
    if not np.allclose(bt[ids],times,atol=1e-8,rtol=0): raise ValueError(f'{record}: input times do not align to raw record')
    settings=json.loads((source_record/'peak_calibration.json').read_text(encoding='utf-8'))
    contexts,ages=observed_contexts(raw,ids,stride,settings)
    events=[(float(r['onset']),float(r['offset'])) for r in read_rows(source_record/'reference_events.csv')]
    end=float(raw[-1,0]);boundary=float(meta['split']['test_boundary']);mean=torch.as_tensor(ck['mean'],device=device,dtype=torch.float32);scale=torch.as_tensor(ck['scale'],device=device,dtype=torch.float32)
    risks=[];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    for lo in range(0,len(inputs),batch_size):
        hi=min(lo+batch_size,len(inputs));r=model(inputs[lo:hi],draws,return_fine=True)
        force=(r['fine_joint']*scale+mean).detach().cpu().numpy()
        first=first_new_onsets(force,contexts[lo:hi],ages[lo:hi],settings,float(r['fine_dt']))
        risks.extend(cumulative_risks(first)[:,1].tolist())
    risk=np.asarray(risks,dtype=float);label=targets(times,events,2.)
    return times,risk,label,events,boundary,end,float(ck['dt'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference-root',required=True); ap.add_argument('--base-run',required=True)
    ap.add_argument('--no-event-run',required=True); ap.add_argument('--constant-run',required=True); ap.add_argument('--mlp-run',required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--gpus',default='0,1,2,3'); ap.add_argument('--tags',help='Optional comma-separated subset for smoke testing')
    a=ap.parse_args(); ref=Path(a.reference_root); runs={'Full DeepONet–Hopf (event loss)':Path(a.base_run),
         'No event loss':Path(a.no_event_run),'Constant future coefficients':Path(a.constant_run)}
    mlp=Path(a.mlp_run); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(a.base_run)/'input_manifest.json').read_text(encoding='utf-8'))
    if not manifest: raise ValueError('Empty 21-record manifest')
    if a.tags:
        wanted=set(a.tags.split(',')); manifest=[r for r in manifest if r['tag'] in wanted]
        if not manifest: ap.error('None of --tags matched the canonical manifest')
    ids=[r['tag'].replace(' ','_') for r in manifest]
    for label,path in runs.items():
        if label=='Full DeepONet–Hopf (event loss)': continue
        status=json.loads((path/'status.json').read_text(encoding='utf-8'))
        failed=[tag for tag in [r['tag'] for r in manifest] if status.get('records',{}).get(tag,{}).get('status')!='complete']
        if failed: raise RuntimeError(f'{label} is incomplete: {failed}')
    device_ids=[int(q) for q in a.gpus.split(',')]
    result_rows=[];alarm_rows=[];event_rows=[];records=[]
    for record in ids:
        for j,(method,base_run) in enumerate(runs.items()):
            dev=torch.device(f'cuda:{device_ids[j%len(device_ids)]}' if torch.cuda.is_available() else 'cpu')
            src=base_run/'records'/record
            times,risk,label,events,boundary,end,dt=simulate_deeponet(src,ref,record,method,dev,a.seed)
            alarms,ev,rr=eval_rows(record,method,times,risk,label,events,boundary,end,dt,out)
            result_rows.append(rr);alarm_rows.extend(alarms);event_rows.extend(ev)
        # MLP uses the exact same normal test origins and labels as the fitted models.
        mlp_record=mlp/'records'/record
        mmeta=json.loads((Path(a.base_run)/'records'/record/'metrics.json').read_text(encoding='utf-8'))
        mrows=read_rows(mlp_record/'risk.csv'); mt=np.asarray([float(r['origin_time']) for r in mrows]); mr=np.asarray([float(r['direct_risk']) for r in mrows]); my=np.asarray([int(float(r['future_onset_label'])) for r in mrows])
        if not np.array_equal(mt,np.asarray(np.load(Path(a.base_run)/'records'/record/'network_inputs.npz')['origin_time'],dtype=float)):
            raise ValueError(f'{record}: direct MLP and DeepONet test origins differ')
        events=[(float(r['onset']),float(r['offset'])) for r in read_rows(Path(a.base_run)/'records'/record/'reference_events.csv')]
        raw=load_raw(ref/'data'/'records'/(record+'.npz')); boundary=float(mmeta['split']['test_boundary']); dt=float(mmeta['dt']); end=float(raw[-1,0])
        alarms,ev,rr=eval_rows(record,'Direct risk MLP',mt,mr,my,events,boundary,end,dt,out)
        result_rows.append(rr);alarm_rows.extend(alarms);event_rows.extend(ev)
        records.append(record);print(f'Finished {record}',flush=True)
    write_rows(out/'per_record.csv',result_rows);write_rows(out/'alarms.csv',alarm_rows);write_rows(out/'events.csv',event_rows)
    methods=list(dict.fromkeys(r['method'] for r in result_rows));summary=[]
    participant_count=len({r['tag'].split('_FD')[0] for r in manifest})
    for method in methods:
        rr=[r for r in result_rows if r['method']==method]
        ev=[r for r in event_rows if r['method']==method]
        aa=[r for r in alarm_rows if r['method']==method]
        score_rows=[]
        for record in records: score_rows+=read_rows(out/f'{record}_{method}_origins.csv')
        pooled=[r for r in score_rows if int(r['scored_2s'])==1]
        all_leads=[float(r['lead_within_2s']) for r in ev if r['lead_within_2s'] not in ('',None)]
        total_events=sum(int(r['events']) for r in rr);covered=sum(int(r['covered_2s']) for r in rr)
        minutes=sum(float(r['normal_minutes']) for r in rr)
        summary.append(dict(method=method,records=len(rr),participants=participant_count,test_events=total_events,
             covered_within_2s=covered,event_coverage=covered/total_events if total_events else None,
             alarms=len(aa),alarms_per_normal_minute=len(aa)/minutes if minutes else None,
             matched_alarms=sum(r['category']=='matched' for r in aa),unmatched_alarms=sum(r['category']=='unmatched' for r in aa),
             tail_censored_alarms=sum(r['category']=='tail_censored' for r in aa),
             median_lead_within_2s=float(np.median(all_leads)) if all_leads else None,
             brier_2s=float(np.mean([(float(r['risk'])-int(r['label']))**2 for r in pooled])) if pooled else None,
             scored_origins=len(pooled),target_prevalence=float(np.mean([int(r['label']) for r in pooled])) if pooled else None,
             mean_risk=float(np.mean([float(r['risk']) for r in pooled])) if pooled else None,
             warnings_within2s_fraction=sum(r['category']=='matched' for r in aa)/(sum(r['category'] in ('matched','unmatched') for r in aa) or 1)))
    write_rows(out/'summary.csv',summary)
    protocol=dict(records=len(records),participants=participant_count,network_history_cycles=1.0,network_history_seconds=1.13,seed=a.seed,
        simulator_horizon_seconds=5.,evaluation_event_window_seconds=2.,paths=128,threshold_strictly_greater_than=.8,
        alarm_confirmation='two consecutive rolling origins separated by 0.05 s',reference='same paper_2020 cycle-gap intervals, zero extra event merging',
        split='within each record: first 52% fit, next 13% validation, final 35% test; same offline normal-only screen',
        full_model_and_constant_auxiliary_training_event_window_seconds=1.,
        no_event_loss='same DeepONet–Hopf training objective with event-loss coefficient zero',
        constant_future_coefficients='one coefficient triple per origin, held fixed over each 5-s SDE rollout; re-inferred at each next origin',
        direct_mlp='MLP predicts probability of reference onset in next 2 s from same 1T force history plus ~2T observed peak context and full-prefix peak age',
        probability='DeepONet–Hopf: fraction of 128 paths whose first new cycle-gap onset falls within 2 s; MLP: sigmoid risk output',
        evaluation='event coverage requires an alarm after prior reference end and within 2 s before onset; Brier uses complete 2-s follow-up normal origins')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Compact chart and browsable report are built on the SSH host with matplotlib available.
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    color=['#2878B5','#D17C2F','#299A82','#9157A4']; fig,axs=plt.subplots(1,3,figsize=(13,4.2))
    for ax,key,title in zip(axs,['event_coverage','alarms_per_normal_minute','brier_2s'],['2-s event coverage','Alarms per normal minute','2-s Brier score']):
        values=[x[key] or 0 for x in summary]; bars=ax.bar(range(len(summary)),values,color=color[:len(summary)])
        ax.set_xticks(range(len(summary)),[x['method'].replace(' (event loss)','') for x in summary],rotation=28,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.3f}' if key!='alarms_per_normal_minute' else f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    def fmt(v,d=3): return '—' if v is None else f'{float(v):.{d}f}'
    table=''.join('<tr>'+''.join(f'<td>{x}</td>' for x in [r['method'],f"{r['covered_within_2s']}/{r['test_events']} ({fmt(r['event_coverage']*100,1)}%)",r['alarms'],fmt(r['alarms_per_normal_minute'],1),fmt(r['median_lead_within_2s']),fmt(r['brier_2s']),r['scored_origins']])+'</tr>' for r in summary)
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2-s Ablation Comparison</title><style>body{font:15px/1.55 Segoe UI,Arial,sans-serif;max-width:1180px;margin:28px auto;padding:0 20px;color:#203247}h1{font-size:27px}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border-bottom:1px solid #dbe3eb;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}.box{padding:14px;background:#f4f7fa;border-radius:8px}</style>'''+f'''<h1>DeepONet–Hopf 2-s Ablation Experiments</h1><div class="box">All methods use the same 21 recordings, 1T force history (1.13 s), last-35% test split, reference labels, offline normal-origin screen and 0.80/two-origin alarm rule. All risk scores target a new onset within 2 s; simulations remain 5 s; seed={a.seed}. The full and constant-coefficient models retain the original auxiliary 1-s soft event objective; the no-event-loss control sets its weight to zero. The direct MLP is trained directly on 2-s labels and receives force history plus the same peak context.</div><table><thead><tr><th>Method</th><th>Coverage within 2 s</th><th>Alarms</th><th>Alarms/min</th><th>Median lead (s)</th><th>Brier (2 s)</th><th>Scored origins</th></tr></thead><tbody>{table}</tbody></table><img src="comparison.png"><p>Event coverage counts an alarm only if it occurs after the previous reference event ends and no more than 2 s before the reference onset. Brier uses normal test origins with a complete 2-s follow-up. These are retrospective within-record results, not held-out-participant validation.</p><p><a href="summary.csv">Summary CSV</a> · <a href="per_record.csv">Per-record results</a> · <a href="events.csv">Event-level matches</a> · <a href="alarms.csv">Alarm table</a> · <a href="protocol.json">Protocol</a></p>'''
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
