"""Run the parameter-DeepONet Hopf experiment over the canonical 21-record manifest.

One independent model per record; four GPU worker processes; fixed settings.
Writes incremental status, per-record plots and pooled event tables.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def collect_results(out,manifest,status):
    import numpy as np
    records=[];events=[];warnings=[]
    for entry in manifest:
        tag=entry['tag'];folder=out/'records'/tag.replace(' ','_')
        info=status['records'][tag]
        if info['status']!='complete':continue
        m=json.loads((folder/'metrics.json').read_text());cfg=m['configuration'];ev=read_csv(folder/'event_warning_validation.csv');al=read_csv(folder/'warning_times.csv')
        participant=tag.split('_FD')[0]
        for row in ev:events.append({'tag':tag,'participant':participant,**row})
        for row in al:warnings.append({'tag':tag,'participant':participant,**row})
        leads=[float(row['lead_seconds']) for row in ev if row['warned_before']=='True']
        score=m['warning_evaluation'];forecast=m['metrics']['parameter_hopf']
        records.append({'tag':tag,'participant':participant,'n_events':score['n_test_events'],
            'warned_before':score['events_warned_before'],'missed':score['events_missed'],'coverage':score['event_coverage'],
            'n_warnings':score['n_warnings'],'unresolved_tail_warnings':score['warnings_without_later_observed_onset'],
            'median_lead_sec':float(np.median(leads)) if leads else None,'mean_lead_sec':float(np.mean(leads)) if leads else None,
            'normal_origins':m['windows'][2],'first_origin':m['split']['first_test_origin'],
            'epochs_run':m['epochs_run'],'best_epoch':m['best_epoch'],'rmse_x':forecast['rmse_xy'][0],'rmse_y':forecast['rmse_xy'][1],
            'reused':bool(info.get('reused_from')),'seconds':info.get('seconds',0),'result_directory':str(folder)})
    recfields=['tag','participant','n_events','warned_before','missed','coverage','n_warnings','unresolved_tail_warnings',
               'median_lead_sec','mean_lead_sec','normal_origins','first_origin','epochs_run','best_epoch','rmse_x','rmse_y','reused','seconds','result_directory']
    eventfields=['tag','participant','reference_onset','reference_end','previous_reference_end','eligible_origins','warned_before','first_warning','lead_seconds','warning_count']
    warningfields=['tag','participant','warning_time','next_reference_onset','before_next_event','lead_seconds']
    write_csv(out/'per_record.csv',records,recfields);write_csv(out/'events.csv',events,eventfields);write_csv(out/'warnings.csv',warnings,warningfields)
    covered=[r for r in events if r['warned_before']=='True'];leads=np.array([float(r['lead_seconds']) for r in covered])
    count=sum(r['n_events'] for r in records);success=len(covered)
    participants=[]
    for person in sorted(set(r['participant'] for r in records),key=lambda s:int(s.split()[-1])):
        rr=[r for r in records if r['participant']==person];nn=sum(r['n_events'] for r in rr);cc=sum(r['warned_before'] for r in rr)
        pl=[float(r['lead_seconds']) for r in covered if r['participant']==person]
        participants.append({'participant':person,'records':len(rr),'n_events':nn,'warned_before':cc,'missed':nn-cc,
                             'coverage':cc/nn if nn else None,'median_lead_sec':float(np.median(pl)) if pl else None})
    write_csv(out/'per_participant.csv',participants,['participant','records','n_events','warned_before','missed','coverage','median_lead_sec'])
    result={'requested_records':len(manifest),'completed_records':len(records),'failed_records':[tag for tag,r in status['records'].items() if r['status']=='failed'],
            'participants':len(participants),'test_reference_events':count,'events_warned_before':success,'events_missed':count-success,
            'event_coverage':success/count if count else None,'warnings':sum(r['n_warnings'] for r in records),
            'unresolved_tail_warnings':sum(r['unresolved_tail_warnings'] for r in records),
            'median_lead_sec':float(np.median(leads)) if len(leads) else None,
            'lead_q25_sec':float(np.quantile(leads,.25)) if len(leads) else None,'lead_q75_sec':float(np.quantile(leads,.75)) if len(leads) else None,
            'lead_min_sec':float(leads.min()) if len(leads) else None,'lead_max_sec':float(leads.max()) if len(leads) else None,
            'normal_origins':sum(r['normal_origins'] for r in records),
            'protocol':'52% fit /13% validation /35% test per record; normal origins screened by offline cycle reference; risk >0.7 issues immediately; one issuance per continuous exceedance; no maximum lead',
            'interpretation':'Within-record conditional evaluation against algorithmic labels. Not held-out-participant validation. Immediate alarm rule selected after the single-record experiment.'}
    if (out/'protocol.json').exists():
        protocol=json.loads((out/'protocol.json').read_text())
        if protocol.get('history_cycles') is not None:
            result['history_cycles']=protocol['history_cycles'];result['network_history']=protocol['network_history']
        if protocol.get('risk_rule')=='cycle_gap':
            merging='no extra event merging' if protocol['merge_gap_cycles']==0 else 'explicit legacy event merging'
            result['protocol']='52% fit /13% validation /35% test; '+protocol.get('reference_rule','legacy')+' cycle-gap labels; '+merging+'; normal origins only; decoded-force cycle probability within '+str(protocol['event_horizon'])+' s; alarm if risk > '+str(protocol['alarm_threshold'])+' for '+str(protocol['confirm'])+' consecutive origins'
            result['risk_rule']='cycle_gap';result['merge_gap_seconds']=protocol['cycle']*protocol['merge_gap_cycles']
            result['records_without_normal_test_origins']=[r['tag'] for r in records if not r['normal_origins']]
            result['evaluated_records']=sum(bool(r['normal_origins']) for r in records)
            result['simulation_horizon_steps']=protocol['horizon']
            result['simulation_horizon_seconds']=protocol.get('simulation_horizon_seconds',protocol['horizon']*.05)
            result['event_horizon_seconds']=protocol['event_horizon']
            result['alarm_rule']={'threshold':protocol['alarm_threshold'],'strict_comparison':'>',
                                 'consecutive_origins':protocol['confirm'],'origin_interval_seconds':.05}
        if protocol.get('label_policy') is not None:result['label_policy']=protocol['label_policy']
    write_json(out/'summary.json',result)
    if records:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,(ax,bx)=plt.subplots(1,2,figsize=(15,max(7,.33*len(records))))
        yy=np.arange(len(records));hit=np.array([r['warned_before'] for r in records]);miss=np.array([r['missed'] for r in records])
        ax.barh(yy,hit,color='#239b56',label='Warned before onset');ax.barh(yy,miss,left=hit,color='#d9d9d9',label='No earlier warning')
        ax.set_yticks(yy,[r['tag'] for r in records]);ax.invert_yaxis();ax.set_xlabel('Reference events');ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.2)
        for i,r in enumerate(records):ax.text(r['n_events']+.1,i,f'{r["warned_before"]}/{r["n_events"]}',va='center',fontsize=8)
        ax.set_xlim(0,max(r['n_events'] for r in records)+1.5)
        for i,r in enumerate(records):
            vv=[float(e['lead_seconds']) for e in covered if e['tag']==r['tag']]
            if vv:bx.scatter(vv,[i]*len(vv),s=22,color='#377eb8',alpha=.75)
        bx.set_yticks(yy,[]);bx.set_ylim(ax.get_ylim());bx.set_xlabel('First-warning lead time (s), covered events');bx.grid(axis='x',alpha=.2)
        fig.suptitle(f'Parameter DeepONet + Hopf: {success}/{count} reference events warned before onset')
        fig.tight_layout();fig.savefig(out/'batch_summary.png',dpi=180);fig.savefig(out/'batch_summary.pdf');plt.close(fig)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reference-root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--observation-aggregation',choices=['endpoint','block_mean'],default='endpoint')
    ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--gpus',default='0,1,2,3')
    ap.add_argument('--reuse-single',help='Existing matching parameter-model result (Pt 31_FD31_1)')
    ap.add_argument('--cycle-mode',action='store_true',help='Decoded-force cycle-gap risk; new weights required')
    ap.add_argument('--history-cycles',type=float,help='Network input duration only, in fixed reference cycles')
    ap.add_argument('--horizon-steps',type=int,default=30,help='Forecast steps at 0.05 s each')
    ap.add_argument('--alarm-threshold',type=float,default=.7)
    ap.add_argument('--confirm',type=int,default=1)
    ap.add_argument('--reference-rule',choices=['paper_2020','legacy'],default='paper_2020')
    ap.add_argument('--merge-gap-cycles',type=float,default=0.,help='Paper rule requires zero; nonzero is an explicit legacy experiment')
    ap.add_argument('--event-loss-weight',type=float,default=.2,
                    help='Auxiliary soft event-loss weight; zero disables event supervision')
    ap.add_argument('--constant-coefficients',action='store_true',
                    help='Hold a history-inferred coefficient triple constant during each forecast')
    ap.add_argument('--tags',help='Optional comma-separated subset of record tags for a smoke run')
    args=ap.parse_args();reference=Path(args.reference_root);out=Path(args.out)
    if args.horizon_steps<1 or args.confirm<1:raise ValueError('horizon-steps and confirm must be positive')
    from cycle_label_protocol import label_specification
    try:label_policy=label_specification(1.13,args.reference_rule,args.merge_gap_cycles)
    except ValueError as exc:ap.error(str(exc))
    if args.cycle_mode and args.reference_rule=='paper_2020' and args.history_cycles is None:args.history_cycles=1.
    if (out/'status.json').exists():raise RuntimeError('Output already has a status file; use a fresh directory to avoid overwriting a batch')
    out.mkdir(parents=True,exist_ok=True);(out/'records').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True);(out/'code').mkdir(exist_ok=True)
    source=Path(__file__).resolve().parent
    names=['generalized_hopf_deeponet.py','normal_warning_protocol.py','batch_parameter_hopf.py','generalized_hopf_README.md','history_window.py','cycle_label_protocol.py']
    if args.cycle_mode or args.merge_gap_cycles>0:names.append('cycle_transition.py')
    for name in names:
        shutil.copy2(source/name,out/'code'/name)
    manifest=json.loads((reference/'config/records.json').read_text())
    if args.tags:
        wanted=set(args.tags.split(','));manifest=[r for r in manifest if r['tag'] in wanted]
        if not manifest:ap.error('None of --tags matched the canonical record manifest')
    write_json(out/'input_manifest.json',manifest)
    tags=[e['tag'] for e in manifest];assert len(tags)==len(set(tags))
    for entry in manifest:
        if sha(reference/entry['record'])!=entry['record_sha256']:raise RuntimeError('Input hash mismatch: '+entry['tag'])
    import numpy as np
    from history_window import history_specification
    with np.load(reference/manifest[0]['record']) as data:
        history_spec=history_specification(data['t'],5,40,args.history_cycles,1.13)
    write_json(out/'protocol.json',{'epochs':args.epochs,'architecture':'parameters','observation_aggregation':args.observation_aggregation,'train_fraction':.65,'validation_within_train':.2,
        'history':history_spec['input_blocks'],'history_cycles':args.history_cycles,'network_history':history_spec,'horizon':args.horizon_steps,'simulation_horizon_seconds':args.horizon_steps*.05,'stride':5,'draws':128,'radius':None if args.cycle_mode else .3,'dwell':None if args.cycle_mode else 3,'alarm_threshold':args.alarm_threshold,'confirm':args.confirm,'seed':2026,
        'risk_rule':'cycle_gap' if args.cycle_mode else 'latent_disk','cycle':1.13,'merge_gap_cycles':args.merge_gap_cycles,'event_horizon':1.,
        'event_loss_weight':args.event_loss_weight,'constant_coefficients':args.constant_coefficients,
        'reference_rule':args.reference_rule,'label_policy':label_policy,
        'reference_root':str(reference),'record_count':len(manifest),'gpu_workers':args.gpus,'input_hashes_verified':True,
        'source_hashes':{p.name:sha(p) for p in (out/'code').iterdir() if p.is_file()}})
    status={'state':'running','records':{tag:{'status':'queued'} for tag in tags}};lock=threading.Lock();todo=queue.Queue()
    if args.reuse_single:
        if args.cycle_mode or args.history_cycles is not None:raise ValueError('Cycle mode and changed history duration require new weights')
        reuse=Path(args.reuse_single);m=json.loads((reuse/'metrics.json').read_text());cfg=m['configuration'];tag=cfg['tag']
        expected={'architecture':'parameters','epochs':args.epochs,'train_fraction':.65,'validation_within_train':.2,'normal_only':True,
                  'confirm':1,'alarm_threshold':.7,'history':40,'horizon':20,'draws':128,'stride':5,'radius':.3,'dwell':3,
                  'reference_rule':args.reference_rule,'merge_gap_cycles':args.merge_gap_cycles}
        if tag not in tags or any(cfg.get(k)!=v for k,v in expected.items()):raise RuntimeError('Reuse settings do not match')
        entry=next(e for e in manifest if e['tag']==tag)
        if Path(m['input_source']).resolve()!=(reference/entry['record']).resolve():raise RuntimeError('Reuse input source mismatch')
        dest=out/'records'/tag.replace(' ','_');dest.mkdir()
        for p in reuse.iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name)
        m['configuration']['out']=str(dest);m['batch_reused_from']=str(reuse);write_json(dest/'metrics.json',m)
        status['records'][tag]={'status':'complete','reused_from':str(reuse),'model_sha256':sha(dest/'model.pt'),'seconds':0}
        print('REUSED '+tag,flush=True)
    for entry in manifest:
        if status['records'][entry['tag']]['status']=='queued':todo.put(entry)
    write_json(out/'status.json',status)
    def worker(gpu):
        while True:
            try:entry=todo.get_nowait()
            except queue.Empty:return
            tag=entry['tag'];safe=tag.replace(' ','_');dest=out/'records'/safe;log=out/'logs'/(safe+'.log');start=time.monotonic()
            with lock:
                status['records'][tag]={'status':'running','gpu':gpu,'log':str(log)};write_json(out/'status.json',status)
                print('START GPU '+gpu+' '+tag,flush=True)
            cmd=[sys.executable,str(out/'code/generalized_hopf_deeponet.py'),'--data',str(reference/entry['record']),
                '--reference-root',str(reference),'--tag',tag,'--out',str(dest),'--architecture','parameters','--normal-only',
                '--train-fraction','0.65','--epochs',str(args.epochs),'--confirm',str(args.confirm),'--alarm-threshold',str(args.alarm_threshold),
                '--observation-aggregation',args.observation_aggregation,
                '--reference-rule',args.reference_rule,'--merge-gap-cycles',str(args.merge_gap_cycles)]
            cmd+=['--event-loss-weight',str(args.event_loss_weight)]
            if args.constant_coefficients:cmd+=['--constant-coefficients']
            if args.cycle_mode:cmd+=['--risk-rule','cycle_gap','--horizon',str(args.horizon_steps),'--event-horizon','1.0']
            if args.history_cycles is not None:cmd+=['--history-cycles',str(args.history_cycles)]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            try:
                with log.open('w') as f:
                    proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=out/'code')
                if proc.returncode:raise RuntimeError(f'Exit {proc.returncode}; see {log}')
                m=json.loads((dest/'metrics.json').read_text());assert m['warning_evaluation']['all_origins_non_fog']
                info={'status':'complete','gpu':gpu,'log':str(log),'seconds':round(time.monotonic()-start,2),
                      'model_sha256':sha(dest/'model.pt') if (dest/'model.pt').exists() else None,'n_events':m['warning_evaluation']['n_test_events'],
                      'warned_before':m['warning_evaluation']['events_warned_before']}
            except Exception as exc:info={'status':'failed','gpu':gpu,'log':str(log),'error':str(exc),'seconds':round(time.monotonic()-start,2)}
            with lock:
                status['records'][tag]=info;write_json(out/'status.json',status)
                print(info['status'].upper()+' '+tag+' '+str(info.get('warned_before','?'))+'/'+str(info.get('n_events','?')),flush=True)
            todo.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus.split(','))) as pool:
        futures=[pool.submit(worker,gpu) for gpu in args.gpus.split(',')]
        for future in futures:future.result()
    summary=collect_results(out,manifest,status)
    status['state']='complete' if not summary['failed_records'] else 'complete_with_failures';write_json(out/'status.json',status)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
