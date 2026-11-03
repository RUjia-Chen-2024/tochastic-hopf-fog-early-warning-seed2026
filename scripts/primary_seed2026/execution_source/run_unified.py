"""Same frozen 5-s paths, cumulative first-onset risks at 1,2,3,5 seconds."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
import random
import hashlib
import concurrent.futures
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'noise_intervention_5s_endpoint_20261027'))
from run_intervention import simulate, rows, write_csv, dump, sha, SEEDS
from multihorizon_events import first_new_onsets, cumulative_risks, HORIZONS, synthetic_checks


def worker(task):
    source, reference, output, records, gpu = task
    source,reference,output=map(Path,[source,reference,output])
    sys.path.insert(0,str(source/'code'))
    from generalized_hopf_deeponet import make_model,threshold_warnings
    from normal_warning_protocol import assess_warnings
    from cycle_transition import observed_contexts,hard_cycle_risk
    torch.set_num_threads(1);device=torch.device(f'cuda:{gpu}')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for index,name in records:
        dest=output/'records'/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'COMPLETE.json').exists():continue
        src=source/'records'/name;before=sha(src/'model.pt')
        ck=torch.load(src/'model.pt',map_location='cpu',weights_only=False)
        args=SimpleNamespace(**ck['args']);assert args.horizon==100 and args.observation_aggregation=='endpoint'
        model=make_model(args,ck['dt']).to(device).eval();model.load_state_dict(ck['state_dict'])
        saved=np.load(src/'network_inputs.npz');times=saved['origin_time']
        inputs=torch.tensor(saved['inputs'],dtype=torch.float32,device=device)
        mean=torch.tensor(ck['mean'],dtype=torch.float32,device=device)
        scale=torch.tensor(ck['scale'],dtype=torch.float32,device=device)
        settings=json.loads((src/'peak_calibration.json').read_text())
        with np.load(reference/'data/records'/(name+'.npz')) as d:
            raw=np.column_stack([d['t'],d['x'],d['y']])
        raw_ids=np.rint((times-raw[0,0])*settings['fs']).astype(int)
        ids=(raw_ids+1)//args.stride-1
        contexts,ages=observed_contexts(raw,ids,args.stride,settings)
        events=[(float(r['onset']),float(r['offset'])) for r in rows(src/'reference_events.csv')]
        boundary=json.loads((src/'metrics.json').read_text())['split']['test_boundary']
        result=[];event_results=[];alarm_results=[];audits=[]
        variants=[('zero_noise',0.,2026),('learned_seed_2026',1.,2026)]
        for variant,multiplier,seed in variants:
            random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
            first_parts=[];risk_parts=[];rng_batches=[]
            draws=1 if multiplier==0 else 128
            path_file=dest/(variant+'_force_paths.npy')
            saved_paths=np.lib.format.open_memmap(path_file,mode='w+',dtype='float32',shape=(len(inputs),draws,500,2))
            for lo in range(0,len(inputs),32):
                hi=min(lo+32,len(inputs))
                rng_batches.append(dict(first=lo,stop=hi,cpu_state=torch.get_rng_state().clone(),
                                        cuda_state=torch.cuda.get_rng_state(device).clone()))
                fine,_,_,ds=simulate(model,inputs[lo:hi],draws,multiplier)
                force=(fine*scale+mean).cpu().numpy()
                saved_paths[lo:hi]=force
                first=first_new_onsets(force,contexts[lo:hi],ages[lo:hi],settings,ds)
                risks=cumulative_risks(first)
                legacy,_=hard_cycle_risk(force,contexts[lo:hi],ages[lo:hi],settings,ds,1.)
                assert np.allclose(risks[:,0],legacy,atol=0,rtol=0,equal_nan=True)
                assert np.all(np.diff(risks,axis=1)>=-1e-12)
                first_parts.append(first);risk_parts.append(risks)
            first=np.concatenate(first_parts);risk=np.concatenate(risk_parts)
            saved_paths.flush();del saved_paths
            state_file=dest/(variant+'_rng_stream.pt')
            torch.save(dict(seed=2026,record_seed=2026,record_offset=0,device=str(device),
                       batch_size=32,draws=draws,batches=rng_batches,
                       numpy_initial_state=np.random.get_state(),python_initial_state=random.getstate()),state_file)
            # Restore an actual saved per-batch stream and reproduce complete decoded paths.
            after_cpu=torch.get_rng_state();after_cuda=torch.cuda.get_rng_state(device)
            chosen=rng_batches[-1]
            torch.set_rng_state(chosen['cpu_state']);torch.cuda.set_rng_state(chosen['cuda_state'],device)
            check,_,_,_=simulate(model,inputs[chosen['first']:chosen['stop']],draws,multiplier)
            replay=(check*scale+mean).cpu().numpy()
            stored=np.load(path_file,mmap_mode='r')[chosen['first']:chosen['stop']]
            assert np.array_equal(replay,stored), 'Saved RNG path replay mismatch'
            torch.set_rng_state(after_cpu);torch.cuda.set_rng_state(after_cuda,device)
            digest=hashlib.sha256()
            with path_file.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
            path_hash=digest.hexdigest()
            np.savez_compressed(dest/(variant+'_first_onsets.npz'),origin_time=times,first_onset_offsets=first,risks=risk,horizons=HORIZONS)
            audits.append(dict(variant=variant,seed=2026,record_offset=0,nested_risks=True,
                          one_second_matches_legacy_definition=True,saved_rng_replay_exact=True,
                          force_paths_file=path_file.name,force_paths_sha256=path_hash,force_paths_bytes=path_file.stat().st_size,
                          rng_file=state_file.name,rng_sha256=sha(state_file)))
            origin_data=[dict(record=name,variant=variant,origin_time=float(t),raw_end=float(raw[-1,0]),
                              peak_age=float(age),test_boundary=boundary) for t,age in zip(times,ages)]
            for k,horizon in enumerate(HORIZONS):
                label=np.array([int(any(1e-9<s-t<=horizon+1e-9 for s,e in events)) for t in times])
                flags=np.array(threshold_warnings(times,risk[:,k],.8,2,float(ck['dt'])))
                _,alarms,er=assess_warnings(times,flags,events,boundary)
                for event in er:
                    previous=event['previous_reference_end'];previous=-np.inf if previous is None else previous
                    hits=[a['warning_time'] for a in alarms if a['warning_time']>previous+1e-9 and
                          1e-9<event['reference_onset']-a['warning_time']<=horizon+1e-9]
                    near=[a for a in hits if event['reference_onset']-a<=1+1e-9]
                    event_results.append(dict(record=name,variant=variant,horizon=horizon,**event,
                        covered_horizon=int(bool(hits)),lead_horizon=event['reference_onset']-min(hits) if hits else None,
                        covered_common_1s=int(bool(near))))
                for alarm in alarms:
                    lead=alarm['lead_seconds']
                    category=('matched' if lead is not None and 1e-9<lead<=horizon+1e-9 else
                              'unmatched' if alarm['warning_time']+horizon<=raw[-1,0]+1e-9 else 'tail_censored')
                    alarm_results.append(dict(record=name,variant=variant,horizon=horizon,match=category,**alarm))
                for j,row in enumerate(origin_data):
                    row[f'risk_{int(horizon)}s']=float(risk[j,k]);row[f'label_{int(horizon)}s']=int(label[j])
                    row[f'alarm_{int(horizon)}s']=int(flags[j]);row[f'complete_{int(horizon)}s']=int(times[j]+horizon<=raw[-1,0]+1e-9)
                print(name,variant,f'H={horizon:g}',f'alarms={int(flags.sum())}',flush=True)
            result+=origin_data
        write_csv(dest/'origins.csv',result);write_csv(dest/'events.csv',event_results);write_csv(dest/'alarms.csv',alarm_results)
        assert sha(src/'model.pt')==before
        dump(dest/'COMPLETE.json',dict(record=name,checkpoint_sha256=before,checkpoint_unchanged=True,
                                       origins=len(times),variants=audits))
    return [n for _,n in records]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);source=Path(a.source)
    hash_file=Path(__file__).with_name('source_hashes.json')
    if not hash_file.exists():
        hash_file=Path(__file__).resolve().parent.parent/'noise_intervention_5s_endpoint_20261027/source_hashes.json'
    expected=json.loads(hash_file.read_text())
    for file,expected_hash in expected.items():assert sha(source/file)==expected_hash,file
    synthetic_checks()
    names=[r['tag'].replace(' ','_') for r in json.loads((source/'input_manifest.json').read_text())]
    dump(out/'protocol.json',dict(model='same frozen 5-s endpoint',forecast_duration_s=5,horizons_s=HORIZONS.tolist(),
         threshold=.8,confirm=2,draws=128,seed=2026,record_seed=2026,record_offset=0,batch_size=32,history_seconds=1.13,
         risk='CDF of first NEW cycle-gap onset within each horizon',
         ongoing_context='Do not count an already overdue gap; wait for predicted recovery peak before a new gap.',
         detector='Same observed context, scaling, SciPy find_peaks and minimum-distance settings as primary',
         compatibility='Primary evaluation is the SAME 1-s column of these multihorizon paths; also checked against legacy 1-s event function',
         event_scoring='Both horizon-specific matching and common 1-s matching are reported',
         probability_scoring='Full per-horizon follow-up plus common full-5-s follow-up',
         random_stream='Reset seed 2026 for every record and variant; no record offset; save CPU/CUDA state before every batch and all force paths',
         deterministic_algorithms=True,cublas_workspace_config=':4096:8',tf32=False,
         environment=dict(torch=torch.__version__,cuda=torch.version.cuda),
         training='Frozen weights: original training seed 2026 and historical validation-noise seed 777 retained; no retraining; only evaluation stream standardized'))
    tasks=[(a.source,a.reference,a.out,[(i,n) for i,n in enumerate(names) if i%4==g],g) for g in range(4)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for done in pool.map(worker,tasks):print('Completed',done,flush=True)
    for filename in ['origins.csv','events.csv','alarms.csv']:
        allrows=[]
        for d in sorted((out/'records').iterdir()):allrows+=rows(d/filename)
        write_csv(out/filename,allrows)
    dump(out/'audit.json',[json.loads((d/'COMPLETE.json').read_text()) for d in sorted((out/'records').iterdir())])
    print('ALL 21 RECORDS COMPLETE',flush=True)


if __name__=='__main__':main()
