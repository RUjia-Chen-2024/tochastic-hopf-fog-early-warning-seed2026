"""Read the archived seed-2026 path arrays; no training or simulation."""
from pathlib import Path
import numpy as np,json,csv,os
run=Path(os.environ["FOG_REMOTE_RUN_ROOT"])
source=Path(os.environ["FOG_REMOTE_SOURCE_ROOT"])
ref=Path(os.environ.get("FOG_REMOTE_REFERENCE_ROOT", str(source.parent/"reference_project/data/records")))
result=[]
for folder in sorted((run/'results/records').iterdir()):
    name=folder.name
    paths=np.load(folder/'learned_seed_2026_force_paths.npy',mmap_mode='r')
    saved=np.load(source/'records'/name/'predictions.npz');truth=saved['truth'];times=saved['origin_time']
    horizons=saved['horizon_seconds'];raw=np.load(ref/(name+'.npz'));tt=raw['t']
    ids=np.rint((times-tt[0])/(tt[1]-tt[0])).astype(int)
    persistence=np.array([[np.mean(raw[k][i-4:i+1]) for k in ['x','y']] for i in ids])
    events=list(csv.DictReader((source/'records'/name/'reference_events.csv').open()))
    ons=np.array([float(r['onset']) for r in events])
    cross=np.any((ons[None,:,None]>times[:,None,None])&(ons[None,:,None]<=times[:,None,None]+horizons[None,None,:]+1e-9),axis=1)
    row=dict(record=name,count=0,absolute_error=0.,squared_error=0.,persist_absolute_error=0.,persist_squared_error=0.,
             interval_hits=0,cross_count=0,cross_absolute_error=0.,cross_persist_absolute_error=0.)
    for lo in range(0,len(times),32):
        hi=min(lo+32,len(times));endpoint=np.asarray(paths[lo:hi,:,4::5,:])
        mean=endpoint.mean(1);low,high=np.quantile(endpoint,[.05,.95],axis=1)
        y=truth[lo:hi];valid=np.isfinite(y)&np.isfinite(mean);error=mean-y;pe=persistence[lo:hi,None,:]-y
        row['count']+=int(valid.sum());row['absolute_error']+=float(np.abs(error[valid]).sum());row['squared_error']+=float((error[valid]**2).sum())
        row['persist_absolute_error']+=float(np.abs(pe[valid]).sum());row['persist_squared_error']+=float((pe[valid]**2).sum())
        row['interval_hits']+=int((valid&(y>=low)&(y<=high)).sum())
        c=valid&cross[lo:hi,:,None];row['cross_count']+=int(c.sum());row['cross_absolute_error']+=float(np.abs(error[c]).sum());row['cross_persist_absolute_error']+=float(np.abs(pe[c]).sum())
    result.append(row)
print(json.dumps(result))
