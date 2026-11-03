"""Train a direct-history-to-2-s-event-risk MLP for each canonical record."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

from history_window import history_specification, network_histories
from cycle_transition import observed_contexts


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_rows(path, data, fields=None):
    if fields is None:
        fields=list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)


def load_raw(path):
    with np.load(path) as d:
        return np.column_stack([d['t'], d['x'], d['y']]).astype(float)


def event_labels(ids, times, events, horizon):
    labels=[]
    for i in ids:
        t=float(times[i])
        active=any(s-1e-9 <= t <= e+1e-9 for s,e in events)
        labels.append(-1. if active else float(any(t+1e-9 < s <= t+horizon+1e-9 for s,e in events)))
    return np.asarray(labels, dtype=np.float32)


def feature_matrix(raw, ids, stride, mean, scale, spec, settings, cycle):
    nblocks=len(raw)//stride
    blocks=raw[:nblocks*stride].reshape(nblocks,stride,3)
    averaged=blocks.mean(1); averaged[:,0]=blocks[:,-1,0]
    t=averaged[:,0]; v=averaged[:,1:]
    x=((v-mean)/scale).astype('float32')
    hist=network_histories(x,raw,ids,stride,mean,scale,spec).reshape(len(ids),-1)
    context, age=observed_contexts(raw,ids,stride,settings)
    finite=np.isfinite(age).astype('float32')
    age_scaled=np.clip(np.nan_to_num(age,nan=0.)/cycle,0.,20.).astype('float32')[:,None]
    return np.concatenate([hist,context.astype('float32'),age_scaled,finite[:,None]],axis=1),t


class DirectRiskMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,128),nn.ReLU(),nn.Dropout(.15),
                               nn.Linear(128,64),nn.ReLU(),nn.Dropout(.10),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)


def fit_one(tag, reference, base, out, horizon, seed):
    dest=out/'records'/tag.replace(' ','_'); dest.mkdir(parents=True,exist_ok=True)
    base_record=base/'records'/tag.replace(' ','_')
    meta=json.loads((base_record/'metrics.json').read_text(encoding='utf-8'))
    ck=torch.load(base_record/'model.pt',map_location='cpu',weights_only=False)
    args=meta['configuration']; stride=int(args['stride']); H=int(args['horizon'])
    raw=load_raw(reference/'data'/'records'/(tag.replace(' ','_')+'.npz'))
    bcount=len(raw)//stride; blocks=raw[:bcount*stride].reshape(bcount,stride,3)
    averaged=blocks.mean(1); averaged[:,0]=blocks[:,-1,0]; times=averaged[:,0]
    b2=int(args['train_fraction']*len(times)); b1=int((1-args['validation_within_train'])*b2)
    spec=history_specification(raw[:,0],stride,int(args.get('history',40)),float(args.get('history_cycles',1.0)),float(args.get('cycle',1.13)))
    settings=json.loads((base_record/'peak_calibration.json').read_text(encoding='utf-8'))
    # Preserve the exact reference intervals produced by the primary 1T fit.
    events=[(float(r['onset']),float(r['offset'])) for r in rows(base_record/'reference_events.csv')]
    train_ids=np.arange(spec['input_blocks']-1,b1-H,2,dtype=int)
    val_ids=np.arange(b1+spec['input_blocks']-1,b2-H,2,dtype=int)
    saved=np.load(base_record/'network_inputs.npz')
    test_times=saved['origin_time'].astype(float)
    test_ids=np.searchsorted(times,test_times)
    if not np.allclose(times[test_ids],test_times,atol=1e-8,rtol=0): raise ValueError(f'{tag}: test time alignment failed')
    mean=np.asarray(ck['mean'],dtype=float); scale=np.asarray(ck['scale'],dtype=float)
    ids=np.concatenate([train_ids,val_ids,test_ids])
    feats,_=feature_matrix(raw,ids,stride,mean,scale,spec,settings,float(args.get('cycle',1.13)))
    ntr,nval,nte=len(train_ids),len(val_ids),len(test_ids)
    xtr,xval,xtest=feats[:ntr],feats[ntr:ntr+nval],feats[ntr+nval:]
    ytr=event_labels(train_ids,times,events,horizon); yval=event_labels(val_ids,times,events,horizon); ytest=event_labels(test_ids,times,events,horizon)
    keep_tr=ytr>=0; keep_val=yval>=0
    if keep_tr.sum()==0 or keep_val.sum()==0: raise ValueError(f'{tag}: no labelled train/validation examples')
    fmean=xtr[keep_tr].mean(0); fscale=xtr[keep_tr].std(0).clip(1e-5)
    xtr=((xtr-fmean)/fscale).astype('float32'); xval=((xval-fmean)/fscale).astype('float32'); xtest=((xtest-fmean)/fscale).astype('float32')
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=DirectRiskMLP(xtr.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    tx=torch.tensor(xtr[keep_tr],device=device); ty=torch.tensor(ytr[keep_tr],device=device)
    vx=torch.tensor(xval[keep_val],device=device); vy=torch.tensor(yval[keep_val],device=device)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=64,shuffle=True)
    loss_fn=nn.BCEWithLogitsLoss(); best=float('inf'); best_epoch=0; best_state=None; patience=0; log=[]
    for epoch in range(1,101):
        model.train(); running=0.
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); running+=float(loss.item())*len(xb)
        model.eval()
        with torch.no_grad(): val_loss=float(loss_fn(model(vx),vy).item())
        log.append(dict(epoch=epoch,train_bce=running/len(tx),validation_bce=val_loss))
        if val_loss < best-1e-6:
            best=val_loss; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else:
            patience+=1
            if patience>=20: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): risk=torch.sigmoid(model(torch.tensor(xtest,device=device))).cpu().numpy()
    records=[]
    for t,p,y in zip(test_times,risk,ytest):
        records.append(dict(origin_time=float(t),direct_risk=float(p),future_onset_label=int(y),normal_origin=1))
    write_rows(dest/'risk.csv',records,['origin_time','direct_risk','future_onset_label','normal_origin'])
    write_rows(dest/'training.csv',log,['epoch','train_bce','validation_bce'])
    torch.save(dict(state_dict=model.state_dict(),feature_mean=fmean,feature_scale=fscale,
                    n_features=xtr.shape[1],seed=seed,best_epoch=best_epoch),dest/'model.pt')
    (dest/'metrics.json').write_text(json.dumps(dict(tag=tag,method='direct-risk-MLP',history_cycles=spec['history_cycles'],
        history_seconds=spec['duration_seconds'],input_blocks=spec['input_blocks'],event_horizon_seconds=horizon,
        target='reference onset in (origin, origin + 2 s], only reference-normal origins',
        inputs=dict(network_history_dimensions=int(spec['input_blocks']*2),peak_context_dimensions=int(settings['cycle']*2*settings['fs'])+1,
                    peak_age_and_valid_flag=2,uses_same_peak_context=True),train_examples=int(keep_tr.sum()),
        validation_examples=int(keep_val.sum()),train_positive_fraction=float(ytr[keep_tr].mean()),
        validation_positive_fraction=float(yval[keep_val].mean()),best_epoch=best_epoch,
        best_validation_bce=best,device=str(device),seed=seed,training_example_stride='every 0.10 s; same as DeepONet'),indent=2),encoding='utf-8')
    return dict(tag=tag,records=len(records),train_examples=int(keep_tr.sum()),validation_examples=int(keep_val.sum()),best_epoch=best_epoch,validation_bce=best)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference-root',required=True); ap.add_argument('--base-run',required=True); ap.add_argument('--out',required=True); ap.add_argument('--event-horizon',type=float,default=2.); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--tags',help='Optional comma-separated subset of record tags for a smoke run')
    a=ap.parse_args(); reference=Path(a.reference_root); base=Path(a.base_run); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base/'input_manifest.json').read_text(encoding='utf-8'))
    if a.tags:
        wanted=set(a.tags.split(','));manifest=[r for r in manifest if r['tag'] in wanted]
        if not manifest:ap.error('None of --tags matched the base manifest')
    summaries=[fit_one(r['tag'],reference,base,out,a.event_horizon,a.seed) for r in manifest]
    (out/'batch_training_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__': main()
