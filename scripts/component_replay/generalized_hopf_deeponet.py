"""Experimental DeepONet parameter operator + generalized Hopf SDE.

Input: whitespace-delimited time,x,y. Optional events CSV: onset,offset (seconds).
Risk is entrance into a chosen latent disk, not calibrated clinical FoG risk.
Events supervise risk only when supplied. Hyperparameters require validation.
Default architecture predicts future beta/frequency/sigma functions. The earlier
additive-drift pilot is available only with --architecture residual.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HopfDeepONet(nn.Module):
    def __init__(self, history, horizon, dt):
        super().__init__()
        self.horizon, self.dt = horizon, dt
        self.encoder = nn.Sequential(nn.Linear(history*2,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.initial = nn.Linear(64,2)
        self.param_head = nn.Linear(64,3)
        self.branch = nn.Linear(64,64)
        self.trunk = nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,64))
        self.decoder = nn.Linear(2,2)
        self.observation_log_std = nn.Parameter(torch.full((2,),-1.))
        self.register_buffer('tau',torch.arange(1,horizon+1).float()[:,None]/horizon)

    def forward(self,x,draws=1):
        h=self.encoder(x.flatten(1)); batch=len(x)
        z0=1.5*torch.tanh(self.initial(h))
        p=self.param_head(h)
        beta=-.98+.96*torch.sigmoid(p[:,0])  # restrict to bistability
        frequency=.1+2.9*torch.sigmoid(p[:,1])
        sigma=.01+.39*torch.sigmoid(p[:,2])
        coeff=self.branch(h).reshape(batch,2,32)
        basis=self.trunk(self.tau).reshape(self.horizon,2,32)
        # Operator learns a history-dependent drift correction, not a risk classifier.
        correction=torch.einsum('bcp,tcp->btc',coeff,basis)/np.sqrt(32)
        bh=beta[:,None,None]; om=2*np.pi*frequency[:,None,None]; sg=sigma[:,None,None]
        def drift(z):
            r2=z.square().sum(-1,keepdim=True)
            radial=bh+(1-bh)*r2-r2.square()
            rotation=torch.stack((-z[...,1],z[...,0]),-1)
            return radial*z+om*rotation
        zh=z0[:,None,:].expand(-1,draws,-1); zj=zh
        pure=[]; joint=[]
        substeps=max(1,int(np.ceil(self.dt/.01))); ds=self.dt/substeps
        for k in range(self.horizon):
            for _ in range(substeps):
                noise=torch.randn_like(zj)*np.sqrt(ds)*sg
                # Tamed Euler prevents polynomial drift blow-up; paired paths share noise.
                dh=drift(zh); dj=drift(zj)+correction[:,k,None,:]
                zh=zh+ds*dh/(1+ds*torch.linalg.vector_norm(dh,dim=-1,keepdim=True))+noise
                zj=zj+ds*dj/(1+ds*torch.linalg.vector_norm(dj,dim=-1,keepdim=True))+noise
            pure.append(zh); joint.append(zj)
        zh=torch.stack(pure,2); zj=torch.stack(joint,2)
        return {'joint':self.decoder(zj),'hopf':self.decoder(zh),'latent':zj,'pure_latent':zh,
                'reconstruction':self.decoder(z0),'correction':correction,
                'params':torch.stack((beta,frequency,sigma),-1)}


class ParameterHopfDeepONet(nn.Module):
    """DeepONet predicts parameter functions; all future states come from the SDE.

    Parameters are held constant within each output interval. There is no neural
    additive drift, direct future-state head or future initial-state adjustment.
    """
    def __init__(self,history,horizon,dt,observation_aggregation="endpoint",constant_coefficients=False):
        super().__init__()
        self.horizon,self.dt=horizon,dt
        self.observation_aggregation=observation_aggregation
        self.constant_coefficients=bool(constant_coefficients)
        self.encoder=nn.Sequential(nn.Linear(history*2,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.initial=nn.Linear(64,2)
        self.param_head=nn.Linear(64,3)
        if not self.constant_coefficients:
            self.branch=nn.Linear(64,3*32)
            self.trunk=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,3*32))
        self.decoder=nn.Linear(2,2)
        self.observation_log_std=nn.Parameter(torch.full((2,),-1.))
        self.register_buffer('tau',torch.arange(1,horizon+1).float()[:,None]/horizon)

    @staticmethod
    def physical_parameters(logits):
        beta=-.98+.96*torch.sigmoid(logits[...,0])
        frequency=.1+2.9*torch.sigmoid(logits[...,1])
        sigma=.01+.39*torch.sigmoid(logits[...,2])
        return torch.stack((beta,frequency,sigma),-1)

    @staticmethod
    def drift(z,beta,frequency):
        r2=z.square().sum(-1,keepdim=True)
        radial=beta+(1-beta)*r2-r2.square()
        rotation=torch.stack((-z[...,1],z[...,0]),-1)
        return radial*z+2*np.pi*frequency*rotation

    def forward(self,x,draws=1,return_fine=False):
        h=self.encoder(x.flatten(1));batch=len(x)
        z0=1.5*torch.tanh(self.initial(h))
        p0=self.param_head(h)
        if self.constant_coefficients:
            # One history-conditioned coefficient triple is held fixed for this
            # forecast. A new triple is inferred at the next rolling origin.
            curve=self.physical_parameters(p0)[:,None,:].expand(-1,self.horizon,-1)
        else:
            branch=self.branch(h).reshape(batch,3,32)
            trunk=self.trunk(self.tau).reshape(self.horizon,3,32)
            delta=torch.einsum('bcp,tcp->btc',branch,trunk)/np.sqrt(32)
            logits=p0[:,None,:]+self.tau[None,:,:]*delta
            curve=self.physical_parameters(logits)
        z=z0[:,None,:].expand(-1,draws,-1);states=[];fine=[]
        substeps=max(1,int(np.ceil(self.dt/.01)));ds=self.dt/substeps
        for k in range(self.horizon):
            beta=curve[:,k,0,None,None];frequency=curve[:,k,1,None,None];sigma=curve[:,k,2,None,None]
            for _ in range(substeps):
                drift=self.drift(z,beta,frequency)
                noise=torch.randn_like(z)*np.sqrt(ds)*sigma
                z=z+ds*drift/(1+ds*torch.linalg.vector_norm(drift,dim=-1,keepdim=True))+noise
                if return_fine or self.observation_aggregation=="block_mean":fine.append(z)
            states.append(z)
        latent=torch.stack(states,2)
        if self.constant_coefficients:
            smoothness=curve.new_zeros(())
        else:
            full_logits=torch.cat((p0[:,None,:],logits),1)
            smoothness=((full_logits[:,1:]-full_logits[:,:-1])*self.horizon).square().mean()
        result={'joint':self.decoder(latent),'latent':latent,'reconstruction':self.decoder(z0),
                'params':curve,'current_params':self.physical_parameters(p0),'parameter_smoothness':smoothness}
        if return_fine or self.observation_aggregation=="block_mean":
            fine_joint=self.decoder(torch.stack(fine,2))
            if self.observation_aggregation=="block_mean":
                result['joint']=fine_joint.reshape(batch,draws,self.horizon,substeps,2).mean(3)
            if return_fine:result.update(fine_joint=fine_joint,fine_dt=ds)
        return result


def make_model(args,dt):
    # Old checkpoints predate the architecture field and use the residual model.
    cls=ParameterHopfDeepONet if getattr(args,'architecture','residual')=='parameters' else HopfDeepONet
    if cls is ParameterHopfDeepONet:
        return cls(args.history,args.horizon,dt,getattr(args,'observation_aggregation','endpoint'),
                   getattr(args,'constant_coefficients',False))
    return cls(args.history,args.horizon,dt)


def entrance_risk(latent,radius,dwell,soft=False):
    r=torch.linalg.vector_norm(latent,dim=-1)
    occupancy=torch.sigmoid((radius-r)/.05) if soft else (r<radius).float()
    # Consecutive occupancy; soft minimum avoids an independence assumption.
    windows=occupancy.unfold(-1,dwell,1).amin(-1)
    return windows.amax(-1).mean(1)


def threshold_warnings(times,risks,threshold=.7,confirm=1,dt=.05):
    """Issue on first above-threshold point; re-arm after risk falls or a gap.

    confirm>1 is retained only for reproducing older experiments. A continuous
    above-threshold run produces one issuance rather than a new one at every sample.
    """
    streak=0;active=False;last_time=None;flags=[]
    for time,risk in zip(times,risks):
        if last_time is not None and time-last_time>1.5*dt:
            streak=0;active=False
        above=risk>threshold
        streak=streak+1 if above else 0
        if not above:active=False
        issued=streak>=confirm and not active
        if issued:active=True
        flags.append(int(issued));last_time=time
    return flags


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--events',help='Optional CSV with onset,offset in data time units (seconds)')
    ap.add_argument('--reference-root',help='Canonical FOG_state_history_paper input directory')
    ap.add_argument('--tag',default='Pt 31_FD31_1')
    ap.add_argument('--architecture',choices=['parameters','residual'],default='parameters',help='parameters: DeepONet parameter curves + Hopf solver; residual: earlier pilot')
    ap.add_argument('--observation-aggregation',choices=['endpoint','block_mean'],default='endpoint')
    ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--stride',type=int,default=5)
    ap.add_argument('--history',type=int,default=40); ap.add_argument('--horizon',type=int,default=20)
    ap.add_argument('--history-cycles',type=float,help='Network input duration in cycles; keeps label cycle and forecast step unchanged')
    ap.add_argument('--draws',type=int,default=128); ap.add_argument('--radius',type=float,default=.3)
    ap.add_argument('--dwell',type=int,default=3); ap.add_argument('--alarm-threshold',type=float,default=.7)
    ap.add_argument('--confirm',type=int,default=1,help='Default: issue immediately when risk exceeds threshold')
    ap.add_argument('--train-fraction',type=float,default=.65,help='Prefix reserved for training and validation')
    ap.add_argument('--validation-within-train',type=float,default=.2)
    ap.add_argument('--normal-only',action='store_true',help='Offline-reference screening of every test origin; point-marker evaluation')
    ap.add_argument('--risk-rule',choices=['latent_disk','cycle_gap'],default='latent_disk')
    ap.add_argument('--cycle',type=float,default=1.13)
    ap.add_argument('--reference-rule',choices=['paper_2020','legacy'],default='paper_2020',help='Paper threshold convention, or explicit historical 1.2T onset')
    ap.add_argument('--merge-gap-cycles',type=float,default=0.,help='Extra event merging; paper_2020 requires zero')
    ap.add_argument('--event-horizon',type=float,default=1.)
    ap.add_argument('--event-loss-weight',type=float,default=.2,
                    help='Weight of the auxiliary soft cycle-event objective; set to zero for the no-event-loss control')
    ap.add_argument('--constant-coefficients',action='store_true',
                    help='Ablation: infer one coefficient triple from history and hold it fixed over each rollout')
    args=ap.parse_args()
    if not 1<=args.dwell<=args.horizon: ap.error('dwell must be between 1 and horizon')
    if args.draws<2 or args.stride<1 or args.confirm<1: ap.error('invalid draws/stride/confirm')
    if not 0<args.train_fraction<1 or not 0<args.validation_within_train<1: ap.error('Invalid split fractions')
    if args.normal_only and not (args.events or args.reference_root): ap.error('normal-only requires reference labels')
    if args.risk_rule=='cycle_gap' and (not args.reference_root or args.architecture!='parameters'):
        ap.error('cycle_gap requires reference-root and parameters architecture')
    from cycle_label_protocol import label_specification,reference_intervals
    try:label_policy=label_specification(args.cycle,args.reference_rule,args.merge_gap_cycles)
    except ValueError as exc:ap.error(str(exc))
    if args.risk_rule=='cycle_gap' and args.reference_rule=='paper_2020' and args.history_cycles is None:
        args.history_cycles=1.
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.set_num_threads(2)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    reference_events=None
    if args.reference_root:
        if args.events: ap.error('Choose reference-root or events, not both')
        sys.path.insert(0,str(Path(args.reference_root).resolve()))
        from fog.coordinates import load_state_only_data_for_tag
        record=load_state_only_data_for_tag(args.tag)
        a=np.column_stack((record['t_orig'],record['x_orig'],record['y_orig']))
        if args.risk_rule=='cycle_gap':
            from cycle_transition import references
            unmerged,reference_events=references(a,Path(args.reference_root),args.cycle,args.merge_gap_cycles,args.reference_rule)
            with (out/'unmerged_reference_events.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['onset','offset']);w.writerows(unmerged)
        else:
            reference_events=reference_intervals(a,Path(args.reference_root),args.cycle,args.reference_rule)
            if args.merge_gap_cycles:
                from cycle_transition import merge_intervals
                reference_events=merge_intervals(reference_events,args.cycle,args.merge_gap_cycles)
        (out/'label_policy.json').write_text(json.dumps(label_policy,indent=2))
        with (out/'reference_events.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['onset','offset']); w.writerows(reference_events)
    else:
        a=np.loadtxt(args.data)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0):
        raise ValueError('Expected finite, increasing time,x,y data')
    # Block averages reduce aliasing compared with unfiltered decimation.
    raw=a.copy();count=len(a)//args.stride
    blocks=a[:count*args.stride].reshape(count,args.stride,3)
    a=blocks.mean(1); a[:,0]=blocks[:,-1,0]  # timestamp when complete block is available
    t=a[:,0]; v=a[:,1:]; dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=.01,atol=1e-5): raise ValueError('Resample irregular data first')
    b2=int(args.train_fraction*len(a)); b1=int((1-args.validation_within_train)*b2)
    mean=v[:b1].mean(0); scale=v[:b1].std(0).clip(1e-6)
    if args.risk_rule=='cycle_gap':
        from cycle_transition import fit_peak_settings,observed_contexts,soft_cycle_risk,hard_cycle_risk
        settings=fit_peak_settings(raw,b1*args.stride,args.cycle,args.reference_rule)
        if args.event_horizon>=settings['onset_delay']:ap.error('event horizon must be less than the onset delay')
        confirmation_lag=max(0.,settings['abnormal_gap']-settings['onset_delay'])
        if args.horizon*dt<args.event_horizon+confirmation_lag+1/settings['fs']-1e-9:
            ap.error('Cycle confirmation needs event horizon + confirmation lag + one raw sample of simulation')
        (out/'peak_calibration.json').write_text(json.dumps(settings,indent=2))
    from history_window import history_specification,network_histories,history_context_rows
    history_spec=history_specification(raw[:,0],args.stride,args.history,args.history_cycles,args.cycle)
    args.history=history_spec['input_blocks']
    if args.history_cycles is not None:
        (out/'network_history.json').write_text(json.dumps(history_spec,indent=2))
    x=((v-mean)/scale).astype('float32'); L,H=args.history,args.horizon
    events=reference_events if reference_events is not None else []
    supervised=bool(args.events or args.reference_root)
    if args.events:
        with open(args.events,encoding='utf-8-sig') as f:
            events=[(float(r['onset']),float(r['offset'])) for r in csv.DictReader(f)]
        if any(e<=s for s,e in events): raise ValueError('Event offsets must exceed onsets')
    def in_event(time): return any(s-1e-9<=time<=e+1e-9 for s,e in events)
    def dataset(lo,hi,test_partition=False):
        # Test uses already available history before the 65% boundary. No retraining.
        first=max(lo,L-1) if test_partition else lo+L-1
        stop=hi if test_partition and args.risk_rule=='cycle_gap' else hi-H
        ids=np.arange(first,stop,1 if test_partition else 2)
        if test_partition and args.normal_only:
            ids=np.array([i for i in ids if not in_event(t[i])],dtype=int)
        if not len(ids):
            if not (test_partition and args.risk_rule=='cycle_gap'):raise ValueError('Partition too short for history and horizon')
            return torch.empty(0,L,2),torch.empty(0,H,2),torch.empty(0),ids,np.empty((0,1)),np.empty(0)
        labels=[]
        for i in ids:
            active=in_event(t[i])
            end=t[i]+args.event_horizon if args.risk_rule=='cycle_gap' else t[i+H]
            labels.append(-1. if active else float(any(t[i]<s<=end for s,e in events)))
        targets=[]
        for i in ids:
            y=np.full((H,2),np.nan,dtype='float32');available=x[i+1:min(i+H+1,len(x))];y[:len(available)]=available;targets.append(y)
        result=(torch.tensor(network_histories(x,raw,ids,args.stride,mean,scale,history_spec)),torch.tensor(np.stack(targets)),torch.tensor(labels),ids)
        if args.risk_rule=='cycle_gap':
            context,age=observed_contexts(raw,ids,args.stride,settings)
            result+= (context,age)
        return result
    train,val,test=dataset(0,b1),dataset(b1,b2),dataset(b2,len(a),True)
    if args.history_cycles is not None:
        fields=['origin_time','history_start_time','history_end_time','duration_seconds','raw_start_index','raw_end_index','raw_samples','input_blocks','oldest_block_samples']
        with (out/'network_history_context.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(history_context_rows(raw,test[3],history_spec))
        np.savez_compressed(out/'network_inputs.npz',inputs=test[0].numpy(),origin_time=t[test[3]])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.risk_rule=='cycle_gap' and not len(test[0]):
        from cycle_transition import save_empty_result
        save_empty_result(out,args,t,v,train,val,events,b1,b2,mean,scale,record['state_file']);return
    model=make_model(args,dt).to(device); optim=torch.optim.AdamW(model.parameters(),lr=1e-3)
    tensors=list(train[:3])
    if args.risk_rule=='cycle_gap':tensors.extend([torch.tensor(train[4]),torch.tensor(train[5])])
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),batch_size=32,shuffle=True)
    obs_scale=torch.tensor(scale,dtype=torch.float32,device=device);obs_mean=torch.tensor(mean,dtype=torch.float32,device=device)
    def objective(b,y,labels,draws=4,contexts=None,ages=None):
        r=model(b,draws,return_fine=True) if args.risk_rule=='cycle_gap' else model(b,draws)
        logstd=model.observation_log_std.clamp(-3,1); variance=(2*logstd).exp()
        def nll(pred):
            # Monte Carlo mixture likelihood over full future trajectories.
            ll=-.5*(((pred-y[:,None]).square()/variance)+2*logstd+np.log(2*np.pi)).sum((-1,-2))
            return -(torch.logsumexp(ll,1)-np.log(draws)).mean()/(H*2)
        pred=nll(r['joint'])
        reconstruction=(r['reconstruction']-b[:,-1]).square().mean()
        if args.architecture=='parameters':
            hp=pred.new_zeros(());consistency=pred.new_zeros(())
            penalty=r['parameter_smoothness']
            loss=pred+.5*reconstruction+.01*penalty
        else:
            hp=nll(r['hopf']);consistency=(r['latent']-r['pure_latent']).square().mean()
            penalty=r['correction'].square().mean()
            loss=pred+.5*hp+.1*consistency+.5*reconstruction+.01*penalty
        event_loss=torch.tensor(0.,device=device)
        if supervised and (labels>=0).any():
            if args.risk_rule=='cycle_gap':
                risk,valid=soft_cycle_risk(r['fine_joint']*obs_scale+obs_mean,contexts,ages,settings,r['fine_dt'],args.event_horizon)
                keep=(labels>=0)&valid
            else:risk=entrance_risk(r['latent'],args.radius,args.dwell,True);keep=labels>=0
            if keep.any():
                risk=risk.clamp(1e-5,1-1e-5)
                event_loss=nn.functional.binary_cross_entropy(risk[keep],labels[keep]);loss=loss+args.event_loss_weight*event_loss
        return loss,[pred,hp,consistency,reconstruction,penalty,event_loss]
    best=float('inf'); records=[]; patience=0
    for ep in range(args.epochs):
        model.train(); total=0.
        for items in loader:
            b,y,labels=items[:3]
            b,y,labels=b.to(device),y.to(device),labels.to(device); optim.zero_grad()
            extra={'contexts':items[3].to(device),'ages':items[4].to(device)} if args.risk_rule=='cycle_gap' else {}
            loss,_=objective(b,y,labels,**extra); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optim.step(); total+=loss.item()*len(b)
        model.eval()
        with torch.no_grad(),torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(777)  # fixed validation noise, stable checkpoint selection
            extra={'contexts':torch.tensor(val[4],device=device),'ages':torch.tensor(val[5],device=device)} if args.risk_rule=='cycle_gap' else {}
            vl,parts=objective(val[0].to(device),val[1].to(device),val[2].to(device),8,**extra)
        records.append([ep+1,total/len(train[0]),vl.item(),*[p.item() for p in parts]])
        if vl.item()<best:
            best=vl.item(); patience=0
            torch.save({'state_dict':model.state_dict(),'args':vars(args),'dt':dt,'mean':mean,'scale':scale},out/'model.pt')
        else: patience+=1
        if ep%10==0: print(f'epoch {ep+1}: train={records[-1][1]:.4f} val={vl.item():.4f}',flush=True)
        if patience>=20: break
    saved=torch.load(out/'model.pt',map_location=device,weights_only=False); model.load_state_dict(saved['state_dict']); model.eval()
    outputs=[]; hopfs=[]; risks=[]; params=[];parameter_curves=[];cycle_info=[];counter=0
    with torch.no_grad():
        for b in test[0].split(16):
            r=model(b.to(device),args.draws,return_fine=True) if args.risk_rule=='cycle_gap' else model(b.to(device),args.draws)
            outputs.append(r['joint'].cpu().numpy())
            if args.architecture=='residual':hopfs.append(r['hopf'].mean(1).cpu().numpy())
            if args.risk_rule=='cycle_gap':
                probability,info=hard_cycle_risk((r['fine_joint']*obs_scale+obs_mean).cpu().numpy(),test[4][counter:counter+len(b)],test[5][counter:counter+len(b)],settings,r['fine_dt'],args.event_horizon)
                risks.extend(probability.tolist());cycle_info.extend(info);counter+=len(b)
            else:risks.extend(entrance_risk(r['latent'],args.radius,args.dwell).cpu().tolist())
            if args.architecture=='parameters':
                params.extend(r['current_params'].cpu().tolist());parameter_curves.append(r['params'].cpu().numpy())
            else:
                params.extend(r['params'].cpu().tolist())
                parameter_curves.append(r['params'][:,None,:].expand(-1,H,-1).cpu().numpy())
    samples=np.concatenate(outputs)*scale+mean; forecast=samples.mean(1)
    lower=np.quantile(samples,.05,axis=1); upper=np.quantile(samples,.95,axis=1)
    truth=test[1].numpy()*scale+mean;parameter_curves=np.concatenate(parameter_curves)
    baseline=np.repeat(test[0][:,-1:].numpy(),H,axis=1)*scale+mean
    metrics={}
    comparisons=[('parameter_hopf' if args.architecture=='parameters' else 'joint',forecast),('persistence',baseline)]
    if hopfs:comparisons.insert(1,('hopf',np.concatenate(hopfs)*scale+mean))
    for name,p in comparisons:
        metrics[name]={'mae_xy':np.nanmean(np.abs(p-truth),(0,1)).tolist(),'rmse_xy':np.sqrt(np.nanmean((p-truth)**2,(0,1))).tolist()}
    alarm=threshold_warnings(t[test[3]],risks,args.alarm_threshold,args.confirm,dt)
    with (out/'risk.csv').open('w',newline='') as f:
        risk_name='transition_probability' if args.risk_rule=='cycle_gap' else 'latent_entry_probability'
        writer=csv.writer(f); writer.writerow(['origin_time',risk_name,'experimental_trigger','beta','frequency_hz','sigma','future_onset_label','normal_origin'])
        for i,idx in enumerate(test[3]): writer.writerow([t[idx],risks[i],alarm[i],*params[i],float(test[2][i]) if supervised else '',int(not in_event(t[idx]))])
    if args.risk_rule=='cycle_gap':
        with (out/'cycle_prediction_context.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['origin_time','peak_age','candidate_onset_time','confirmation_deadline_time','status'])
            for idx,row in zip(test[3],cycle_info):w.writerow([t[idx],row['peak_age'],t[idx]+row['candidate_onset_offset'],t[idx]+row['confirmation_deadline_offset'],row['status']])
    with (out/'parameter_forecasts.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['origin_time','horizon_sec','forecast_time','beta','frequency_hz','sigma','instantaneous_unstable_radius'])
        for i,idx in enumerate(test[3]):
            for k,(beta,freq,sigma) in enumerate(parameter_curves[i]):
                writer.writerow([t[idx],(k+1)*dt,t[idx]+(k+1)*dt,beta,freq,sigma,np.sqrt(-beta)])
    summary={'configuration':vars(args),'device':str(device),'dt':dt,'windows':[len(s[0]) for s in [train,val,test]],'best_epoch':int(np.argmin(np.array(records)[:,2])+1),
             'metrics':metrics,'experimental_triggers':sum(alarm),'risk_range':[float(np.nanmin(risks)),float(np.nanmax(risks))] if np.isfinite(risks).any() else [None,None],
             'interpretation':'Latent disk entrance probability, NOT calibrated FoG probability. No bifurcation identification established.',
             'events_used':supervised,'reference_event_count':len(events),'reference_type':'cycle-defined algorithmic gait disruption' if args.reference_root else 'external CSV' if args.events else 'none',
             'input_source':record['state_file'] if args.reference_root else args.data,
             'intervals':'5-95% latent process samples; excludes observation noise and parameter uncertainty',
             'split':{'fit_fraction':b1/len(a),'validation_end_fraction':b2/len(a),'test_fraction':1-b2/len(a),
                      'fit_end_time':float(t[b1-1]),'validation_end_time':float(t[b2-1]),'test_boundary':float(t[b2]),
                      'first_test_origin':float(t[test[3][0]]),'normal_only':args.normal_only,'scaling':'fit prefix only'},
             'epochs_run':len(records)}
    summary['model']='DeepONet parameter functions -> generalized Hopf SDE -> linear observation map' if args.architecture=='parameters' else 'residual pilot'
    summary['parameter_names']=['beta','frequency_hz','sigma']
    summary['parameter_ranges']={'beta':[-.98,-.02],'frequency_hz':[.1,3.0],'sigma':[.01,.4]}
    summary['parameter_forecast_shape']=list(parameter_curves.shape)
    if args.risk_rule=='cycle_gap':
        summary['interpretation']='Probability of decoded force cycle-gap onset within event horizon; fixed fit-prefix peak calibration; unknown censored contexts never count as missing steps.'
        summary['risk_rule']='cycle_gap';summary['reference_type']='cycle-defined reference; '+('no extra event merging' if args.merge_gap_cycles==0 else 'explicit legacy event merging')
        summary['cycle_evaluable_origins']=int(np.isfinite(risks).sum());summary['reference_merge_gap_seconds']=args.merge_gap_cycles*args.cycle
        summary['unmerged_reference_count']=len(unmerged)
    if args.reference_root:summary['label_policy']=label_policy
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    np.savetxt(out/'training.csv',records,delimiter=',',header='epoch,train,val,nll_forecast,nll_auxiliary_hopf,consistency,reconstruction,regularization,event')
    np.savez_compressed(out/'predictions.npz',truth=truth,mean=forecast,q05=lower,q95=upper,origin_time=t[test[3]],risk=risks,current_parameters=params,parameter_curves=parameter_curves,horizon_seconds=np.arange(1,H+1)*dt)
    if args.normal_only:
        from normal_warning_protocol import evaluate_and_plot
        summary['warning_evaluation']=evaluate_and_plot(out,t,v,x,train,model,device,mean,scale,events,b2,args,dt)
        (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    else:
        fig,axs=plt.subplots(3,1,figsize=(12,9)); future=t[test[3]]+H*dt
        for c in range(2):
            axs[c].plot(future,truth[:,-1,c],label='observed'); axs[c].plot(future,forecast[:,-1,c],label='joint mean')
            axs[c].fill_between(future,lower[:,-1,c],upper[:,-1,c],alpha=.2,label='process 5-95%'); axs[c].set_ylabel('xy'[c]); axs[c].legend()
        axs[2].plot(t[test[3]],risks,label='latent disk entry risk'); axs[2].axhline(args.alarm_threshold,ls='--',color='red',label='experimental threshold')
        axs[2].set(xlabel='Time (s)',ylabel='Probability',ylim=(-.02,1.02)); axs[2].legend()
        fig.suptitle('Generalized Hopf + DeepONet pilot (risk uncalibrated)'); fig.tight_layout(); fig.savefig(out/'results.png',dpi=150); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
