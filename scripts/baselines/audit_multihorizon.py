"""Audit saved path events and evaluate horizon-matched baselines without neural retraining."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
INPUTS=ROOT/'analysis/two_second_primary/inputs'
RUN=INPUTS/'unified_seed2026_20261027' if INPUTS.exists() else WORK/'unified_seed2026_20261027'
CACHE=INPUTS/'baseline_cache' if INPUTS.exists() else WORK/'peak_age_baseline_20261026/results'
OUT=ROOT/'analysis/two_second_primary';OUT.mkdir(parents=True,exist_ok=True)
HS=[1,2,3,5];D=1.695

def flags(t,p):
    out=[];streak=0;active=False;last=None
    for time,risk in zip(t,p):
        if last is not None and time-last>.075+1e-9:streak=0;active=False
        above=np.isfinite(risk) and risk>.8
        streak=streak+1 if above else 0
        if not above:active=False
        fire=bool(streak>=2 and not active)
        if fire:active=True
        out.append(fire);last=time
    return np.array(out)

def labels(t,onsets,h):
    diff=onsets[None,:]-np.asarray(t)[:,None]
    return ((diff>1e-9)&(diff<=h+1e-9)).any(axis=1).astype(int)

def survival_at(model,t):
    v=np.asarray(t,float);ix=np.searchsorted(model['times'],np.round(v,6)+1e-9,side='right')-1
    return np.r_[1.,model['survival']][ix+1]

def survival_risks(age,model,h,renewal):
    a=np.round(np.asarray(age),6);sa=survival_at(model,a);sd=survival_at(model,[D])[0]
    gate=(D-a>1e-9)&(D-a<=h+1e-9)
    risk=np.where(gate,np.divide(sd,sa,out=np.full_like(sa,.5),where=sa>1e-12),0.)
    if renewal:
        grid=.005;times=np.asarray(model['times']);mass=-np.diff(np.r_[1.,model['survival']])
        ticks=np.rint(times/grid).astype(int);deadline=round(D/grid);n=round(h/grid)
        # Probability of a first abnormal gap after a newly observed renewal.
        fresh=np.zeros(n+1)
        short=(times<=D+1e-9)&(mass>0)&(ticks>0)
        for u in range(n+1):
            k=short&(ticks<=u)
            fresh[u]=(sd if u>=deadline else 0.)+np.sum(mass[k]*fresh[u-ticks[k]])
        for i,ai in enumerate(a):
            if sa[i]<=1e-12:continue
            allowed=(times>ai+1e-9)&(times-ai<=h+1e-9)&(mass>0)
            # If the current gap is still normal, an accepted peak must arrive
            # by its deadline to restart without an earlier abnormal onset.
            if ai<D-1e-9:allowed&=times<=D+1e-9
            ix=np.rint((h-(times[allowed]-ai))/grid).astype(int)
            risk[i]+=np.sum(mass[allowed]*fresh[np.clip(ix,0,n)])/sa[i]
    risk[~np.isfinite(a)]=np.nan
    assert np.nanmax(risk)<=1+1e-10
    return np.clip(risk,0,1)

def fit_logistic(train,val,test,h,onsets):
    y=labels(train.origin_time,onsets,h);v=labels(val.origin_time,onsets,h)
    constant=(y.sum()+.5)/(len(y)+1)
    if len(y)<2 or len(np.unique(y))<2:
        return np.full(len(test),constant),{'constant_fallback':float(constant),'fit_origins':len(y)}
    trials=[]
    for degree in [1,3]:
        design=lambda a:np.column_stack([(np.asarray(a)/D)**j for j in range(1,degree+1)])
        scaler=StandardScaler().fit(design(train.age))
        for c in [.01,.1,1.,10.,100.]:
            fit=LogisticRegression(C=c,max_iter=2000,random_state=2026).fit(scaler.transform(design(train.age)),y)
            p=fit.predict_proba(scaler.transform(design(val.age)))[:,1]
            score=np.mean((p-v)**2) if len(val) else float('inf')
            trials.append((score,degree,c,fit,scaler))
    score,degree,c,fit,scaler=min(trials,key=lambda x:x[:3])
    x=np.column_stack([(np.asarray(test)/D)**j for j in range(1,degree+1)])
    spec={'degree':degree,'C':c,'validation_brier':float(score),'fit_origins':len(y),
          'validation_origins':len(v),'intercept':fit.intercept_.tolist(),'coefficient':fit.coef_.tolist(),
          'feature_mean':scaler.mean_.tolist(),'feature_scale':scaler.scale_.tolist()}
    return fit.predict_proba(scaler.transform(x))[:,1],spec

def evaluate(rec,method,h,t,p,ref,end,boundary):
    f=flags(t,p);alarm=t[f];onset=ref.onset.to_numpy();y=labels(t,onset,h)
    complete=t+h<=end+1e-9;scored=complete&np.isfinite(p);common=(t+5<=end+1e-9)&np.isfinite(p)
    es=[]
    for i,row in enumerate(ref.itertuples()):
        if row.onset<boundary-1e-9:continue
        prior=ref.iloc[i-1].offset if i else -np.inf
        hit=alarm[(alarm>prior+1e-9)&(alarm<row.onset-1e-9)&(row.onset-alarm<=h+1e-9)]
        anyhit=alarm[(alarm>prior+1e-9)&(alarm<row.onset-1e-9)]
        es.append(dict(record=rec,method=method,horizon=h,onset=row.onset,end=row.offset,previous_end=prior,
            covered=int(len(hit)>0),lead=row.onset-hit[0] if len(hit) else None,
            first_alarm=float(hit[0]) if len(hit) else None,covered_1s=int(np.any(row.onset-hit<=1+1e-9)),
            covered_any=int(len(anyhit)>0),lead_any=row.onset-anyhit[0] if len(anyhit) else None,
            eligible_origins=int(np.sum((t>prior+1e-9)&(t<row.onset-1e-9)&(row.onset-t<=h+1e-9)))))
    ar=[]
    for u in alarm:
        future=onset[onset>u+1e-9];lead=future[0]-u if len(future) else np.inf
        category='matched' if lead<=h+1e-9 else 'unmatched' if u+h<=end+1e-9 else 'tail_censored'
        ar.append(dict(record=rec,method=method,horizon=h,time=u,category=category))
    row=dict(record=rec,participant=rec.split('_')[1],method=method,horizon=h,events=len(es),covered=sum(e['covered'] for e in es),
             alarms=len(ar),matched=sum(a['category']=='matched' for a in ar),unmatched=sum(a['category']=='unmatched' for a in ar),
             tail_censored=sum(a['category']=='tail_censored' for a in ar),normal_origins=len(t),scored=int(scored.sum()),
             positives=int(y[scored].sum()),brier_sum=float(np.sum((p[scored]-y[scored])**2)),
             common_n=int(common.sum()),common_brier_sum=float(np.sum((p[common]-y[common])**2)))
    origins=pd.DataFrame(dict(record=rec,method=method,horizon=h,time=t,risk=p,label=y,complete=complete.astype(int),
                             common_complete=common.astype(int),alarm=f.astype(int)))
    return row,es,ar,origins

def main():
    data=pd.read_csv(RUN/'results/multihorizon_risks.csv');old=pd.read_csv(RUN/'results/multihorizon_evaluation.csv').set_index('horizon_s')
    zero=pd.read_csv(RUN/'results/origins.csv');zero=zero[zero.variant=='zero_noise']
    records=[];events=[];alarms=[];origins=[];specs={};audit=[];hashes={}
    def load(path):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    for rec,group in data.groupby('record',sort=False):
        src=RUN/'source_snapshot/records'/rec;ref=load(src/'reference_events.csv')
        settings=json.loads((src/'metrics.json').read_text());fit_end=settings['split']['fit_end_time'];val_end=settings['split']['validation_end_time']
        tr=load(CACHE/'records'/rec/'train_origins.csv');va=load(CACHE/'records'/rec/'validation_origins.csv')
        sm=json.loads((CACHE/'records'/rec/'baseline_models.json').read_text())['age_survival']
        assert abs(sm['fit_last_raw_time']-fit_end)<1e-8
        t=group.origin_time.to_numpy();age=group.peak_age.to_numpy();end=group.raw_end.iloc[0];boundary=group.test_boundary.iloc[0]
        z=zero[zero.record==rec];assert np.allclose(z.origin_time,t,atol=1e-9,rtol=0)
        onset=ref.onset.to_numpy();specs[rec]={'survival':sm,'logistic':{},'constant':{}}
        first_path=RUN/'results/records'/rec/'learned_seed_2026_first_onsets.npz'
        hashes[str(first_path)]=hashlib.sha256(first_path.read_bytes()).hexdigest()
        with np.load(first_path) as saved:
            first=saved['first_onset_offsets'];assert np.allclose(saved['origin_time'],t,rtol=0,atol=1e-9)
        for h in HS:
            y=labels(t,onset,h);p=(first<=h+1e-9).mean(axis=1)
            assert np.array_equal(y,group[f'label_{h}s'])
            assert np.array_equal(p,group[f'risk_{h}s'])
            assert np.array_equal(flags(t,p).astype(int),group[f'alarm_{h}s'])
            assert np.array_equal((t+h<=end+1e-9).astype(int),group[f'complete_{h}s'])
            train=tr[np.isfinite(tr.age)&(tr.origin_time+h<=fit_end+1e-9)].copy()
            validation=va[np.isfinite(va.age)&(va.origin_time+h<=val_end+1e-9)].copy()
            assert all(not np.any((ref.onset-1e-9<=u)&(u<=ref.offset+1e-9)) for u in train.origin_time)
            logistic,spec=fit_logistic(train,validation,age,h,onset)
            q=(labels(train.origin_time,onset,h).sum()+.5)/(len(train)+1)
            specs[rec]['logistic'][h]=spec;specs[rec]['constant'][h]={'p':float(q),'n':len(train)}
            current=survival_risks(age,sm,h,False);renewal=survival_risks(age,sm,h,True)
            if h==1:assert np.allclose(current,renewal,atol=1e-10,equal_nan=True)
            methods={'DeepONet--Hopf':p,'Inter-peak survival':renewal,'Current-gap survival':current,
                     'Pure peak clock':((D-age>1e-9)&(D-age<=h+1e-9)).astype(float),
                     'Peak-age logistic':logistic,'Fit-prefix constant':np.full(len(t),q),
                     'Constant 0.5':np.full(len(t),.5),'Zero diffusion':z[f'risk_{h}s'].to_numpy()}
            for name,prob in methods.items():
                row,es,ar,od=evaluate(rec,name,h,t,prob,ref,end,boundary)
                records.append(row);events+=es;alarms+=ar;origins.append(od)
        audit.append({'record':rec,'origins':len(t),'labels_risks_flags_and_censoring_exact':True})
        print(rec,'audited and baselines fitted',flush=True)
    rr=pd.DataFrame(records);ee=pd.DataFrame(events);aa=pd.DataFrame(alarms);oo=pd.concat(origins,ignore_index=True)
    metrics=[]
    for (method,h),g in rr.groupby(['method','horizon'],sort=False):
        er=ee[(ee.method==method)&(ee.horizon==h)];s=g.select_dtypes('number').sum()
        item=dict(method=method,horizon=int(h),events=int(s.events),covered=int(s.covered),alarms=int(s.alarms),
                  matched=int(s.matched),unmatched=int(s.unmatched),tail_censored=int(s.tail_censored),
                  alarm_rate=float(s.alarms/(s.normal_origins*.05/60)),median_lead=float(er.lead.median()),
                  mean_lead=float(er.lead.mean()),covered_1s=int(er.covered_1s.sum()),covered_any=int(er.covered_any.sum()),
                  median_any=float(er.lead_any.median()),scored=int(s.scored),prevalence=float(s.positives/s.scored),
                  brier=float(s.brier_sum/s.scored),common_n=int(s.common_n),common_brier=float(s.common_brier_sum/s.common_n))
        if method=='DeepONet--Hopf':
            for key,oldkey in [('covered','covered_within_horizon'),('alarms','alarms'),('median_lead','median_lead_s'),('brier','brier')]:
                assert abs(item[key]-old.loc[h,oldkey])<1e-10,(h,key,item[key],old.loc[h,oldkey])
        metrics.append(item)
    mm=pd.DataFrame(metrics)
    rr.to_csv(OUT/'per_record.csv',index=False);ee.to_csv(OUT/'events.csv',index=False);aa.to_csv(OUT/'alarms.csv',index=False)
    oo.to_csv(OUT/'origins.csv',index=False);mm.to_csv(OUT/'summary.csv',index=False)
    (OUT/'baseline_models.json').write_text(json.dumps(specs,indent=2),encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps({'records':audit,'hashes':hashes,'seed':2026,'neural_retrained':False,
       'simulation_rerun':False,'main_horizon':2,'horizon_selected_after_results':True,
       'survival_extension':'Kaplan-Meier renewal recursion, 0.005-s grid; current-gap-only retained separately',
       'tail_fallback':'If S(age)=0, retain current-gap fallback 0.5 when its positive deadline is in range, otherwise zero',
       'logistic_refit':'Per horizon on normal fit-prefix origins with complete horizon; degree/C selected on validation only'},indent=2),encoding='utf-8')
    # Paired participant bootstrap for the two-second comparisons, fixed risks.
    rng=np.random.default_rng(2026);people=sorted(rr.participant.unique());draw=rng.integers(len(people),size=(5000,len(people)))
    def samples(method):
        g=rr[(rr.method==method)&(rr.horizon==2)].groupby('participant').sum(numeric_only=True).loc[people]
        a=g[['covered','events','alarms','normal_origins','brier_sum','scored']].to_numpy()[draw].sum(1)
        return np.column_stack([a[:,0]/a[:,1],a[:,2]/(a[:,3]*.05/60),a[:,4]/a[:,5]])
    main=samples('DeepONet--Hopf');ci={}
    for method in ['Inter-peak survival','Pure peak clock','Peak-age logistic','Fit-prefix constant']:
        diff=main-samples(method);ci[method]=dict(zip(['coverage_difference','alarm_rate_difference','brier_difference'],np.quantile(diff,[.025,.975],axis=0).T.tolist()))
    (OUT/'paired_intervals.json').write_text(json.dumps(ci,indent=2))
    print(mm[mm.horizon==2].to_string(index=False),flush=True)

if __name__=='__main__':main()
