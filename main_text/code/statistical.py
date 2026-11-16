"""Main-text statistical models, using the archived fitting and renewal formulas.

The selected saved fits are in models/statistical/baseline_models.json.
The renewal grid and empirical-tail fallback are unchanged.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
D = 1.695

def fit_survival(raw,stop,settings):
    d=np.abs(raw[:stop,1]-raw[:stop,2])/settings['peak_scale']
    peaks,_=find_peaks(d,distance=settings['distance_samples'],height=settings['height'],prominence=settings['prominence'])
    intervals=np.diff(raw[peaks,0])
    durations=intervals.tolist();observed=[True]*len(intervals)
    if len(peaks):
        durations.append(float(raw[stop-1,0]-raw[peaks[-1],0]));observed.append(False)
    durations=np.round(np.asarray(durations,float),8);observed=np.asarray(observed,bool)
    unique=np.unique(durations);survival=[];p=1.
    for value in unique:
        at_risk=int(np.sum(durations>=value-1e-10))
        failures=int(np.sum(np.isclose(durations,value,atol=1e-10,rtol=0)&observed))
        if at_risk:p*=1-failures/at_risk
        survival.append(p)
    return {'kind':'Kaplan_Meier','times':unique.tolist(),'survival':survival,
            'completed_gaps':len(intervals),'right_censored_gaps':int(bool(len(peaks))),
            'training_detected_peaks':len(peaks),'fit_last_raw_time':float(raw[stop-1,0]),
            'unsupported_conditional_tail_fallback':.5}

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

def predict_baseline(spec, age, horizon=2., method='survival'):
    age = np.atleast_1d(np.asarray(age, dtype=float))
    key = str(int(horizon))
    if key not in ('1','2','3','5') or float(key) != horizon:
        raise ValueError('Saved statistical fits support 1, 2, 3, or 5 seconds')
    if method == 'survival':
        return survival_risks(age, spec['survival'], horizon, renewal=True)
    if method == 'peak_clock':
        return np.where(np.isfinite(age), ((D-age>1e-9)&(D-age<=horizon+1e-9)).astype(float), np.nan)
    if method == 'fit_constant':
        return np.full(age.shape, spec['constant'][key]['p'])
    if method == 'constant_half':
        return np.full(age.shape, .5)
    if method != 'peak_age_logistic':
        raise ValueError('Unknown baseline: '+method)
    fit = spec['logistic'][key]
    if 'constant_fallback' in fit:
        return np.full(age.shape, fit['constant_fallback'])
    x = np.column_stack([(age/D)**j for j in range(1,fit['degree']+1)])
    x = (x-np.asarray(fit['feature_mean']))/np.asarray(fit['feature_scale'])
    return expit(x@np.asarray(fit['coefficient'])[0]+fit['intercept'][0])


def persistence(last_observed_block_mean, steps=100):
    return np.repeat(np.asarray(last_observed_block_mean,dtype=float)[None,:],steps,axis=0)
