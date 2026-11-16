"""Cycle-gap references and risk from decoded Hopf force paths.

References are offline. Prediction peak calibration uses the fit prefix only;
all observed peak contexts stop at their forecast origin.
"""
import numpy as np
import torch
from scipy.signal import find_peaks
from torch.nn import functional as F
from cycle_label_protocol import label_specification,reference_intervals


def merge_intervals(intervals,cycle=1.13,min_gap_cycles=1.5):
    merged=[];limit=cycle*min_gap_cycles
    for start,end in sorted(intervals):
        if merged and start-merged[-1][1]<limit-1e-9:
            merged[-1][1]=max(merged[-1][1],float(end))
        else:merged.append([float(start),float(end)])
    return [tuple(v) for v in merged]


def references(raw,reference_root,cycle=1.13,merge_gap_cycles=0.,reference_rule='paper_2020'):
    label_specification(cycle,reference_rule,merge_gap_cycles)
    unmerged=reference_intervals(raw,reference_root,cycle,reference_rule)
    return unmerged,merge_intervals(unmerged,cycle,merge_gap_cycles)


def fit_peak_settings(raw,stop,cycle=1.13,reference_rule='paper_2020'):
    specification=label_specification(cycle,reference_rule)
    fs=float(1/np.median(np.diff(raw[:,0])));d=np.abs(raw[:stop,1]-raw[:stop,2])
    distance=max(2,round(.45*cycle*fs))
    peaks,_=find_peaks(d,distance=distance,prominence=max(.15*np.std(d),1e-8))
    scale=float(d[peaks].mean()) if len(peaks) else max(float(np.max(d)),1e-8)
    return {'cycle':cycle,'fs':fs,'peak_scale':max(scale,1e-8),'distance_samples':distance,
            'height':.4,'prominence':.08,'onset_delay':specification['onset_delay_seconds'],'abnormal_gap':specification['abnormal_gap_seconds'],
            'reference_rule':reference_rule,'onset_convention':specification['onset_convention'],
            'calibration_stop':int(stop),'calibration_end_time':float(raw[stop-1,0]),'coarse_fit_peaks':len(peaks)}


def observed_contexts(raw,ids,stride,settings):
    d=np.abs(raw[:,1]-raw[:,2])/settings['peak_scale'];length=round(2*settings['cycle']*settings['fs'])+1
    contexts=[];ages=[]
    for index in ids:
        stop=(int(index)+1)*stride;prefix=d[:stop]
        peaks,_=find_peaks(prefix,distance=settings['distance_samples'],height=settings['height'],prominence=settings['prominence'])
        ages.append(float(raw[stop-1,0]-raw[peaks[-1],0]) if len(peaks) else np.nan)
        recent=prefix[-length:]
        contexts.append(np.pad(recent,(length-len(recent),0),mode='edge'))
    return np.array(contexts,dtype='float32'),np.array(ages,dtype='float32')


def hard_cycle_risk(force_paths,contexts,ages,settings,fine_dt,event_horizon=1.):
    """Probability of a future onset, not of a censored no-peak tail.

    With event_horizon < onset_delay, only the latest observed peak can create an onset
    in that horizon. Paths must extend beyond its abnormal-gap deadline. Lack of
    sufficient simulation is unknown (NaN), never counted as a missing step.
    """
    if event_horizon>=settings['onset_delay']:raise ValueError('This clock implementation requires event horizon < onset_delay')
    paths=np.asarray(force_paths);B,M,N,_=paths.shape
    d=np.abs(paths[...,0]-paths[...,1])/settings['peak_scale'];risk=[];info=[]
    for b,age in enumerate(ages):
        onset=settings['onset_delay']-float(age);deadline=settings['abnormal_gap']-float(age)
        row={'peak_age':float(age),'candidate_onset_offset':onset,'confirmation_deadline_offset':deadline}
        if not np.isfinite(age):risk.append(np.nan);row['status']='no_confirmed_observed_peak'
        elif not (1e-9<onset<=event_horizon+1e-9):risk.append(0.);row['status']='no_new_onset_in_horizon'
        elif deadline>fine_dt*N+1e-9:risk.append(np.nan);row['status']='insufficient_simulated_future'
        else:
            outcomes=[];history_length=len(contexts[b]);minimum=.45*settings['cycle']-float(age)
            for m in range(M):
                sequence=np.r_[contexts[b],d[b,m]]
                peaks,_=find_peaks(sequence,distance=settings['distance_samples'],height=settings['height'],prominence=settings['prominence'])
                times=(peaks-history_length+1)*fine_dt
                future=times[(peaks>=history_length)&(times>=minimum-1e-9)]
                next_peak=float(future[0]) if len(future) else np.inf
                outcomes.append(next_peak>deadline+1e-9)
            risk.append(float(np.mean(outcomes)));row['status']='evaluable'
        info.append(row)
    return np.array(risk),info


def soft_cycle_risk(force_paths,contexts,ages,settings,fine_dt,event_horizon=1.):
    """Piecewise differentiable peak-presence proxy used only for training.

    Local-maximum locations form a detached mask. Height and local prominence
    remain differentiable; the hard SciPy rule is always used at evaluation.
    """
    B,M,N,_=force_paths.shape;d=(force_paths[...,0]-force_paths[...,1]).abs()/settings['peak_scale']
    history=contexts[:,None,:].expand(-1,M,-1);seq=torch.cat((history,d),-1);flat=seq.reshape(B*M,1,-1)
    w=max(3,round(.25*settings['cycle']/fine_dt));left=-F.max_pool1d(F.pad(-flat,(w-1,0),mode='replicate'),w,stride=1)
    right=-F.max_pool1d(F.pad(-flat,(0,w-1),mode='replicate'),w,stride=1)
    prominence=flat-torch.maximum(left,right)
    local=(flat>=F.max_pool1d(flat,3,stride=1,padding=1)).detach()
    strength=torch.sigmoid((flat-settings['height'])/.05)*torch.sigmoid((prominence-settings['prominence'])/.02)*local
    strength=strength.reshape(B,M,-1)[...,-N:]
    times=torch.arange(1,N+1,device=d.device,dtype=d.dtype)*fine_dt
    onset=settings['onset_delay']-ages;deadline=settings['abnormal_gap']-ages
    valid=torch.isfinite(ages)&(onset>1e-9)&(onset<=event_horizon+1e-9)&(deadline<=N*fine_dt+1e-9)
    window=(times[None,:]<=deadline[:,None]+1e-9)&(times[None,:]>=.45*settings['cycle']-ages[:,None]-1e-9)
    probability=(1-(strength*window[:,None,:]).amax(-1)).mean(1)
    return torch.where(valid,probability,torch.zeros_like(probability)),valid


def save_empty_result(out,args,t,v,train,val,events,b1,b2,mean,scale,input_source):
    """An event can occupy the full test suffix: report, do not invent origins."""
    import csv,json
    from normal_warning_protocol import assess_warnings,write_csv
    score,alarms,er=assess_warnings([],[],events,float(t[b2]))
    m={'configuration':vars(args),'device':'not_run','windows':[len(train[0]),len(val[0]),0],'best_epoch':None,'epochs_run':0,
       'metrics':{'parameter_hopf':{'mae_xy':[None,None],'rmse_xy':[None,None]}},'experimental_triggers':0,'risk_range':[None,None],
       'reference_event_count':len(events),'reference_type':'cycle-defined reference; see label_policy.json','input_source':input_source,
       'risk_rule':'cycle_gap','cycle_evaluable_origins':0,'skip_reason':'No normal test origins under selected reference labels; no test model fitted',
       'split':{'fit_fraction':b1/len(t),'validation_end_fraction':b2/len(t),'test_fraction':1-b2/len(t),
                'fit_end_time':float(t[b1-1]),'validation_end_time':float(t[b2-1]),'test_boundary':float(t[b2]),
                'first_test_origin':None,'normal_only':True,'scaling':'fit prefix only'},'warning_evaluation':score,
       'reference_merge_gap_seconds':args.merge_gap_cycles*args.cycle,'model':'parameter Hopf, not fitted because no eligible test origins'}
    m['label_policy']=label_specification(args.cycle,args.reference_rule,args.merge_gap_cycles)
    (out/'metrics.json').write_text(json.dumps(m,indent=2));(out/'warning_summary.json').write_text(json.dumps(score,indent=2))
    write_csv(out/'event_warning_validation.csv',er,['reference_onset','reference_end','previous_reference_end','eligible_origins','warned_before','first_warning','lead_seconds','warning_count'])
    write_csv(out/'warning_times.csv',alarms,['warning_time','next_reference_onset','before_next_event','lead_seconds'])
    write_csv(out/'risk.csv',[],['origin_time','transition_probability','experimental_trigger','beta','frequency_hz','sigma','future_onset_label','normal_origin'])
    (out/'NO_NORMAL_TEST_ORIGINS.txt').write_text(m['skip_reason'])
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,4));ax.plot(t,v[:,0],label='X');ax.plot(t,v[:,1],label='Y');ax.axvline(t[b2],color='red',ls='--',label='65% split')
    ax.set(title='No normal test origins: no prediction issued',xlabel='Time (s)',ylabel='Force');ax.legend();fig.tight_layout();fig.savefig(out/'raw_force_warning.png',dpi=180);plt.close(fig)
    print(json.dumps(m,indent=2))
