"""Offline screening/evaluation explicitly requested: normal origins only, point markers.
Reference labels select eligible origins and score outcomes; future labels never enter
the neural network. This is conditional evaluation, not a standalone causal normal-state gate.
"""
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def assess_warnings(origins,triggers,events,boundary):
    origins=np.asarray(origins);triggers=np.asarray(triggers,bool)
    assert len(origins)==len(triggers) and np.all(np.diff(origins)>0)
    assert all(not any(s-1e-9<=t<=e+1e-9 for s,e in events) for t in origins)
    alarms=[];event_rows=[]
    for t in origins[triggers]:
        next_events=[(s,e) for s,e in events if s>t+1e-9]
        next_onset=next_events[0][0] if next_events else None
        alarms.append({'warning_time':float(t),'next_reference_onset':next_onset,
                       'before_next_event':bool(next_events),'lead_seconds':next_onset-float(t) if next_events else None})
    for k,(s,e) in enumerate(events):
        if s<boundary:continue  # ongoing events at boundary are not new forecast targets
        previous_end=events[k-1][1] if k else -np.inf
        # Test-boundary samples are eligible; reference-event endpoints are not.
        hits=[a['warning_time'] for a in alarms if a['warning_time']>=boundary-1e-9
              and a['warning_time']>previous_end+1e-9 and a['warning_time']<s-1e-9]
        eligible=origins[(origins>=boundary-1e-9)&(origins>previous_end+1e-9)&(origins<s-1e-9)]
        first=min(hits) if hits else None
        event_rows.append({'reference_onset':s,'reference_end':e,'previous_reference_end':previous_end if np.isfinite(previous_end) else None,
                           'eligible_origins':int(len(eligible)),'warned_before':bool(hits),'first_warning':first,
                           'lead_seconds':s-first if hits else None,'warning_count':len(hits)})
    covered=sum(r['warned_before'] for r in event_rows)
    summary={'n_test_events':len(event_rows),'events_warned_before':covered,'events_missed':len(event_rows)-covered,
             'event_coverage':covered/len(event_rows) if event_rows else None,'n_warnings':len(alarms),
             'warnings_without_later_observed_onset':sum(a['next_reference_onset'] is None for a in alarms),
             'all_origins_non_fog':True,'criterion':'previous reference end < warning time < next reference onset; no maximum lead',
             'screening':'Offline reference labels select normal test origins; not a deployable normal-state detector',
             'reference':'Cycle-defined algorithmic reference, not independent clinical ground truth'}
    return summary,alarms,event_rows


def write_csv(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def evaluate_and_plot(out,t,v,x,train,model,device,mean,scale,events,b2,args,dt):
    out=Path(out); rows=list(csv.DictReader((out/'risk.csv').open()))
    rt=np.array([float(r['origin_time']) for r in rows]);risk=np.array([float(r.get('transition_probability',r.get('latent_entry_probability'))) for r in rows])
    triggers=np.array([r['experimental_trigger']=='1' for r in rows])
    summary,alarms,er=assess_warnings(rt,triggers,events,float(t[b2]))
    write_csv(out/'warning_times.csv',alarms,['warning_time','next_reference_onset','before_next_event','lead_seconds'])
    write_csv(out/'event_warning_validation.csv',er,['reference_onset','reference_end','previous_reference_end','eligible_origins','warned_before','first_warning','lead_seconds','warning_count'])
    (out/'warning_summary.json').write_text(json.dumps(summary,indent=2))
    # Real training predictions, same fit prefix as the trained checkpoint.
    ids=train[3];H=args.horizon
    if (out/'train_fit.csv').exists():
        fit=np.atleast_2d(np.loadtxt(out/'train_fit.csv',delimiter=',',skiprows=1))
        assert len(fit)==len(ids) and np.allclose(fit[:,0],t[ids])
        true,estimate=fit[:,1],fit[:,2]  # saved fit; no resampling on warning-only replay
    else:
        torch.manual_seed(2026);pred=[]
        with torch.no_grad():
            for b in train[0].split(16):pred.append(model(b.to(device),32)['joint'].mean(1).cpu().numpy())
        pred=np.concatenate(pred)*scale+mean
        current=v[ids,0]-v[ids,1]
        true=v[ids+H,0]-v[ids+H,1]-current
        estimate=pred[:,-1,0]-pred[:,-1,1]-current
    mae=float(np.abs(true-estimate).mean())
    write_csv(out/'train_fit.csv',[{'time':float(ti),'true_delta':float(y),'predicted_delta':float(p)} for ti,y,p in zip(t[ids],true,estimate)],['time','true_delta','predicted_delta'])
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'savefig.dpi':180})
    legends=[Line2D([],[],color='#00843d',ls='-.',label='Warning issued'),Line2D([],[],color='#d62728',ls='--',label='Reference FOG onset'),
             Line2D([],[],color='#555555',ls=':',label='65% split'),Line2D([],[],color='#7b3294',ls=':',label='First normal prediction')]
    def markers(ax,zoom=False):
        for r in er:ax.axvline(r['reference_onset'],color='#d62728',ls='--',lw=1.5)
        for a in alarms:ax.axvline(a['warning_time'],color='#00843d',ls='-.',lw=1.3)
        ax.axvline(t[b2],color='#555555',ls=':',lw=1.4);ax.axvline(rt[0],color='#7b3294',ls=':',lw=1.4)
        ax.grid(alpha=.23);ax.set_xlim(max(t[0],t[b2]-1) if zoom else t[0],t[-1]);ax.set_xlabel('Time (s)')
    # Break the curve over excluded origins; no visual interpolation through FOG.
    cycle_mode=getattr(args,'risk_rule','latent_disk')=='cycle_gap'
    grid=t[b2:] if cycle_mode else t[b2:len(t)-H];curve=np.full(len(grid),np.nan)
    indexes=np.searchsorted(grid,rt);assert np.allclose(grid[indexes],rt)
    curve[indexes]=risk
    loss=np.atleast_2d(np.loadtxt(out/'training.csv',delimiter=','))
    fig=plt.figure(figsize=(16,10));gs=fig.add_gridspec(2,2,height_ratios=[1,1.1])
    ax=fig.add_subplot(gs[0,0]);ax.plot(loss[:,0],loss[:,1],'b-',label='Fit (first 52%)');ax.plot(loss[:,0],loss[:,2],color='orange',label='Validation (52-65%)')
    ax.set(title='Joint training loss',xlabel='Epoch',ylabel='Objective');ax.legend();ax.grid(alpha=.23)
    ax=fig.add_subplot(gs[0,1]);ax.plot(t[ids],true,color='blue',alpha=.7,label='True delta');ax.plot(t[ids],estimate,'r--',alpha=.8,label='Predicted delta')
    ax.set(title=f'Training fit: force-difference delta (MAE={mae:.2f})',xlabel='Origin time (s)',ylabel='Delta in original units');ax.legend();ax.grid(alpha=.23)
    ax=fig.add_subplot(gs[1,:]);markers(ax,True);line,=ax.plot(grid,curve,color='black',lw=1.4,label='Cycle-abnormality probability' if cycle_mode else 'Risk at normal origins')
    thresh=ax.axhline(args.alarm_threshold,color='orange',ls='--',label=f'Fixed threshold={args.alarm_threshold:g}')
    ax.scatter(rt[triggers],risk[triggers],marker='x',color='#00843d',s=50,zorder=5)
    label_lanes=[-np.inf]*3;label_gap=(t[-1]-max(t[0],t[b2]-1))*.075
    for r in er:
        label=f'Onset {r["reference_onset"]:.3f}s'
        if r['warned_before']:label+=f'\nLead {r["lead_seconds"]:.3f}s'
        else:label+='\nNo earlier warning'
        lane=next((j for j,last in enumerate(label_lanes) if r['reference_onset']-last>label_gap),int(np.argmin(label_lanes)))
        label_lanes[lane]=r['reference_onset']
        near_left=r['reference_onset']-ax.get_xlim()[0]<label_gap/2
        near_right=ax.get_xlim()[1]-r['reference_onset']<label_gap/2
        align='left' if near_left else 'right' if near_right else 'center'
        ax.annotate(label,(r['reference_onset'],1.27-.10*lane),xytext=(3 if near_left else -3 if near_right else 0,0),
                    textcoords='offset points',ha=align,va='top',fontsize=8,color='#a51c30')
    ax.set(title='Rolling warning: normal origins only; no predictions during reference FOG',ylabel='Cycle event probability' if cycle_mode else 'Latent-region risk',ylim=(-.03,1.31))
    ax.legend(handles=[line,thresh]+legends,loc='upper center',bbox_to_anchor=(.5,-.14),fontsize=8,ncol=3)
    fig.tight_layout();fig.savefig(out/'warning_dashboard.png');fig.savefig(out/'warning_dashboard.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,5.5));markers(ax)
    l1,=ax.plot(t,v[:,0],color='#4c9bd1',lw=1,label='X force (5-sample block mean)')
    l2,=ax.plot(t,v[:,1],color='#ff9845',lw=1,label='Y force (5-sample block mean)')
    ax.set(title='Force signals: warning times and reference onset times',ylabel='Force (original units)');ax.legend(handles=[l1,l2]+legends,loc='upper left',fontsize=9,ncol=2)
    for r in er:
        ax.annotate(f'FOG {r["reference_onset"]:.3f}s',(r['reference_onset'],.98),xycoords=ax.get_xaxis_transform(),xytext=(-4,0),textcoords='offset points',ha='right',va='top',fontsize=9,color='#d62728')
    for a in alarms:
        ax.annotate(f'Warning {a["warning_time"]:.3f}s',(a['warning_time'],.04),xycoords=ax.get_xaxis_transform(),xytext=(-4,0),textcoords='offset points',ha='right',va='bottom',fontsize=9,color='#00843d')
    fig.tight_layout();fig.savefig(out/'raw_force_warning.png');fig.savefig(out/'raw_force_warning.pdf');plt.close(fig)
    if getattr(args,'architecture','residual')=='parameters':
        with np.load(out/'predictions.npz') as saved:
            current=saved['current_parameters'];future=saved['parameter_curves'][:,-1]
        fig,axs=plt.subplots(3,1,figsize=(13,8),sharex=True)
        for j,(name,ax) in enumerate(zip(['beta','frequency (Hz)','noise sigma'],axs)):
            pnow=np.full(len(grid),np.nan);pnext=pnow.copy()
            pnow[indexes]=current[:,j];pnext[indexes]=future[:,j]
            ax.plot(grid,pnow,color='#377eb8',label='Parameter at origin')
            ax.plot(grid,pnext,color='#e66101',ls='--',label=f'Predicted parameter at +{H*dt:.2f}s')
            markers(ax,True);ax.set_ylabel(name);ax.legend(loc='upper left',fontsize=8)
        fig.suptitle('DeepONet parameter forecasts used by the Hopf equation')
        fig.tight_layout();fig.savefig(out/'parameter_forecasts.png');fig.savefig(out/'parameter_forecasts.pdf');plt.close(fig)
    return summary


def main():
    """Replay saved risks, optionally override confirmation, without retraining."""
    import argparse
    import hashlib,shutil
    from generalized_hopf_deeponet import make_model,threshold_warnings
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True)
    ap.add_argument('--confirm',type=int,help='Override confirmation count; 1 means immediate warning')
    a=ap.parse_args()
    if a.confirm is not None and a.confirm<1:ap.error('confirm must be positive')
    out=Path(a.results);torch.set_num_threads(2)
    ck=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
    summary=json.loads((out/'metrics.json').read_text())
    args=argparse.Namespace(**summary.get('configuration',ck['args']))
    if a.confirm is not None:
        backup=out/'before_immediate_alarm'
        if not backup.exists():
            backup.mkdir()
            for name in ['metrics.json','risk.csv','warning_times.csv','warning_summary.json','event_warning_validation.csv',
                         'warning_dashboard.png','warning_dashboard.pdf','raw_force_warning.png','raw_force_warning.pdf',
                         'parameter_forecasts.png','parameter_forecasts.pdf']:
                if (out/name).exists():shutil.copy2(out/name,backup/name)
        old_confirm=args.confirm;args.confirm=a.confirm
        protected=['model.pt','predictions.npz','parameter_forecasts.csv','training.csv','train_fit.csv']
        hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in protected if (out/name).exists()}
        risk_rows=list(csv.DictReader((out/'risk.csv').open()))
        times=[float(r['origin_time']) for r in risk_rows];risks=[float(r.get('transition_probability',r.get('latent_entry_probability'))) for r in risk_rows]
        flags=threshold_warnings(times,risks,args.alarm_threshold,args.confirm,ck['dt'])
        for row,flag in zip(risk_rows,flags):row['experimental_trigger']=str(flag)
        write_csv(out/'risk.csv',risk_rows,list(risk_rows[0]))
        summary['configuration']['confirm']=args.confirm
        summary['experimental_triggers']=sum(flags)
        summary['warning_postprocessing']={'previous_confirm':old_confirm,'confirm':args.confirm,
            'condition':'risk > threshold; first exceedance per continuous run',
            'retrained':False,'source_hashes':hashes}
    if args.reference_root:
        with np.load(Path(args.reference_root)/'data/records'/(args.tag.replace(' ','_')+'.npz')) as z:
            raw=np.column_stack((z['t'],z['x'],z['y']))
    else:raw=np.loadtxt(args.data)
    blocks=raw[:len(raw)//args.stride*args.stride].reshape(-1,args.stride,3)
    averaged=blocks.mean(1);t=blocks[:,-1,0];v=averaged[:,1:]
    b2=int(args.train_fraction*len(t));b1=int((1-args.validation_within_train)*b2)
    x=((v-ck['mean'])/ck['scale']).astype('float32');ids=np.arange(args.history-1,b1-args.horizon,2)
    from history_window import history_specification,network_histories
    specification=history_specification(raw[:,0],args.stride,args.history,getattr(args,'history_cycles',None),getattr(args,'cycle',1.13))
    train=(torch.tensor(network_histories(x,raw,ids,args.stride,ck['mean'],ck['scale'],specification)),None,None,ids)
    events=[(float(r['onset']),float(r['offset'])) for r in csv.DictReader((out/'reference_events.csv').open())]
    model=make_model(args,ck['dt']);model.load_state_dict(ck['state_dict']);model.eval()
    result=evaluate_and_plot(out,t,v,x,train,model,torch.device('cpu'),ck['mean'],ck['scale'],events,b2,args,ck['dt'])
    summary['warning_evaluation']=result
    if a.confirm is not None:
        assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
        summary['warning_postprocessing']['protected_files_unchanged']=True
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
