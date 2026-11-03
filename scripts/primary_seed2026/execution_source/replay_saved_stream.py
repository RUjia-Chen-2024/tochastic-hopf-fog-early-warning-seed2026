"""Replay saved per-batch RNG states, comparing full paths bit for bit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from run_intervention import simulate
from multihorizon_events import first_new_onsets,cumulative_risks

p=argparse.ArgumentParser();p.add_argument('--record',default='Pt_10_FD10_1')
p.add_argument('--source',required=True);p.add_argument('--results',default='results');a=p.parse_args()
source=Path(a.source);dest=Path(a.results)/'records'/a.record
sys.path.insert(0,str(source/'code'))
from generalized_hopf_deeponet import make_model
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ck=torch.load(source/'records'/a.record/'model.pt',map_location='cpu',weights_only=False)
inputs=np.load(source/'records'/a.record/'network_inputs.npz')['inputs']
checks=[]
for variant in ['zero_noise','learned_seed_2026']:
    state=torch.load(dest/(variant+'_rng_stream.pt'),map_location='cpu',weights_only=False)
    device=torch.device(state['device'])
    model=make_model(SimpleNamespace(**ck['args']),ck['dt']).to(device).eval();model.load_state_dict(ck['state_dict'])
    mean=torch.tensor(ck['mean'],dtype=torch.float32,device=device);scale=torch.tensor(ck['scale'],dtype=torch.float32,device=device)
    saved=np.load(dest/(variant+'_force_paths.npy'),mmap_mode='r')
    matched=0
    for b in state['batches']:
        torch.set_rng_state(b['cpu_state']);torch.cuda.set_rng_state(b['cuda_state'],device)
        x=torch.tensor(inputs[b['first']:b['stop']],dtype=torch.float32,device=device)
        fine,_,_,_=simulate(model,x,state['draws'],0. if variant=='zero_noise' else 1.)
        decoded=(fine*scale+mean).cpu().numpy()
        assert np.array_equal(decoded,saved[b['first']:b['stop']]),(variant,b['first'])
        matched+=len(x)
    checks.append(dict(record=a.record,variant=variant,seed=2026,origins=matched,batches=len(state['batches']),
                       all_paths_exact=True,draws=state['draws'],device=str(device)))
out=Path(a.results)/'full_record_replay_check.json';out.write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
