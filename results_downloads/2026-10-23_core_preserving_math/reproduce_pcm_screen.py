from pathlib import Path
import json, sys, time, math
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
import argparse
parser=argparse.ArgumentParser(description='Native PCM parameter screen; synthetic weights, not model accuracy')
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--realizations',type=int,default=128)
args=parser.parse_args()
if args.realizations<2: parser.error('Use at least two device realizations')
ROOT=args.repo; OUT=args.output; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'tests'))
from test_coordinator_transfer import load_runtime
r=load_runtime(ROOT/'DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb',OUT/'runtime.py')
torch.set_num_threads(2)
cfg=replace(r.Config(),require_cuda=False,fast_mode=False)
configs=[(1,64,4),(2,64,4),(4,64,4),(1,64,16),(1,256,4),(2,256,4),(4,256,4),(4,256,16),(1,256,16),(1,1024,4),(1,4096,4)]
print('Model-level native PCM screen: synthetic conductance distributions, NOT user accuracy',flush=True)
rows=[]; started=time.monotonic()
for distribution,power in [('uniform',1),('small_weight_heavy',3)]:
 base=torch.linspace(-1,1,513)
 weights=(base.sign()*base.abs().pow(power)).reshape(1,-1)
 signal=float(weights.square().mean())
 for realization in range(args.realizations):
  root=60_000_000+realization*10_000
  main=r.PCMPhysicalMatrix(weights,cfg,root+10)
  programmed=main.logical_weight(main.g_programmed)
  current=main.read_conductances(5*r.YEAR,root+101_100)
  for B,R,K in configs:
   c=replace(cfg,reference_cell_budget=R,reference_read_repeats=K)
   bank=r.PCMClassReferenceBank(c,B,root+30,torch.device('cpu'))
   gain=bank.gains(5*r.YEAR,root+501_111)
   corrected=main.logical_weight(r.apply_driftslice(main,current,gain,bank.edges))
   rows.append(dict(distribution=distribution,realization=realization,B=B,R=R,K=K,
    refs_per_class=R//B,reference_reads_per_observation=R*K,
    mse_to_target=float((corrected-weights).square().mean()),
    mse_to_programmed20=float((corrected-programmed).square().mean()),signal=signal,
    gain_max=float(gain.max()),gain_min=float(gain.min()),
    gain_cap_fraction=float(gain.ge(cfg.gain_max-1e-7).float().mean())))
  if (realization+1)%32==0: print(distribution,realization+1,'/',args.realizations,round(time.monotonic()-started,1),'s',flush=True)
d=pd.DataFrame(rows); d.to_csv(OUT/'native_pcm_model_screen_raw.csv',index=False)
s=d.groupby(['distribution','B','R','K']).agg(mean_mse_to_target=('mse_to_target','mean'),mean_mse_to_programmed20=('mse_to_programmed20','mean'),signal=('signal','first'),gain_cap_fraction=('gain_cap_fraction','mean'),highest_gain=('gain_max','max')).reset_index()
s['nrmse_to_target']=np.sqrt(s.mean_mse_to_target/s.signal)
s['nrmse_to_programmed20_normalized_by_target_signal']=np.sqrt(s.mean_mse_to_programmed20/s.signal)
control=s[(s.B==1)&(s.R==64)&(s.K==4)][['distribution','mean_mse_to_target']].rename(columns={'mean_mse_to_target':'control_mse'})
s=s.merge(control,on='distribution',validate='many_to_one')
s['mse_change_vs_control_pct']=100*(s.mean_mse_to_target/s.control_mse-1)
s.to_csv(OUT/'native_pcm_model_screen_summary.csv',index=False)
print(s.to_string(index=False))
metadata=dict(scope='Native AIHWKit 1.1.0 PCM model; synthetic target weights, not trained-network accuracy',
    device='cpu',torch=str(torch.__version__),realizations_per_distribution=args.realizations,
    weights_per_realization=513,distributions=['uniform','small_weight_heavy'],
    programming_noise_scale=cfg.programming_noise_scale,read_noise_scale=cfg.read_noise_scale,
    drift_scale=cfg.drift_scale,gmax_us=cfg.gmax_us,reference_floor_us=cfg.reference_floor_us,
    gain_min=cfg.gain_min,gain_max=cfg.gain_max,programming_bounds='P4 existing [0,gmax] cap',
    t_s=5*r.YEAR,seed_root=60_000_000,shared_main_devices_across_configurations=True,
    core_formula_changed=False,network_accuracy_evaluated=False)
(OUT/'native_pcm_screen_scope.json').write_text(json.dumps(metadata,indent=2))
print('Completed in',time.monotonic()-started,'seconds')
