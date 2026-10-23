"""Read-only audit probes; native AIHWKit CPU, no Drive/setup/full-run execution."""
from pathlib import Path
import ast, contextlib, copy, hashlib, io, json, math, sys
from dataclasses import asdict,replace
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from aihwkit.nn import AnalogLinear
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--notebook',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.repo/'tests'))
import test_notebook_gradients as loader
import test_coordinator_transfer as worker_loader
OUT=args.out; OUT.mkdir(parents=True,exist_ok=False)
source=args.notebook; original_bytes=source.read_bytes()
with contextlib.redirect_stdout(io.StringIO()):
 r=loader.load_runtime(source,OUT/'runtime.py')
 old=worker_loader.load_runtime(args.repo/'DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb',OUT/'completed_worker_runtime.py')
torch.set_num_threads(2); device=torch.device('cpu')
report={'scope':'Native synthetic CPU probes; no user checkpoint, Drive, datasets, setup, full training or GPU run executed'}
n=json.loads(original_bytes)
report['all_code_cells_compile']=all(compile(''.join(c['source']),f'cell_{i}','exec') is not None for i,c in enumerate(n['cells']) if c['cell_type']=='code')
# Exercise only the stage entry with stubs; no stage is allowed to run.
ns=dict(RUN_FULL_RETENTION=True,FULL_EXP=SimpleNamespace(cfg=object()),OUTPUT_DIR=OUT,
 begin_check_session=lambda *a:None,require_smoke_pass=lambda *a:None)
try:
 exec(compile(''.join(n['cells'][45]['source']),'uploaded_cell_45','exec'),ns)
except NameError as e:
 report['retention_entry_failure']={'type':type(e).__name__,'message':str(e)}
else: raise AssertionError('Expected misspelled function to fail')
# Compare current settings with the completed worker protocol, with a common synthetic dataset hash.
uc=r.make_config('resnet32',False,r.full_run_id()).resolved()
oc=old.make_config('resnet32',False,old.full_run_id()).resolved()
report['configuration_differences']={k:{'uploaded':v,'completed_worker':asdict(oc)[k]} for k,v in asdict(uc).items() if v!=asdict(oc)[k]}
fixture_hashes={'same_synthetic_dataset':'fixture'}
uf=r.compute_fingerprints_resnet(uc,fixture_hashes,r.RESNET_SPEC)
of=old.compute_fingerprints_resnet(oc,fixture_hashes,old.RESNET_SPEC)
report['training_fingerprint_differs_from_completed_worker']=uf['train']!=of['train']
report['pcm_code_hash_matches_completed_worker']=uf['code_hashes']['pcm_retention']==of['code_hashes']['pcm_retention']
# Same scalar NSR, different actual weight gradients when bypassing AIHWKit's hook.
cfg=replace(r.Config(),require_cuda=False,fast_mode=False,strict_154_epochs=False)
layer=AnalogLinear(8,3,bias=False,rpu_config=r.make_hwa_rpu(cfg))
g=torch.Generator().manual_seed(713)
w=torch.randn(3,8,generator=g)*torch.tensor([.01,.1,.5]).view(-1,1)
layer.set_weights(w)
holder=torch.nn.Module();holder.fc1=layer;holder.layers=lambda:r.OrderedDict([('fc1',layer)])
p=r.physical_weight_parameter(layer)
original=r.nsr_penalty(holder,cfg);original.backward(); hooked=p.grad.detach().clone()
layer.zero_grad(set_to_none=True)
leaves=r.hook_free_tile_weights(holder)
bypassed=r.nsr_penalty(r.TileWeightView(leaves),cfg);bypassed.backward();r.add_tile_penalty_grads(holder,leaves)
raw=p.grad.detach().clone(); tile,=list(layer.analog_tiles()); scales=tile.get_scales().detach()
assert torch.equal(original,bypassed)
torch.testing.assert_close(hooked,raw/scales.view(-1,1).square(),rtol=2e-6,atol=1e-8)
report['nsr_gradient_route']={'same_scalar_loss':True,'gradients_identical':bool(torch.equal(hooked,raw)),
 'per_row_scales':scales.tolist(),'native_gradient_l2':float(hooked.double().norm()),
 'v2_bypassed_gradient_l2':float(raw.double().norm()),
 'native_equals_bypassed_divided_by_scale_squared':True,
 'bypass_only_in_resnet_trainer': 'hook_free_tile_weights' in r.train_hwa_resnet.__code__.co_names and 'hook_free_tile_weights' not in r.train_hwa_all.__code__.co_names}
# Per-entry finiteness does not guarantee that the FP32 norm reduction is finite.
m=torch.nn.Linear(2,1,bias=False);m.weight.grad=torch.full_like(m.weight,3e20)
finite=bool(torch.isfinite(m.weight.grad).all())
try:r.clip_grad_norm_or_explain(m,.8,'synthetic finite-entry norm overflow')
except FloatingPointError as e: report['fp32_norm_overflow']={'entries_finite':finite,'error':str(e),'fp64_norm':float(m.weight.grad.double().norm())}
else:raise AssertionError('Expected an overflowing norm')
# Tiny actual ResNet continuations: interrupted and uninterrupted HWA and NSR.
cfg=replace(r.Config(),require_cuda=False,fast_mode=True,strict_154_epochs=False,
 epochs=2,digital_epochs=1,train_repeats=1,batch_size=2,test_batch_size=2,validation_reads=1)
g=torch.Generator().manual_seed(181)
trainset=TensorDataset(torch.randn(4,3,32,32,generator=g),torch.tensor([1,2,3,4]))
valset=TensorDataset(torch.randn(4,3,32,32,generator=g),torch.tensor([5,6,7,8]))
def loaders():
 train=DataLoader(trainset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(182));train.split_name='train'
 val=DataLoader(valset,batch_size=2);val.split_name='val';return train,val
trainchecks={}
with (OUT/'native_training_probe.log').open('w') as log,contextlib.redirect_stdout(log):
 digital=r.make_digital_resnet32(71).eval();identity=r.state_dict_hash(digital.state_dict())
 for method in [r.HWA,r.NSR]:
  train,val=loaders(); resumed=OUT/'native-fixtures'/method/'resumed';refdir=OUT/'native-fixtures'/method/'reference'
  r._INTERRUPT_HOOK.update(stage=f'hwa:{method}',after_epoch=1)
  try:r.train_hwa_resnet(method,cfg,0,digital,identity,train,val,device,resumed,'synthetic-audit-fp')
  except r.SimulatedInterruption:pass
  else:raise AssertionError('Interruption hook did not fire')
  train,val=loaders();a=r.train_hwa_resnet(method,cfg,0,digital,identity,train,val,device,resumed,'synthetic-audit-fp')
  train,val=loaders();b=r.train_hwa_resnet(method,cfg,0,digital,identity,train,val,device,refdir,'synthetic-audit-fp')
  af=r.torch_load(resumed/f'{method}_final.pt');bf=r.torch_load(refdir/f'{method}_final.pt')
  trainchecks[method]={'resume_model_bit_identical':r.state_dict_hash(af['model'])==r.state_dict_hash(bf['model']),
   'resume_histories_identical':a['history']==b['history'],'optimizer_steps':a['optimizer_steps'],
   'epochs_completed':a['analog_epochs_completed'],'best_restore_accuracy_difference_pp':a['restore_abs_diff_pp'],
   'first_physical_weight_update':a['first_step_max_weight_change'],
   'positive_nsr_penalty' :a['history'][1]['train_extra']>0 if method==r.NSR else None}
report['native_tiny_resnet_training']=trainchecks
assert all(x['resume_model_bit_identical'] and x['resume_histories_identical'] and x['optimizer_steps']==4 for x in trainchecks.values())
# No checkpoint provided: only preserve the original GPU output as evidence, not a reproduced cause.
errors=[]
for i,c in enumerate(n['cells']):
 for o in c.get('outputs',[]):
  if o.get('output_type')=='error': errors.append(dict(cell_index=i,ename=o.get('ename'),evalue=o.get('evalue')))
report['saved_gpu_errors']=errors
report['original_upload_unchanged']=source.read_bytes()==original_bytes
(OUT/'probe_results.json').write_text(json.dumps(report,indent=2,default=str))
print(json.dumps(report,indent=2,default=str))
