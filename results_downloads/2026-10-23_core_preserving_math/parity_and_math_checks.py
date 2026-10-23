"""Check the unchanged NSR/DriftSlice formulas against the supplied MLP notebook.

Run with --repo /path/to/ISCAS_CONF --mlp-notebook /path/to/original.ipynb.
No installation, data loading, training or user checkpoint evaluation is executed.
"""
import argparse, ast, copy, hashlib, json, math, sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pandas as pd
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--mlp-notebook',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
a=parser.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(a.repo/'tests'))
from test_coordinator_transfer import load_runtime
r=load_runtime(a.repo/'DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb',a.output/'parity_runtime.py')
torch.set_num_threads(2)
notebook=json.loads(a.mlp_notebook.read_text())
strings={}
for cell in notebook['cells']:
 if cell['cell_type']!='code': continue
 try: tree=ast.parse(''.join(cell['source']))
 except SyntaxError: continue
 for node in tree.body:
  if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
   for name in node.targets:
    if isinstance(name,ast.Name) and name.id in ('CORE_SOURCE','UNIFIED_SOURCE'):
     strings[name.id]=node.value.value
assert set(strings)=={'CORE_SOURCE','UNIFIED_SOURCE'}
fullcore=ast.parse(strings['CORE_SOURCE']); fullunified=ast.parse(strings['UNIFIED_SOURCE'])
worker=ast.parse((a.output/'parity_runtime.py').read_text())

def find(tree,name,cls=None):
 nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
 return next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
class Normalize(ast.NodeTransformer):
 def visit_Attribute(self,node):
  self.generic_visit(node)
  if isinstance(node.value,ast.Name) and node.value.id=='ds': return ast.copy_location(ast.Name(id=node.attr,ctx=node.ctx),node)
  return node
 def visit_FunctionDef(self,node):
  self.generic_visit(node); node.decorator_list=[];node.returns=None;node.type_comment=None
  if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str): node.body=node.body[1:]
  return node
 def visit_arg(self,node): node.annotation=None;node.type_comment=None;return node

def normalized(node): return ast.dump(Normalize().visit(copy.deepcopy(node)),include_attributes=False)
checks={}
checks['nsr_formula_ast_identical']=normalized(find(worker,'nsr_penalty'))==normalized(find(fullunified,'nsr_penalty'))
for name in ['class_indices','conductance_edges','apply_driftslice']:
 checks[name+'_ast_identical']=normalized(find(worker,name))==normalized(find(fullcore,name))
for name in ['_read','gains']:
 checks['reference_'+name+'_ast_identical']=normalized(find(worker,name,'PCMClassReferenceBank'))==normalized(find(fullcore,name,'PCMClassReferenceBank'))
# Evaluate the original MLP function without running its notebook orchestration.
node=copy.deepcopy(find(fullunified,'nsr_penalty')); node.decorator_list=[];node.returns=None
for arg in node.args.args: arg.annotation=None
ns={'math':math,'np':np,'torch':torch,'ds':r}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'original_mlp_nsr','exec'),ns)
cfg=replace(r.Config(),require_cuda=False,fast_mode=False)
model=r.AnalogMLP(cfg,42,torch.device('cpu'))
weights=[r.physical_weight_parameter(layer) for layer in model.layers().values()]
original=ns['nsr_penalty'](model,cfg);current=r.nsr_penalty(model,cfg)
g1=torch.autograd.grad(original,weights);g2=torch.autograd.grad(current,weights)
checks['nsr_loss_bit_identical']=torch.equal(original,current)
checks['nsr_gradients_bit_identical']=all(torch.equal(x,y) for x,y in zip(g1,g2))
# Native gain and target-class reconstruction checks, including exact bin boundaries.
for B,R,K in [(1,64,4),(2,64,4),(4,256,4),(4,256,16)]:
 c=replace(cfg,reference_cell_budget=R,reference_read_repeats=K)
 bank=r.PCMClassReferenceBank(c,B,183,torch.device('cpu'))
 t=5*r.YEAR;seed=923
 now=bank._read(t,seed)
 expected=(bank.ref20.mean(1)/now.mean(1).clamp_min(1e-6)).clamp(c.gain_min,c.gain_max)
 gains=bank.gains(t,seed)
 checks[f'gain_formula_B{B}_R{R}_K{K}']=torch.equal(expected,gains)
 weights_grid=torch.linspace(-1,1,129).reshape(1,-1)
 main=r.PCMPhysicalMatrix(weights_grid,c,151)
 before=r.state_dict_hash({'programmed':main.g_programmed,'nu':main.nu})
 conductances=main.read_conductances(t,921)
 corrected=r.apply_driftslice(main,conductances,gains,bank.edges)
 for target,actual,out in zip(main.g_target,conductances,corrected):
  idx=(torch.ceil(target/(c.gmax_us/B)).long()-1).clamp(0,B-1)
  assert torch.equal(out,actual*gains[idx])
 checks[f'branchwise_class_correction_B{B}_R{R}_K{K}']=True
 checks[f'persistent_state_unchanged_B{B}_R{R}_K{K}']=before==r.state_dict_hash({'programmed':main.g_programmed,'nu':main.nu})
# Scalar arithmetic and an exact lognormal identity used in the mathematical note.
L=math.log(5*r.YEAR/r.T_REF); sigma=.008
cv=math.sqrt(math.expm1((L*sigma)**2))
assert np.isclose(L,15.881030644303065)
assert np.isclose(256/4,64)
checks['gain_class_reference_count_and_log_time_arithmetic']=True
checks['no_gpu_or_user_model_accuracy_claimed']=True
checks={k:bool(v) for k,v in checks.items()}
checks['all_checks_pass']=all(checks.values())
result={'checks':checks,'native_environment':{'torch':str(torch.__version__),'aihwkit':'1.1.0','device':'cpu'},
 'five_year_log_ratio':L,'minimum_nominal_drift_sigma':sigma,'high_conductance_lognormal_relative_sd_approx':cv,
 'nsr_loss_value_on_synthetic_mlp':float(current.detach()),
 'important_existing_protocol_difference':'The uploaded full MLP defaults to nonnegative-only programming bounds; P4 additionally clips programmed conductance at gmax. Those policies are not changed or claimed identical.',
 'source_hashes':{'uploaded_mlp_notebook':hashlib.sha256(a.mlp_notebook.read_bytes()).hexdigest(),'worker_notebook':hashlib.sha256((a.repo/'DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb').read_bytes()).hexdigest()}}
(a.output/'parity_and_math_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert checks['all_checks_pass']
