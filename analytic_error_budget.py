"""Compute original headroom.py analytical error budget on NEW unified checkpoints.

This is an auxiliary model-based analysis, not a physical circuit implementation.
It intentionally preserves the user-supplied headroom formulas while removing
its external fixed paths.  Requires SciPy for Gaussian CDF/PDF.
"""
import argparse
import importlib.util
from pathlib import Path
import torch
import pandas as pd


def load_legacy_headroom():
    path = Path(__file__).parent/'legacy_reference'/'headroom_original.py'
    spec = importlib.util.spec_from_file_location('original_headroom',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Directory containing fold* checkpoints')
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args()
    H=load_legacy_headroom()
    files=sorted(args.run.glob('fold*_best_validation.pt')) + sorted(args.run.glob('fold*_FP32-Direct_analog_checkpoint.pt'))
    if not files:
        raise FileNotFoundError('No unified study checkpoints found in '+str(args.run))
    rows=[]
    for f in files:
        pack=torch.load(f,map_location='cpu',weights_only=False)
        if not (pack.get('analog_epochs_completed') == 154 or pack.get('method')=='FP32-Direct'):
            raise RuntimeError(f'Checkpoint is not an eligible completed method: {f.name}')
        state=pack['model']
        tiles=[tensor.detach().cpu().numpy() for key,tensor in state.items()
               if key.endswith('tile.weight') and isinstance(tensor,torch.Tensor)]
        if not tiles:
            raise RuntimeError(f'No Torch-tile weights in {f.name}; inspect state_dict format')
        for t in (H.DAY,H.YEAR,5*H.YEAR):
            output=H.combine(tiles,t)
            rows.append(dict(checkpoint=f.name,method=pack.get('method'),t_s=t,
                **{k:output[k] for k in ('none','gdc_mean','global_mse','state_oracle','device_oracle')}))
    out=args.out or args.run/'analytical_headroom.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('Saved',out,'for',len(files),'checkpoints')

if __name__=='__main__':main()
