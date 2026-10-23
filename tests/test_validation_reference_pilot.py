"""Native AIHWKit CPU comparison: the pilot control matches original retention."""
from dataclasses import replace, asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import test_coordinator_transfer as base

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reference_pilot", ROOT/"validation_reference_budget_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ForbiddenLoader:
    def __iter__(self):
        raise AssertionError("Training or test loader was evaluated")


class PilotTests(unittest.TestCase):
    def test_native_control_and_source_preservation(self):
        torch.set_num_threads(2)
        fixtures = ROOT.parent/"worker-fix"/"test-release"
        if not fixtures.exists():
            self.skipTest("Native two-epoch HWA/NSR fixtures are unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            r = base.load_runtime(ROOT/"DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb", out/"runtime.py")
            cfg = replace(r.Config(), require_cuda=False, fast_mode=False, strict_154_epochs=False,
                epochs=2, digital_epochs=1, train_repeats=1, realizations=30, retention_points=12,
                batch_size=2, test_batch_size=2, include_original_gdc=False)
            fdir = out/"training"/"fold0"
            fdir.mkdir(parents=True)
            for tag, method in [("HWA", r.HWA), ("NSR", r.NSR)]:
                shutil.copyfile(fixtures/tag/"resumed"/f"{method}_best_validation.pt",
                                fdir/f"{method}_best_validation.pt")
            exp = SimpleNamespace(cfg=cfg, arch="resnet32", device=torch.device("cpu"),
                train_dir=fdir.parent, run_dir=out, base_dir=out,
                fp={"train":"fixture-fp", "retention":"fixture-retention"})
            gen = torch.Generator().manual_seed(21)
            dataset = TensorDataset(torch.randn(2,3,32,32,generator=gen),torch.tensor([1,2]))
            val = DataLoader(dataset,batch_size=2,generator=torch.Generator().manual_seed(22))
            val.split_name = "val"
            namespace = dict(r.__dict__)
            namespace["build_cifar10_loaders"] = lambda *a, **kw: (ForbiddenLoader(),val,ForbiddenLoader(),{})
            before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fdir.glob("*.pt")}
            before_cfg = asdict(cfg)
            saved_rng = torch.get_rng_state().clone()
            dest = pilot.run_pilot(namespace,exp,realizations=1,download=False)
            self.assertEqual(before_cfg,asdict(cfg))
            self.assertTrue(torch.equal(saved_rng,torch.get_rng_state()))
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fdir.glob("*.pt")})
            records = pd.read_csv(dest/"validation_records.csv")
            self.assertEqual(len(records),8)
            self.assertTrue(records.split.eq("val").all())
            self.assertEqual(set(records.reference_cell_budget),{64,256})
            self.assertTrue(dest.with_suffix(".zip").is_file())
            # Native full retention on identical tiny data supplies an independent control.
            pack=r.torch_load(fdir/f"{r.HWA}_best_validation.pt")
            model=r.load_analog_resnet_from_pack(cfg,pack,exp.device)
            snap=r.snapshot_analog_resnet(model)
            test=DataLoader(dataset,batch_size=2,generator=torch.Generator().manual_seed(22))
            test.split_name="test"
            expected, meta = r.retention_one_resnet(r.HWA,model,snap,cfg,0,0,1,test,exp.device)
            control=records[records.method.eq(r.HWA)&records.reference_cell_budget.eq(64)]
            expected=expected[expected.compensation.eq("DS-B1")&expected.t_index.isin([0,11])]
            check=control.merge(expected,on="t_index",suffixes=("_pilot","_native"),validate="one_to_one")
            np.testing.assert_allclose(check.accuracy_pilot,check.accuracy_native,rtol=0,atol=0)
            np.testing.assert_allclose(check.loss_pilot,check.loss_native,rtol=0,atol=1e-12)
            np.testing.assert_allclose(check.gain,check.class_gain_0,rtol=0,atol=1e-12)
            manifest=json.loads((dest/"pilot_scope.json").read_text())
            self.assertFalse(manifest["new_training_executed"])
            self.assertFalse(manifest["official_test_evaluated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
