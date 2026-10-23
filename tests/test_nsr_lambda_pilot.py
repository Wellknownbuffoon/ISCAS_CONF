"""Native AIHWKit 1.1.0 CPU checks for matched continuation and exact resume."""
from dataclasses import asdict, replace
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import test_coordinator_transfer as base

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lambda_pilot", ROOT/"nsr_lambda_training_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ForbiddenTestLoader:
    def __iter__(self):
        raise AssertionError("The official test set must never be iterated")


class RecordingLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orders = []

    def __iter__(self):
        order = []
        self.orders.append(order)
        for x, y in super().__iter__():
            order.extend(y.tolist())
            yield x, y


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        fixtures = ROOT.parent/"worker-fix"/"test-release"
        if not fixtures.exists():
            self.skipTest("Native two-epoch HWA fixture unavailable")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.r = base.load_runtime(ROOT/"DriftSlice_ISCAS_P4_worker_NSR_FIXED.ipynb", self.out/"runtime.py")
        self.cfg = replace(self.r.Config(), require_cuda=False, fast_mode=False, strict_154_epochs=False,
            epochs=2, digital_epochs=1, train_repeats=1, realizations=30, retention_points=12,
            batch_size=2, test_batch_size=2, include_original_gdc=False)
        fdir = self.out/"training"/"fold0"
        fdir.mkdir(parents=True)
        self.source_path = fdir/f"{self.r.HWA}_best_validation.pt"
        shutil.copyfile(fixtures/"HWA"/"resumed"/self.source_path.name, self.source_path)
        self.exp = SimpleNamespace(cfg=self.cfg, arch="resnet32", device=torch.device("cpu"),
            train_dir=fdir.parent, run_dir=self.out/"complete", base_dir=self.out,
            fp={"train":"fixture-fp", "retention":"fixture-retention"})
        gen = torch.Generator().manual_seed(108)
        self.dataset = TensorDataset(torch.randn(4, 3, 32, 32, generator=gen), torch.tensor([1, 2, 3, 4]))
        self.loaders = []
        self.ns = dict(self.r.__dict__)

        def loaders(*args, **kwargs):
            train = RecordingLoader(self.dataset, batch_size=2, shuffle=True,
                generator=torch.Generator().manual_seed(55))
            val = DataLoader(self.dataset, batch_size=2, generator=torch.Generator().manual_seed(56))
            self.loaders.append(train)
            return train, val, ForbiddenTestLoader(), {}
        self.ns["build_cifar10_loaders"] = loaders

    def fake_validation(self, r, model, cfg, fold, val, device, realizations):
        # Deterministic function of actual learned physical weights; selection is tested separately.
        w = r["physical_weight_parameter"](next(iter(model.layers().values())))
        value = float(w.detach().abs().mean())
        return [dict(realization=i, t_index=ti, t_s=ts, accuracy=acc+value, loss=1., gain=1., split="val")
                for i in range(realizations) for ti, ts, acc in ((0,20.,90.), (11,157788000.,85.))]

    def run_quiet(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return pilot.run_pilot(self.ns, self.exp, additional_epochs=2, realizations=1,
                evaluation_epochs=(0,1,2), download=False, **kwargs)

    def test_matched_training_exact_resume_and_source_preservation(self):
        source_hash = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        config = asdict(self.cfg)
        saved_rng = torch.get_rng_state().clone()
        with patch.object(pilot, "_retention_validation", self.fake_validation):
            complete = self.run_quiet()
            self.assertTrue(torch.equal(saved_rng, torch.get_rng_state()))
            orders = self.loaders[-1].orders
            self.assertEqual(len(orders), 6)
            self.assertEqual(orders[:2], orders[2:4])
            self.assertEqual(orders[:2], orders[4:6])
            history = pd.read_csv(complete/"training_history.csv")
            self.assertTrue(history.loc[history.nsr_lambda.eq(0), "train_extra"].eq(0).all())
            self.assertTrue(history.loc[history.nsr_lambda.gt(0), "train_extra"].gt(0).all())
            self.assertTrue(history.groupby("nsr_lambda").optimizer_steps.max().eq(4).all())
            self.assertTrue((complete.with_suffix(".zip")).is_file())
            summary = pd.read_csv(complete/"selected_checkpoints.csv")
            self.assertEqual(summary.initial_model_identity.nunique(), 1)
            contract = json.loads((complete/"pilot_contract.json").read_text())
            self.assertFalse(contract["official_test_evaluated"])
            self.assertEqual(contract["reference_cell_budget"],64)
            self.exp.run_dir = self.out/"resumed"

            def stop(fold, coefficient, epoch):
                if coefficient == .1 and epoch == 1:
                    raise InterruptedError("Simulated Colab disconnect")
            with self.assertRaises(InterruptedError):
                self.run_quiet(_after_epoch=stop)
            self.assertTrue(torch.equal(saved_rng, torch.get_rng_state()))
            resumed = self.run_quiet()
            for coefficient in (0., .1, 1.):
                name = f"fold0/lambda_{coefficient:g}/training.slotB.pt"
                a, b = self.r.torch_load(complete/name), self.r.torch_load(resumed/name)
                for key in ("model", "optimizer", "scheduler", "best_model", "history", "validation_records", "loader_rng"):
                    self.assertEqual(self.r.state_dict_hash(a[key]), self.r.state_dict_hash(b[key]), key)
                self.assertGreater(b["first_step_max_weight_change"], 0.)
            # Completed candidates perform no training on repeated invocation.
            again = self.run_quiet()
            self.assertEqual(again, resumed)
            self.assertEqual(self.loaders[-1].orders, [])
            # A readable but foreign latest checkpoint is rejected, not silently reused.
            foreign_path = resumed/"fold0/lambda_0/training.slotB.pt"
            bad = self.r.torch_load(foreign_path)
            bad["pilot_fingerprint"] = "foreign"
            self.r.atomic_torch_save(bad, foreign_path)
            with self.assertRaisesRegex(RuntimeError, "Foreign pilot checkpoint"):
                self.run_quiet()
        self.assertEqual(config, asdict(self.cfg))
        self.assertEqual(source_hash, hashlib.sha256(self.source_path.read_bytes()).hexdigest())

    def test_native_validation_matches_original_retention(self):
        model = self.r.load_analog_resnet_from_pack(self.cfg, self.r.torch_load(self.source_path), self.exp.device)
        val = DataLoader(self.dataset, batch_size=2, generator=torch.Generator().manual_seed(56))
        val.split_name = "val"
        before = self.r.state_dict_hash(model.state_dict())
        rng = torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            actual = pd.DataFrame(pilot._retention_validation(self.ns, model, self.cfg, 0, val, self.exp.device, 1))
        self.assertEqual(before, self.r.state_dict_hash(model.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        test = DataLoader(self.dataset, batch_size=2, generator=torch.Generator().manual_seed(56))
        test.split_name = "test"  # Synthetic data only: native reference requires this label.
        expected, _ = self.r.retention_one_resnet(self.r.HWA, model, self.r.snapshot_analog_resnet(model),
            self.cfg, 0, 0, 1, test, self.exp.device)
        expected = expected[expected.compensation.eq("DS-B1") & expected.t_index.isin((0,11))]
        joined = actual.merge(expected, on="t_index", suffixes=("_pilot", "_native"), validate="one_to_one")
        np.testing.assert_allclose(joined.accuracy_pilot, joined.accuracy_native, rtol=0, atol=0)
        np.testing.assert_allclose(joined.loss_pilot, joined.loss_native, rtol=0, atol=1e-12)
        np.testing.assert_allclose(joined.gain, joined.class_gain_0, rtol=0, atol=1e-12)

    def test_selection_guard_ties_and_epoch_zero_fallback(self):
        baseline = dict(accuracy_20s=90., accuracy_5y=85., epoch=0)
        self.assertTrue(pilot._better(dict(accuracy_20s=89.8, accuracy_5y=86., epoch=5), baseline, baseline, .25))
        self.assertFalse(pilot._better(dict(accuracy_20s=89.7, accuracy_5y=87., epoch=5), baseline, baseline, .25))
        self.assertFalse(pilot._better(dict(accuracy_20s=90., accuracy_5y=84., epoch=10), baseline, baseline, .25))
        self.assertFalse(pilot._better(dict(accuracy_20s=90., accuracy_5y=85., epoch=10), baseline, baseline, .25))


if __name__ == "__main__":
    unittest.main(verbosity=2)
