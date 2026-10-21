"""Regression of the uploaded coordinator's zero-row transfer failure.

Uses AIHWKit 1.1.0 and definition cells only. No datasets, Drive, setup or full runs.
Run with --original-notebook PATH to also check fingerprint compatibility.
"""
import argparse
import contextlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb'
ORIGINAL = None
OUT = None
MEASUREMENTS = {}


def load_runtime(path, destination):
    n = json.loads(Path(path).read_text())
    indices = [4, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 38, 42]
    parts = ['from __future__ import annotations\n']
    for i in indices:
        parts.append(''.join(n['cells'][i]['source']).replace('from __future__ import annotations\n', ''))
    destination.write_text('\n\n'.join(parts))
    spec = importlib.util.spec_from_file_location(destination.stem, destination)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(runtime)
    return runtime


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.r = load_runtime(NOTEBOOK, OUT/'fixed_runtime.py')
        cls.cfg = cls.r.Config(require_cuda=False, fast_mode=True)

    def fixture(self, device='cpu'):
        r = self.r
        digital = r.make_digital_resnet32(17).to(device).eval()
        with torch.no_grad():
            # Realistic collapsed output filters, exact zero, a whole zero conv and fc row.
            digital.stage3[4].conv1.weight[:3] = 3e-11
            digital.stem.weight[0] = 0
            digital.stage2[1].conv1.weight.zero_()
            digital.fc.weight[0].zero_()
        analog = r.transfer_pretrained_resnet_to_analog(self.cfg, digital, torch.device(device)).eval()
        return digital, analog

    def make_pack(self, digital, analog):
        r = self.r
        state = r.cpu_cloned_state(analog)
        parent = r.state_dict_hash(digital.state_dict())
        return {'kind': 'fp32_direct', 'train_fp': 'synthetic-fp', 'fold': 0,
                'model': state, 'model_identity': r.state_dict_hash(state),
                'digital_identity': parent, 'analog_epochs': 0, 'optimizer_steps': 0,
                # Historical report was wrong. Deliberately keep it to test resume.
                'transfer_errors': [{'layer': 'old', 'tensor': 'tile_weight_row_absmax_minus_omega',
                                     'max_abs_error': .8, 'rms_error': float('nan'), 'max_abs_value': .8}]}

    def verify(self, digital, pack, arch='resnet32'):
        exp = SimpleNamespace(cfg=self.cfg, arch=arch, methods=())
        self.r.verify_training_fold(exp, 0, {self.r.DIRECT: pack}, pack['digital_identity'], digital)

    def test_01_native_aihwkit_zero_rows_and_correct_report(self):
        r = self.r
        digital, analog = self.fixture()
        layer = analog.stage3[4].conv1
        tile, = layer.analog_tiles()
        self.assertTrue(torch.all(tile.tile.weight[:3] == 0))
        self.assertTrue(torch.all(tile.get_mapping_scales()[:3] == 1))
        # Original diagnostic falsely rejects this valid mapping with error = omega.
        original_error = float((tile.tile.weight.detach().abs().amax(1) - self.cfg.weight_scaling_omega).abs().max())
        self.assertAlmostEqual(original_error, .8, places=6)
        report = r.transfer_error_report_resnet(analog, digital, self.cfg.weight_scaling_omega)
        mapping = [row for row in report if row['tensor'].startswith('tile_')]
        self.assertEqual(len(mapping), 68)
        self.assertLess(max(row['max_abs_error'] for row in mapping), 1e-5)
        zeros = sum(row['zero_rows'] for row in mapping if row['tensor'].startswith('tile_weight'))
        self.assertGreaterEqual(zeros, 37)
        MEASUREMENTS['native_resnet_zero_rows'] = {
            'old_false_failure_error': original_error, 'new_mapping_max_error': max(row['max_abs_error'] for row in mapping),
            'zero_rows': zeros, 'analog_layers': 34}

    def test_02_resume_ignores_stale_report_preserves_checkpoint(self):
        r = self.r
        digital, analog = self.fixture()
        pack = self.make_pack(digital, analog)
        path = OUT/'saved_direct.pt'
        torch.save(pack, path)
        before = path.read_bytes()
        loaded = r.torch_load(path)
        rng_before = torch.get_rng_state().clone()
        self.verify(digital, loaded)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(loaded['transfer_errors'][0]['max_abs_error'], .8)
        MEASUREMENTS['saved_checkpoint_rechecked_without_rewrite'] = True

    def test_03_corruption_is_rejected(self):
        r = self.r
        for corruption in ('zero_row', 'active_row', 'mapping_scale', 'bias', 'batchnorm', 'nan'):
            with self.subTest(corruption=corruption):
                digital, analog = self.fixture()
                tile, = analog.stage3[4].conv1.analog_tiles()
                with torch.no_grad():
                    if corruption == 'zero_row':
                        tile.tile.weight[0, 0] = .1
                    elif corruption == 'active_row':
                        tile.tile.weight[3].mul_(.5)
                    elif corruption == 'mapping_scale':
                        tile.get_mapping_scales()[0] = 2.
                    elif corruption == 'bias':
                        fc_tile, = analog.fc.analog_tiles()
                        fc_tile.bias.add_(1.)
                    elif corruption == 'batchnorm':
                        analog.stem_bn.running_mean.add_(1.)
                    else:
                        tile.tile.weight[0, 0] = float('nan')
                # Rehash so tensor/BN checks, rather than the identity guard, must detect corruption.
                pack = self.make_pack(digital, analog)
                with self.assertRaises((r.ScientificCheckError, RuntimeError, AssertionError)):
                    self.verify(digital, pack)

    def test_04_identity_guard(self):
        r = self.r
        digital, analog = self.fixture()
        pack = self.make_pack(digital, analog)
        pack['model_identity'] = 'foreign'
        with self.assertRaisesRegex(r.ScientificCheckError, 'identity mismatch'):
            self.verify(digital, pack)

    def test_05_mlp_mapping_with_zero_row(self):
        r = self.r
        digital = r.DigitalMLP().eval()
        with torch.no_grad():
            digital.fc1.weight[0].zero_()
        analog = r.transfer_pretrained_digital_to_analog(self.cfg, digital, 17, torch.device('cpu')).eval()
        rng_before = torch.get_rng_state().clone()
        self.verify(digital, self.make_pack(digital, analog), arch='mlp')
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_06_all_stage_fingerprints_unchanged(self):
        if ORIGINAL is None:
            self.skipTest('--original-notebook not supplied')
        r = self.r
        old = load_runtime(ORIGINAL, OUT/'original_runtime.py')
        # Colab defines notebook objects in __main__. Emulate that for its smoke hash;
        # imported-module objects would otherwise be intentionally excluded.
        notebook_objects = [obj for runtime in (old, r) for obj in vars(runtime).values()
                            if (inspect.isfunction(obj) or inspect.isclass(obj))
                            and getattr(obj, '__module__', None) == runtime.__name__]
        original_modules = [(obj, obj.__module__) for obj in notebook_objects]
        self.addCleanup(lambda: [setattr(obj, '__module__', name) for obj, name in original_modules])
        for obj in notebook_objects:
            obj.__module__ = '__main__'
        results = {}
        for arch in ('mlp', 'resnet32'):
            args = ({'synthetic-data': 'same-data'},)
            if arch == 'resnet32':
                before = old.compute_fingerprints_resnet(old.Config(require_cuda=False), *args, old.RESNET_SPEC)
                after = r.compute_fingerprints_resnet(r.Config(require_cuda=False), *args, r.RESNET_SPEC)
            else:
                before = old.compute_fingerprints(old.Config(require_cuda=False), *args)
                after = r.compute_fingerprints(r.Config(require_cuda=False), *args)
            for stage in ('train', 'retention', 'ablation', 'diagnostics'):
                self.assertEqual(before[stage], after[stage], (arch, stage))
            self.assertNotEqual(before['code_hashes']['notebook_all_code'], after['code_hashes']['notebook_all_code'])
            results[arch] = {key: after[key] for key in ('train', 'retention', 'ablation', 'diagnostics')}
        MEASUREMENTS['unchanged_stage_fingerprints'] = results

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in validation environment')
    def test_07_cuda_zero_rows(self):
        digital, analog = self.fixture('cuda')
        self.verify(digital, self.make_pack(digital, analog))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-notebook', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args, rest = parser.parse_known_args()
    ORIGINAL = args.original_notebook
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TransferTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'aihwkit': __import__('aihwkit').__version__, 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(), 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'measurements': MEASUREMENTS}
    (OUT/'results.json').write_text(json.dumps(report, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
