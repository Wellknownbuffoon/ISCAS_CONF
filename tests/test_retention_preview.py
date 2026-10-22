"""Preview keeps native retention records and full Config; uses a tiny CPU fixture."""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import test_coordinator_transfer as base

ROOT = Path(__file__).resolve().parents[1]
OUT = None
CHECKPOINT = None
MEASUREMENTS = {}
CELL = (ROOT/'retention_preview_cell.py').read_text()


class PreviewTests(unittest.TestCase):
    def test_01_all_worker_configs_science_and_smoke_scope_are_unchanged(self):
        for role in ('P1_worker_HWA', 'P2_worker_DAR', 'P3_worker_DriftAttention', 'P4_worker_NSR'):
            a_path = ROOT/f'DriftSlice_ISCAS_{role}_RETENTION_ONLY.ipynb'
            b_path = ROOT/f'DriftSlice_ISCAS_{role}_RETENTION_PREVIEW.ipynb'
            a = base.load_runtime(a_path, OUT/f'{role}_full.py')
            b = base.load_runtime(b_path, OUT/f'{role}_preview.py')
            self.assertEqual(asdict(a.Config()), asdict(b.Config()))
            self.assertEqual(a.WORKER_MODELS, b.WORKER_MODELS)
            self.assertEqual(a.smoke_scope_identity(), b.smoke_scope_identity())
            self.assertFalse(b.RUN_FULL_TRAINING or b.RUN_FULL_RETENTION or b.RUN_REFERENCE_ABLATION
                             or b.RUN_MECHANISM_DIAGNOSTICS or b.GENERATE_PAPER_RESULTS)
            self.assertTrue(b.REQUIRE_SMOKE_PASS_BEFORE_FULL)
            data = {'fixture': 'same-dataset'}
            x = a.compute_fingerprints_resnet(a.make_config('resnet32', False, a.full_run_id()), data, a.RESNET_SPEC)
            y = b.compute_fingerprints_resnet(b.make_config('resnet32', False, b.full_run_id()), data, b.RESNET_SPEC)
            for stage in ('train', 'retention', 'ablation', 'diagnostics'):
                self.assertEqual(x[stage], y[stage])
            defs = []
            for p in (a_path, b_path):
                found = {}
                for i, c in enumerate(json.loads(p.read_text())['cells']):
                    if c['cell_type'] == 'code':
                        src = ''.join(c['source']);compile(src, f'{role}/{i}', 'exec')
                        for node in ast.parse(src).body:
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                found[node.name] = ast.dump(node, include_attributes=False)
                defs.append(found)
            self.assertEqual(*defs)
            n = json.loads(b_path.read_text())
            self.assertEqual(''.join(n['cells'][-1]['source']), CELL)
        MEASUREMENTS['all_four_workers'] = 'Scientific functions, Config, stage fingerprints and smoke scope unchanged'

    def test_02_preview_native_equality_cache_and_honest_statistics(self):
        if CHECKPOINT is None:
            self.skipTest('--hwa-checkpoint not supplied')
        root = OUT/'actual-native'
        root.mkdir()
        r = base.load_runtime(ROOT/'DriftSlice_ISCAS_P1_worker_HWA_RETENTION_PREVIEW.ipynb', root/'runtime.py')
        torch.set_num_threads(2)
        cfg = r.Config(require_cuda=False, fast_mode=False, strict_154_epochs=False, epochs=2, digital_epochs=1, train_repeats=1,
                       realizations=30, retention_points=12, batch_size=2, test_batch_size=2,
                       validation_reads=1, include_original_gdc=False)
        exp = object.__new__(r.Experiment)
        exp.cfg = cfg;exp.arch = 'resnet32';exp.methods = (r.HWA,);exp.include_direct = False
        exp.write_manifest = False;exp.role_tag = 'worker-HWA';exp.run_dir = root
        exp.train_dir = root/'training';exp.ret_dir = root/'retention';exp.base_dir = root
        exp.device = torch.device('cpu');exp.fp = {'train': 'fixture-fp', 'retention': 'fixture-retention'}
        fdir = exp.train_dir/'fold0';fdir.mkdir(parents=True)
        shutil.copyfile(CHECKPOINT, fdir/f'{r.HWA}_best_validation.pt')
        torch.save({'kind': 'synthetic-placeholder-not-read'}, fdir/'digital_best_validation.pt')
        # Shared readiness guard requires Direct even when this worker evaluates HWA only.
        torch.save({'kind': 'synthetic-placeholder-not-read'}, fdir/f'{r.DIRECT}_analog_checkpoint.pt')
        pack = r.torch_load(fdir/f'{r.HWA}_best_validation.pt')
        selected = exp.ret_dir/'fold0'/'selected_bins.json';selected.parent.mkdir(parents=True)
        selected.write_text(json.dumps({'selection_method': r.HWA, 'selection_model_identity': pack['model_identity'], 'bins': 1}))
        gen = torch.Generator().manual_seed(123)
        ds = TensorDataset(torch.randn(2, 3, 32, 32, generator=gen), torch.tensor([1, 2]))
        val = DataLoader(ds, batch_size=2);val.split_name = 'val'
        test = DataLoader(ds, batch_size=2);test.split_name = 'test'
        model = r.load_analog_resnet_from_pack(cfg, pack, exp.device)
        snap = r.snapshot_analog_resnet(model)
        reference, reference_meta = r.retention_one_resnet(r.HWA, model, snap, cfg, 0, 0, 1, test, exp.device)
        later, later_meta = r.retention_one_resnet(r.HWA, model, snap, cfg, 0, 1, 1, test, exp.device)
        later_path = exp.ret_dir/'fold0'/r.HWA/'real01.pt';later_path.parent.mkdir(parents=True)
        torch.save({'fp': exp.fp['retention'], 'model_identity': pack['model_identity'], 'selected_bins': 1,
                    'results': later, 'meta': later_meta}, later_path)
        later_bytes = later_path.read_bytes()
        checkpoint_bytes = (fdir/f'{r.HWA}_best_validation.pt').read_bytes()
        original_cfg = asdict(cfg);original_fp = dict(exp.fp)
        ops = r.ARCH_OPS['resnet32']
        adapted = SimpleNamespace(**{**vars(ops), 'loaders': lambda *a, **k: (None, val, test, {})})
        context = r.__dict__.copy()
        context.update(OUTPUT_DIR=root, open_full_experiment=lambda: exp, require_smoke_pass=lambda *a: None)
        with patch.dict(r.ARCH_OPS, resnet32=adapted):
            exec(compile(CELL, 'preview-cell', 'exec'), context)
        first = exp.ret_dir/'fold0'/r.HWA/'real00.pt'
        result = r.torch_load(first)
        pd.testing.assert_frame_equal(reference, result['results'], check_exact=True)
        self.assertEqual(reference_meta, result['meta'])
        self.assertEqual(result['fp'], exp.fp['retention'])
        self.assertEqual(asdict(cfg), original_cfg)
        self.assertEqual(exp.fp, original_fp)
        self.assertEqual(later_bytes, later_path.read_bytes())
        self.assertEqual(checkpoint_bytes, (fdir/f'{r.HWA}_best_validation.pt').read_bytes())
        out = context['_ret_out']
        status = json.loads((out/'retention_preview_status.json').read_text())
        self.assertTrue(status['preview']);self.assertFalse(status['complete'])
        self.assertEqual(status['realizations_per_fold'], 1)
        self.assertEqual(status['planned_realizations_per_fold'], 30)
        raw = pd.read_csv(out/'preview_test_accuracy_vs_retention_raw.csv')
        self.assertEqual(raw.realization.unique().tolist(), [0])
        self.assertEqual(len(raw), len(reference))
        first_bytes = first.read_bytes()
        adapted.retention = lambda *a, **k: self.fail('cached native shard was recomputed')
        with patch.dict(r.ARCH_OPS, resnet32=adapted):
            exec(compile(CELL, 'preview-cell', 'exec'), context)
        self.assertEqual(first_bytes, first.read_bytes())
        self.assertEqual(later_bytes, later_path.read_bytes())
        MEASUREMENTS['native_preview'] = {'records': len(raw), 'planned_realizations': 30,
            'preview_realizations': 1, 'bit_identical_to_native_retention': True,
            'cached_result_not_recomputed': True, 'other_realization_and_checkpoint_bytes_unchanged': True,
            'config_and_fingerprints_unchanged': True, 'full_run_status_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hwa-checkpoint', type=Path)
    args = parser.parse_args();OUT = args.out;CHECKPOINT = args.hwa_checkpoint
    OUT.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreviewTests))
    (OUT/'results.json').write_text(json.dumps({'tests_run': result.testsRun, 'errors': len(result.errors),
        'failures': len(result.failures), 'skipped': len(result.skipped), 'cuda_available': torch.cuda.is_available(),
        'measurements': MEASUREMENTS}, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
