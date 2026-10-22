"""Retention-only controls and exports, using synthetic saved shards without training."""
import argparse
import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

import test_coordinator_transfer as base

ROOT = Path(__file__).resolve().parents[1]
OUT = None
MEASUREMENTS = {}
EXPORT = (ROOT/'retention_accuracy_export_cell.py').read_text()


class RetentionOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.r = base.load_runtime(ROOT/'DriftSlice_ISCAS_P1_worker_HWA_RETENTION_ONLY.ipynb', OUT/'runtime.py')

    def test_01_all_notebooks_keep_science_checkpoints_and_smoke_compatibility(self):
        report = {}
        for role in ('P0_coordinator', 'P1_worker_HWA', 'P2_worker_DAR', 'P3_worker_DriftAttention', 'P4_worker_NSR'):
            old_path = ROOT/f'DriftSlice_ISCAS_{role}_FIXED.ipynb'
            new_path = ROOT/f'DriftSlice_ISCAS_{role}_RETENTION_ONLY.ipynb'
            old = json.loads(old_path.read_text())
            new = json.loads(new_path.read_text())
            definitions = []
            for notebook in (old, new):
                defs = {}
                for i, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        source = ''.join(cell['source'])
                        compile(source, f'{role}/cell{i}', 'exec')
                        for node in ast.parse(source).body:
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                defs[node.name] = ast.dump(node, include_attributes=False)
                definitions.append(defs)
            self.assertEqual(*definitions, role)
            a = base.load_runtime(old_path, OUT/f'{role}_old.py')
            b = base.load_runtime(new_path, OUT/f'{role}_new.py')
            self.assertFalse(b.RUN_FULL_TRAINING)
            self.assertTrue(b.RUN_FULL_RETENTION)
            self.assertFalse(b.RUN_REFERENCE_ABLATION)
            self.assertFalse(b.RUN_MECHANISM_DIAGNOSTICS)
            self.assertFalse(b.GENERATE_PAPER_RESULTS)
            self.assertTrue(b.REQUIRE_SMOKE_PASS_BEFORE_FULL)
            self.assertTrue(b.RUN_SMOKE_TEST)
            self.assertEqual(a.WORKER_MODELS, b.WORKER_MODELS)
            self.assertEqual(a.smoke_scope_identity(), b.smoke_scope_identity())
            self.assertEqual(asdict(a.make_config('resnet32', False, a.full_run_id())),
                             asdict(b.make_config('resnet32', False, b.full_run_id())))
            data = {'synthetic-data': 'same-sha'}
            fp_a = a.compute_fingerprints_resnet(a.make_config('resnet32', False, a.full_run_id()), data, a.RESNET_SPEC)
            fp_b = b.compute_fingerprints_resnet(b.make_config('resnet32', False, b.full_run_id()), data, b.RESNET_SPEC)
            for stage in ('train', 'retention', 'ablation', 'diagnostics'):
                self.assertEqual(fp_a[stage], fp_b[stage], (role, stage))
            self.assertEqual(''.join(new['cells'][-1]['source']), EXPORT)
            report[role] = 'Config, stages, scientific functions and smoke scope verified'
        MEASUREMENTS['five_notebooks'] = report

    def fixture(self, label, empty=False):
        r = self.r
        root = OUT/label
        cfg = replace(r.Config(require_cuda=False), train_repeats=3, realizations=2,
                      retention_points=12, include_original_gdc=False)
        exp = SimpleNamespace(cfg=cfg, run_dir=root, ret_dir=root/'retention', train_dir=root/'training',
                              fp={'train': 'train-fixture', 'retention': 'retention-fixture'},
                              models=(r.HWA, r.NSR))
        paths = []
        if not empty:
            for fold in range(cfg.train_repeats):
                for mi, model in enumerate(exp.models):
                    ident = f'{model}:fold{fold}'
                    path = exp.train_dir/f'fold{fold}'/f'{model}_best_validation.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({'kind': 'hwa_best_validation', 'train_fp': exp.fp['train'], 'fold': fold,
                                'method': model, 'model_identity': ident}, path)
                    for real in range(cfg.realizations):
                        rows = []
                        for ci, comp in enumerate(r.compensations_for(cfg)):
                            for ti, ts in enumerate(r.exact_retention_times(cfg)):
                                rows.append({'fold': fold, 'method': model, 'compensation': comp,
                                             'realization': real, 't_index': ti, 't_s': float(ts),
                                             'accuracy': 60 + 5*fold + 2*real + .01*ti + mi + ci,
                                             'split': 'test'})
                        path = exp.ret_dir/f'fold{fold}'/model/f'real{real:02}.pt'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({'fp': exp.fp['retention'], 'model_identity': ident,
                                    'results': pd.DataFrame(rows), 'meta': {}}, path)
                        paths.append(path)
        return exp, paths

    def export(self, exp):
        context = self.r.__dict__.copy()
        context.update(FULL_EXP=exp, PARALLEL_ROLE='worker')
        exec(compile(EXPORT, 'retention_accuracy_export_cell.py', 'exec'), context)
        return context['_ret_out']

    def test_02_complete_export_has_correct_fold_statistics_and_plots(self):
        exp, paths = self.fixture('complete')
        before = {str(p): p.read_bytes() for p in exp.run_dir.rglob('*.pt')}
        out = self.export(exp)
        status = json.loads((out/'retention_export_status.json').read_text())
        self.assertTrue(status['complete'])
        self.assertEqual(status['records_present'], 576)
        summary = pd.read_csv(out/'test_accuracy_vs_retention_summary.csv')
        row = summary[(summary.method == self.r.HWA) & (summary.compensation == 'none') & (summary.t_index == 0)].iloc[0]
        self.assertAlmostEqual(row.test_accuracy_mean_pct, 66.)
        self.assertAlmostEqual(row.test_accuracy_sd_across_folds_pp, 5.)
        self.assertEqual(len(list(out.glob('*.png'))), 4)
        self.assertEqual(len(list(out.glob('*.pdf'))), 4)
        self.assertEqual(before, {str(p): p.read_bytes() for p in exp.run_dir.rglob('*.pt')})
        MEASUREMENTS['complete_export'] = {'records': 576, 'mean_accuracy_pct': 66., 'sd_across_folds_pp': 5.,
                                           'plots_png_pdf': 8, 'checkpoint_and_shard_bytes_unchanged': True}

    def test_03_missing_record_cannot_produce_complete_mean(self):
        exp, paths = self.fixture('missing')
        pack = self.r.torch_load(paths[0])
        pack['results'] = pack['results'].iloc[1:]
        torch.save(pack, paths[0])
        out = self.export(exp)
        status = json.loads((out/'retention_export_status.json').read_text())
        self.assertFalse(status['complete'])
        self.assertEqual(status['missing_records'], 1)
        summary = pd.read_csv(out/'test_accuracy_vs_retention_summary.csv')
        row = summary[(summary.method == self.r.HWA) & (summary.compensation == 'none') & (summary.t_index == 0)].iloc[0]
        self.assertTrue(np.isnan(row.test_accuracy_mean_pct))
        self.assertFalse(row.complete)

    def test_04_empty_retention_does_not_require_ablation_or_diagnostics(self):
        exp, _ = self.fixture('empty', empty=True)
        # There are deliberately no abl_dir/diag_dir attributes or files.
        out = self.export(exp)
        status = json.loads((out/'retention_export_status.json').read_text())
        self.assertFalse(status['complete'])
        self.assertEqual(status['missing_records'], 576)
        summary = pd.read_csv(out/'test_accuracy_vs_retention_summary.csv')
        self.assertTrue(summary.test_accuracy_mean_pct.isna().all())

    def test_05_invalid_input_is_rejected_before_output(self):
        for corruption in ('duplicate', 'foreign_fp', 'foreign_model', 'wrong_time', 'validation', 'accuracy'):
            with self.subTest(corruption=corruption):
                exp, paths = self.fixture(corruption)
                pack = self.r.torch_load(paths[0])
                if corruption == 'duplicate':
                    pack['results'] = pd.concat([pack['results'], pack['results'].iloc[:1]])
                elif corruption == 'foreign_fp':
                    pack['fp'] = 'foreign'
                elif corruption == 'foreign_model':
                    pack['model_identity'] = 'foreign'
                elif corruption == 'wrong_time':
                    pack['results'].loc[0, 't_s'] += 1
                elif corruption == 'validation':
                    pack['results'].loc[0, 'split'] = 'val'
                else:
                    pack['results'].loc[0, 'accuracy'] = float('inf')
                torch.save(pack, paths[0])
                with self.assertRaises(self.r.ScientificCheckError):
                    self.export(exp)
                self.assertFalse((exp.run_dir/'retention_accuracy').exists())

    def test_06_retention_coordinator_refuses_missing_training_before_training_helper(self):
        n = json.loads((ROOT/'DriftSlice_ISCAS_P0_coordinator_RETENTION_ONLY.ipynb').read_text())
        source = next(''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code'
                      and '# coordinator: verify all workers' in ''.join(c['source']))
        exp, _ = self.fixture('missing-training', empty=True)
        exp.methods = exp.models
        calls = []
        context = self.r.__dict__.copy()
        context.update(RUN_FULL_RETENTION=True, PARALLEL_ROLE='coordinator', OUTPUT_DIR=OUT,
                       begin_check_session=lambda *a: None, require_smoke_pass=lambda *a: None,
                       open_full_experiment=lambda **k: exp,
                       coordinator_post_training=lambda *a: calls.append('training helper'))
        with self.assertRaisesRegex(self.r.ScientificCheckError, 'no training was started'):
            exec(compile(source, 'coordinator-retention', 'exec'), context)
        self.assertFalse(calls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RetentionOnlyTests))
    (OUT/'results.json').write_text(json.dumps({'tests_run': result.testsRun, 'errors': len(result.errors),
        'failures': len(result.failures), 'measurements': MEASUREMENTS,
        'cuda_available': torch.cuda.is_available(), 'full_training_or_retention_run': False}, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
