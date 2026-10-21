"""Validate worker notebooks against the corrected coordinator, using synthetic CPU fixtures.

Runs inherited zero-row/corruption tests plus real interrupted training for each selected method.
No notebook installation, Drive, CIFAR downloads or full experiment cells execute.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import test_coordinator_transfer as base

ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT/'DriftSlice_ISCAS_P0_coordinator_FIXED.ipynb'
ORIGINAL = None
MEASUREMENTS = {}


class WorkerTests(base.TransferTests):
    def test_06_all_stage_fingerprints_unchanged(self):
        """All stage fingerprints agree with the coordinator after explicit config alignment."""
        r = self.r
        coordinator = base.load_runtime(COORDINATOR, base.OUT/'coordinator_runtime.py')
        data = {'synthetic-fixture': 'same-dataset-hash'}
        self.assertEqual(r.PARALLEL_ROLE, 'worker')
        self.assertEqual(r.WORKER_MODELS, self.expected_models)
        self.assertEqual(asdict(r.make_config('resnet32', False, r.full_run_id())),
                         asdict(coordinator.make_config('resnet32', False, coordinator.full_run_id())))
        self.assertEqual((r.Config().batch_size, r.Config().test_batch_size,
                          r.Config().epochs, r.Config().train_repeats, r.Config().strict_154_epochs),
                         (64, 128, 100, 3, False))
        # Check every code cell other than the execution flags is identical. This also
        # covers definitions outside the minimal test loader, including paper helpers.
        worker_json = json.loads(base.NOTEBOOK.read_text())
        coord_json = json.loads(COORDINATOR.read_text())
        for i, cell in enumerate(worker_json['cells']):
            if cell['cell_type'] == 'code' and i != 3:
                self.assertEqual(cell['source'], coord_json['cells'][i+1]['source'], f'cell {i}')
        results = {}
        for arch in ('mlp', 'resnet32'):
            cfg = r.make_config(arch, False, r.full_run_id(arch))
            ccfg = coordinator.make_config(arch, False, coordinator.full_run_id(arch))
            fp = (r.compute_fingerprints_resnet(cfg, data, r.RESNET_SPEC) if arch == 'resnet32'
                  else r.compute_fingerprints(cfg, data))
            cfp = (coordinator.compute_fingerprints_resnet(ccfg, data, coordinator.RESNET_SPEC)
                   if arch == 'resnet32' else coordinator.compute_fingerprints(ccfg, data))
            for stage in ('train', 'retention', 'ablation', 'diagnostics'):
                self.assertEqual(fp[stage], cfp[stage], (arch, stage))
            results[arch] = {stage: fp[stage] for stage in ('train', 'retention', 'ablation', 'diagnostics')}
        if ORIGINAL is not None:
            old = base.load_runtime(ORIGINAL, base.OUT/'uploaded_worker_runtime.py')
            old_cfg = old.make_config('resnet32', False, old.full_run_id())
            old_fp = old.compute_fingerprints_resnet(old_cfg, data, old.RESNET_SPEC)
            self.assertNotEqual(old_fp['train'], results['resnet32']['train'])
            # Config is the only change to training identity: training/PCM definitions
            # and hyperparameters other than the five explicit coordinator values stay intact.
            aligned = replace(old_cfg, batch_size=64, test_batch_size=128, epochs=100,
                              train_repeats=3, strict_154_epochs=False)
            self.assertEqual(old.compute_fingerprints_resnet(aligned, data, old.RESNET_SPEC)['train'],
                             results['resnet32']['train'])
        MEASUREMENTS['coordinator_fingerprints'] = results
        MEASUREMENTS['assigned_models'] = list(self.expected_models)

    def test_08_actual_assigned_method_resume(self):
        r = self.r
        cfg = replace(self.cfg, epochs=2, digital_epochs=1, train_repeats=1,
                      batch_size=2, test_batch_size=2, validation_reads=1)
        r.LOCAL_CACHE_STR = str(base.OUT/'local-cache')
        device = torch.device('cpu')
        method = self.expected_models[0]
        digital = r.make_digital_resnet32(42).eval()
        with torch.no_grad():
            digital.stage3[4].conv1.weight[:3] = 3e-11
        parent = r.state_dict_hash(digital.state_dict())
        rng = torch.Generator().manual_seed(123)
        train_ds = TensorDataset(torch.randn(2, 3, 32, 32, generator=rng), torch.tensor([1, 2]))
        val_ds = TensorDataset(torch.randn(2, 3, 32, 32, generator=rng), torch.tensor([3, 4]))

        def loaders():
            train = DataLoader(train_ds, batch_size=2, shuffle=True,
                               generator=torch.Generator().manual_seed(124))
            val = DataLoader(val_ds, batch_size=2)
            train.split_name = 'train'
            val.split_name = 'val'
            return train, val

        resumed = base.OUT/'resumed'
        reference = base.OUT/'uninterrupted'
        r._INTERRUPT_HOOK.update(stage=f'hwa:{method}', after_epoch=1)
        train, val = loaders()
        with self.assertRaises(r.SimulatedInterruption):
            r.train_hwa_resnet(method, cfg, 0, digital, parent, train, val, device, resumed, 'fixture-fp')
        train, val = loaders()
        pack = r.train_hwa_resnet(method, cfg, 0, digital, parent, train, val, device, resumed, 'fixture-fp')
        train, val = loaders()
        ref_pack = r.train_hwa_resnet(method, cfg, 0, digital, parent, train, val, device, reference, 'fixture-fp')
        self.assertEqual(pack['optimizer_steps'], 2)
        self.assertGreater(pack['first_step_max_weight_change'], 0)
        self.assertEqual(pack['history'], ref_pack['history'])
        a = r.torch_load(resumed/f'{method}_final.pt')
        b = r.torch_load(reference/f'{method}_final.pt')
        self.assertEqual(r.state_dict_hash(a['model']), r.state_dict_hash(b['model']))
        if method in (r.NSR, r.DAR, r.ATT):
            self.assertGreater(pack['history'][1]['train_extra'], 0)
        gate_resume_verified = False
        if method == r.ATT:
            # The gate is training-only, so its continuation state lives in the rotating
            # training checkpoint rather than the deployment checkpoint.
            a_training, _ = r.load_rotating(resumed/f'{method}_training.pt')
            b_training, _ = r.load_rotating(reference/f'{method}_training.pt')
            for key in ('gate', 'gate_optimizer', 'optimizer', 'scheduler'):
                self.assertIsNotNone(a_training[key])
                self.assertEqual(r.state_dict_hash(a_training[key]), r.state_dict_hash(b_training[key]), key)
            self.assertTrue(a_training['gate_optimizer']['state'])
            for state in a_training['gate_optimizer']['state'].values():
                self.assertEqual(int(state['step']), 2)
                self.assertTrue(torch.isfinite(state['exp_avg']).all())
                self.assertTrue(torch.isfinite(state['exp_avg_sq']).all())
            gate_resume_verified = True
        before = {p.name: p.read_bytes() for p in resumed.glob('*.pt')}
        train, val = loaders()
        with patch.object(r, 'transfer_pretrained_resnet_to_analog', side_effect=AssertionError('completed model retrained')):
            loaded = r.train_hwa_resnet(method, cfg, 0, digital, parent, train, val, device, resumed, 'fixture-fp')
        self.assertEqual(loaded['model_identity'], pack['model_identity'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in resumed.glob('*.pt')})
        MEASUREMENTS['actual_training'] = {'method': method, 'epochs': 2, 'optimizer_steps': 2,
            'resume_bit_identical_to_uninterrupted': True, 'completed_checkpoint_not_retrained_or_rewritten': True,
            'epoch2_extra_loss': pack['history'][1]['train_extra']}
        if method == r.ATT:
            MEASUREMENTS['actual_training']['gate_and_optimizer_resume_bit_identical'] = gate_resume_verified

    def test_09_worker_manifest_read_only_and_model_ownership(self):
        r = self.r
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            captured['methods'] = args[6]
            return SimpleNamespace(**captured)

        with patch.object(r, 'Experiment', side_effect=capture), \
             patch.dict(r.__dict__, BASE_DIR=base.OUT/'fake-drive', OUTPUT_DIR=base.OUT/'fake-output',
                        DEVICE=torch.device('cpu')):
            exp = r.open_full_experiment()
        self.assertFalse(exp.write_manifest)
        self.assertEqual(exp.methods, (self.expected_models[0],))
        self.assertEqual(exp.include_direct, r.DIRECT in self.expected_models)
        folder = base.OUT/'manifest-check'
        folder.mkdir()
        manifest_path = folder/'manifest.json'
        content = json.dumps({'arch': 'resnet32', 'fingerprints': {'train': 'same-fp'},
                              'coordinator': 'read-only fixture'}).encode()
        manifest_path.write_bytes(content)
        # Exercise the real worker manifest validation without dataset/Drive setup.
        e = object.__new__(r.Experiment)
        e.write_manifest = False
        e.run_dir = folder
        e.run_id = 'fixture'
        e.role_tag = r.role_tag()
        e.arch = 'resnet32'
        e.fp = {'train': 'same-fp', 'env': {}}
        e.include_direct = r.DIRECT in self.expected_models
        e.methods = (self.expected_models[0],)
        e._open_manifest('error')
        self.assertEqual(manifest_path.read_bytes(), content)
        e.fp['train'] = 'foreign-fp'
        with self.assertRaisesRegex(r.ScientificCheckError, 'differs from the coordinator'):
            e._open_manifest('error')
        self.assertEqual(manifest_path.read_bytes(), content)
        MEASUREMENTS['worker_manifest_unchanged_and_foreign_fingerprint_rejected'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--original-hwa', type=Path)
    parser.add_argument('--original-nsr', type=Path)
    parser.add_argument('--original-dar', type=Path)
    parser.add_argument('--original-attention', type=Path)
    parser.add_argument('--workers', nargs='+', choices=('HWA', 'DAR', 'DriftAttention', 'NSR'),
                        default=('HWA', 'NSR'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'aihwkit': __import__('aihwkit').__version__, 'torch': torch.__version__,
              'cuda_available': torch.cuda.is_available(), 'workers': {}}
    success = True
    cases = {
        'HWA': ('P1_worker_HWA', ('PCM-HWA', 'FP32-Direct'), args.original_hwa),
        'DAR': ('P2_worker_DAR', ('PCM-DAR-HWA',), args.original_dar),
        'DriftAttention': ('P3_worker_DriftAttention', ('PCM-DriftAttention-HWA',), args.original_attention),
        'NSR': ('P4_worker_NSR', ('PCM-NSR-HWA',), args.original_nsr),
    }
    for tag in args.workers:
        prefix, models, original = cases[tag]
        base.OUT = args.out/tag
        base.OUT.mkdir()
        base.NOTEBOOK = ROOT/f'DriftSlice_ISCAS_{prefix}_FIXED.ipynb'
        ORIGINAL = original
        WorkerTests.expected_models = models
        MEASUREMENTS.clear()
        base.MEASUREMENTS.clear()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(WorkerTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report['workers'][tag] = {'tests_run': result.testsRun, 'failures': len(result.failures),
                                 'errors': len(result.errors), 'skipped': len(result.skipped),
                                 'measurements': {**base.MEASUREMENTS, **MEASUREMENTS}}
        success &= result.wasSuccessful()
    (args.out/'results.json').write_text(json.dumps(report, indent=2) + '\n')
    sys.exit(not success)
