"""Sixteen lightweight research-contract tests. Full 30+154 training is NOT run.

Run with the already installed environment; no package installation is performed:
python tests/test_pipeline.py --out /path/to/a/new/verification-directory
"""
import argparse
import contextlib
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from aihwkit.optim import AnalogAdam
import driftslice_core as ds
import unified_experiment as ue
from native_pcm_probe import native_comparison

OUTPUT = None


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def loaders():
    g = torch.Generator().manual_seed(417)
    data = TensorDataset(torch.rand(24, 1, 28, 28, generator=g),
                         torch.randint(0, 10, (24,), generator=g))
    validation = TensorDataset(torch.rand(8, 1, 28, 28, generator=g),
                               torch.randint(0, 10, (8,), generator=g))
    return (DataLoader(data, batch_size=8, shuffle=True,
                       generator=torch.Generator().manual_seed(418)),
            DataLoader(validation, batch_size=8, shuffle=False))


class PipelineTests(unittest.TestCase):
    metrics = {}

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.device = torch.device('cpu')
        cls.cfg = ds.Config(fast_mode=True, require_cuda=False).resolved()
        cls.root = OUTPUT / 'fixtures'
        cls.root.mkdir(parents=True, exist_ok=False)
        ue.enable_source_provenance()
        cls.fingerprint = ds.scientific_source_hash()
        cls.train, cls.val = loaders()
        cls.digital_dir = cls.root / 'digital'
        cls.digital_dir.mkdir()
        with (OUTPUT / 'fixture_training.log').open('w') as log, contextlib.redirect_stdout(log):
            cls.digital, cls.digital_history = ds.train_digital_pretrained(
                cls.cfg, 0, cls.train, cls.val, cls.device, cls.digital_dir, cls.fingerprint)
            cls.models, cls.paths = {}, {}
            for method in (ue.HWA, ue.ATT, ue.ATT_ONLY, ue.GAIN_ONLY):
                p = cls.root / method
                p.mkdir()
                train, val = loaders()
                model, history, metadata = ue.train_hwa_all(
                    method, cls.cfg, 0, cls.digital, train, val, cls.device, p, cls.fingerprint)
                cls.models[method], cls.paths[method] = model, p
        cls.metrics['fixture_scope'] = 'SMOKE TEST: synthetic 24 train/8 validation; 2+2 epochs; CPU'
        cls.metrics['full_training'] = 'NOT EXECUTED (explicitly prohibited); guards and shortened loops tested'

    def newdir(self, name):
        path = self.root / name
        path.mkdir(exist_ok=False)
        return path

    def test_01_fp32_epochs_and_selection(self):
        full = ds.Config(require_cuda=False).resolved()
        self.assertEqual((full.digital_epochs, full.epochs), (30, 154))
        with self.assertRaises(ValueError):
            replace(full, digital_epochs=31).resolved()
        selected = load(self.digital_dir / 'fold0_digital_pretrained.pt')
        final = load(self.digital_dir / 'fold0_digital_final.pt')
        self.assertEqual(selected['epochs_completed'], 2)
        self.assertEqual(selected['selection_split'], 'validation_only')
        rows = selected['history']
        best = min(rows, key=lambda r: (-r['val_accuracy'], r['epoch']))
        self.assertEqual(selected['selected_epoch'], best['epoch'])
        self.assertEqual(final['checkpoint_epoch'], 2)
        self.assertEqual(selected['optimizer_steps'], 6)
        self.assertNotEqual(selected['initial_state_sha256'], selected['selected_state_sha256'])
        self.metrics['fp32'] = {k: selected[k] for k in (
            'epochs_completed', 'selected_epoch', 'optimizer_steps', 'selected_state_sha256')}

    def test_02_transfer_weight_and_bias(self):
        model = ds.transfer_pretrained_digital_to_analog(self.cfg, self.digital, 42, self.device)
        ue.ensure_same_pretrained_weights(model, self.digital)
        for name, layer in model.layers().items():
            tile, = list(layer.analog_tiles())
            self.assertTrue(tile.digital_bias)
            self.assertLessEqual(model.transfer_report[name]['weight_max_abs_error'], 2e-4)
            self.assertLessEqual(model.transfer_report[name]['bias_max_abs_error'], 2e-4)
        self.metrics['transfer_errors'] = model.transfer_report

    def test_03_no_post_transfer_reinitialization(self):
        before = ds.state_fingerprint(self.digital)
        for method, p in self.paths.items():
            saved = load(p / f'fold0_{method}_training.pt')
            self.assertEqual(saved['pretrained_provenance']['selected_state_sha256'], before)
            initial = ds.transfer_pretrained_digital_to_analog(self.cfg, self.digital, 42, self.device)
            for name, layer in initial.layers().items():
                torch.testing.assert_close(saved['initial_refs'][name], ds.physical_weight_parameter(layer),
                                           atol=0, rtol=0)
        self.assertEqual(ds.state_fingerprint(self.digital), before)

    def test_04_analogadam_groups_updates_and_decay(self):
        model = ue.direct_baseline(self.cfg, self.digital, 42, self.device)
        opt, weights = ue.make_hwa_optimizer(model, self.cfg)
        self.assertIsInstance(opt, AnalogAdam)
        grouped = [id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        self.assertTrue(all(g['weight_decay'] == 0 for g in opt.param_groups))
        old = [w.detach().clone() for w in weights]
        model.train()
        x, y = next(iter(self.train))
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        gradients = [float(w.grad.norm()) for w in weights]
        opt.step()
        deltas = [float((w.detach()-v).abs().max()) for w, v in zip(weights, old)]
        self.assertTrue(all(d > 0 for d in deltas))
        before_decay = {id(p): p.detach().clone() for p in model.parameters()}
        ue.decay_analog_weights(opt, weights, self.cfg)
        ids = {id(w) for w in weights}
        for p in model.parameters():
            expected = before_decay[id(p)]
            if id(p) in ids:
                group, = [g for g in opt.param_groups if any(p is q for q in g['params'])]
                expected = expected * (1 - group['lr'] * self.cfg.weight_decay)
            torch.testing.assert_close(p, expected, atol=0, rtol=0)
        self.metrics['analogadam'] = {'class': type(opt).__name__, 'groups': len(opt.param_groups),
                                     'weight_gradient_norms': gradients, 'weight_max_updates': deltas,
                                     'manual_decay_only_on_physical_weights': True}

    def test_05_full_mode_cannot_complete_early(self):
        with self.assertRaises(ValueError):
            ds.Config(epochs=153, strict_154_epochs=False).resolved()
        for length in (0, 1, 153, 155):
            with self.assertRaises(RuntimeError):
                ds.validate_epoch_history([{'epoch': i+1} for i in range(length)], 154, 'HWA')
        ds.validate_epoch_history([{'epoch': i+1} for i in range(154)], 154, 'HWA')
        # Runtime resume-cursor guard executes before any further optimizer step.
        p = self.newdir('bad_cursor')
        saved = load(self.paths[ue.HWA] / f'fold0_{ue.HWA}_training.pt')
        saved['next_epoch'] = 1
        torch.save(saved, p / f'fold0_{ue.HWA}_training.pt')
        train, val = loaders()
        with self.assertRaisesRegex(RuntimeError, 'cursor/history'):
            ue.train_hwa_all(ue.HWA, self.cfg, 0, self.digital, train, val,
                             self.device, p, self.fingerprint)

    def test_06_final_and_best_checkpoint_distinction(self):
        p = self.newdir('forced_validation_selection')
        train, val = loaders()
        # Controlled selection fixture, not claimed MNIST accuracy measurements.
        with patch.object(ue, 'validation_score', side_effect=[(80., .5), (70., .8)]):
            ue.train_hwa_all(ue.HWA, self.cfg, 0, self.digital, train, val,
                             self.device, p, self.fingerprint)
        final = load(p / f'fold0_{ue.HWA}_final.pt')
        best = load(p / f'fold0_{ue.HWA}_best_validation.pt')
        self.assertEqual(final['checkpoint_epoch'], 2)
        self.assertEqual(best['checkpoint_epoch'], 1)
        for saved in (final, best):
            self.assertEqual(saved['selected_best_validation_epoch'], 1)
            self.assertEqual(saved['analog_epochs_completed'], 2)
            self.assertEqual(saved['optimizer_state_epoch'], 2)
            self.assertIn('state', saved['optimizer'])
            self.assertIn('last_epoch', saved['scheduler'])
        self.assertTrue(any(not torch.equal(final['model'][k], best['model'][k])
                            for k in final['model'] if isinstance(final['model'][k], torch.Tensor)))

    def test_07_fp32_direct_zero_hwa_steps(self):
        with patch.object(ue, 'make_hwa_optimizer', side_effect=AssertionError('Direct constructed HWA optimizer')):
            direct = ue.direct_baseline(self.cfg, self.digital, 42, self.device)
            ue.ensure_same_pretrained_weights(direct, self.digital)
            snapshot = ds.snapshot_analog_model(direct)
            pcm = ds.build_main_pcm(snapshot, self.cfg, 500)
            w, _ = ds.plain_effective_weights(pcm, ds.YEAR, 501)
            self.assertTrue(torch.isfinite(ds.SnapshotMLP(snapshot, w, self.cfg)(next(iter(self.val))[0])).all())

    def test_08_pcm_programming_and_native_comparison(self):
        w = torch.linspace(-1, 1, 512).reshape(16, 32)
        m = ds.PCMPhysicalMatrix(w, self.cfg, 44)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(m.g_target, m.g_programmed)))
        self.assertTrue(all((g >= 0).all() for g in m.g_programmed))
        reconstructed = m.logical_weight(m.g_target)
        torch.testing.assert_close(reconstructed, w, atol=1e-7, rtol=1e-6)
        above = torch.tensor([self.cfg.gmax_us + 2.])
        torch.testing.assert_close(ds.bound_programmed_conductance(above, self.cfg), above)
        self.assertEqual(float(ds.bound_programmed_conductance(
            above, replace(self.cfg, pcm_conductance_bounds='legacy_gmax_clip'))), self.cfg.gmax_us)
        self.metrics['native_vs_custom'] = native_comparison()
        self.metrics['programmed_above_gmax_count'] = sum(int((g > self.cfg.gmax_us).sum()) for g in m.g_programmed)

    def test_09_drift_and_persistent_programmed_state(self):
        m = ds.PCMPhysicalMatrix(torch.linspace(-1, 1, 96).reshape(8, 12), self.cfg, 9)
        before = [g.clone() for g in (*m.g_programmed, *m.nu)]
        g0, g1 = m.read_conductances(20., 8, noiseless=True), m.read_conductances(ds.YEAR, 9, noiseless=True)
        self.assertTrue(any((a-b).abs().max() > 0 for a, b in zip(g0, g1)))
        for a, b in zip(before, (*m.g_programmed, *m.nu)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for a, b in zip(g1, m.read_conductances(ds.YEAR, 9, noiseless=True)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_10_reference_time_convention(self):
        self.assertEqual(ds.elapsed_pcm_time(20.), 0.)
        self.assertEqual(ds.elapsed_pcm_time(40.), 20.)
        with self.assertRaises(ValueError):
            ds.elapsed_pcm_time(19.)
        cfg = replace(self.cfg, read_noise_scale=0.)
        m = ds.PCMPhysicalMatrix(torch.tensor([[.2, -.8]]), cfg, 123)
        for programmed, nu, at20, at40 in zip(m.g_programmed, m.nu,
                m.read_conductances(20., 1), m.read_conductances(40., 1)):
            torch.testing.assert_close(at20, programmed)
            torch.testing.assert_close(at40, programmed * 2 ** (-nu))

    def test_11_snapshot_matches_analog_no_drift(self):
        model = self.models[ue.HWA].eval()
        snap = ds.snapshot_analog_model(model)
        deployed = ds.SnapshotMLP(snap, {k: v['weight'] for k, v in snap.items()}, self.cfg).eval()
        x, _ = next(iter(self.val))
        with torch.inference_mode(), ds.random_stream(771, self.device):
            expected = model(x)
        with torch.inference_mode(), ds.random_stream(771, self.device):
            actual = deployed(x)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
        self.metrics['snapshot_output_max_error'] = float((actual-expected).abs().max())
        bad_weights = {k: v['weight'].clone() for k, v in snap.items()}
        bad_weights['fc1'].fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            ds.evaluate_snapshot(ds.SnapshotMLP(snap, bad_weights, self.cfg), self.val, self.device)

    def test_12_attention_gainonly_raw_deployment(self):
        self.assertEqual(self.cfg.input_intervention_semantics, 'training_only_raw_deployment')
        x, _ = next(iter(self.val)); x = x.flatten(1)
        gate = ue.AttentionGate(cg=self.cfg.attention_input_gain)
        for method in (ue.ATT, ue.ATT_ONLY, ue.GAIN_ONLY):
            augmented, alpha = ue.training_input(x, method, self.cfg, gate)
            self.assertFalse(torch.equal(augmented, x))
            if method == ue.GAIN_ONLY:
                torch.testing.assert_close(augmented, x * self.cfg.attention_input_gain)
            model = self.models[method].eval()
            snap = ds.snapshot_analog_model(model)
            deployed = ds.SnapshotMLP(snap, {k: v['weight'] for k, v in snap.items()}, self.cfg).eval()
            with torch.inference_mode(), ds.random_stream(155, self.device):
                raw_validation = model(x)
            with torch.inference_mode(), ds.random_stream(155, self.device):
                raw_deployment = deployed(x)
            torch.testing.assert_close(raw_validation, raw_deployment, atol=3e-5, rtol=3e-4)
            saved = load(self.paths[method] / f'fold0_{method}_best_validation.pt')
            if method != ue.GAIN_ONLY:
                self.assertIsNotNone(saved['gate'])
            self.assertFalse(any(k.startswith('gate') for k in saved['model']))

    def test_13_gdc_clipping_and_independent_safe_reference(self):
        w = torch.full((3, 64), .8)
        snap = {'bias': torch.full((3,), 100.), 'scale': torch.tensor([.8, 1., 1.2]),
                'input_range': torch.tensor(.5)}
        io = ds.make_hwa_rpu(self.cfg).forward
        legacy, legacy_diag = ds.global_gdc_gain(w, w*.6, snap, io, 42, self.device, True)
        safe, safe_diag = ds.safe_global_gdc_gain(w, w*.6, snap, io, 42, self.device)
        self.assertEqual(legacy_diag['reference_input_clip_fraction'], 1.)
        self.assertEqual(legacy_diag['reference_adc_clip_fraction'], 1.)
        self.assertEqual(safe_diag['reference_input_clip_fraction'], 0.)
        self.assertEqual(safe_diag['reference_adc_clip_fraction'], 0.)
        continuous_io = copy.deepcopy(io)
        continuous_io.inp_res = continuous_io.out_res = continuous_io.out_noise = 0.
        continuous, _ = ds.safe_global_gdc_gain(w, w*.6, snap, continuous_io, 42, self.device)
        ideal = ds.ideal_global_gdc_gain(w, w*.6, snap)
        self.assertAlmostEqual(continuous, 1/.6, places=5)
        self.assertAlmostEqual(ideal, 1/.6, places=5)
        # Preserve the historical algorithm/RNG exactly on a noisy, nonsaturated case.
        mixed = torch.linspace(-.05, .07, 192).reshape(3, 64)
        zero_bias = dict(snap, bias=torch.zeros_like(snap['bias']))
        ones = torch.ones(1, 64)
        with torch.inference_mode(), ds.random_stream(912, self.device):
            before = torch.stack([ds.shared_mvm(ones, mixed, zero_bias, io).abs().mean()
                                  for _ in range(4)]).mean()
            now = torch.stack([ds.shared_mvm(ones, mixed*.6, zero_bias, io).abs().mean()
                               for _ in range(4)]).mean()
        observed = ds.global_gdc_gain(mixed, mixed*.6, snap, io, 912, self.device)
        self.assertEqual(observed, float(before/now))
        self.metrics['gdc_controlled_fixture'] = {'legacy_gain': legacy, 'safe_measured_gain': safe,
            'ideal_gain': ideal, 'legacy_diagnostics': legacy_diag, 'safe_diagnostics': safe_diag}

    def test_14_b1_versus_class_dependent_behavior(self):
        cfg = replace(self.cfg, programming_noise_scale=0., read_noise_scale=0.)
        b1 = ds.PCMClassReferenceBank(cfg, 1, 43, self.device)
        b4 = ds.PCMClassReferenceBank(cfg, 4, 43, self.device)
        self.assertEqual(b1.target.numel(), cfg.reference_cell_budget)
        self.assertEqual(b4.target.numel(), cfg.reference_cell_budget)
        g1, g4 = b1.gains(ds.YEAR, 44), b4.gains(ds.YEAR, 44)
        self.assertEqual(g1.numel(), 1)
        self.assertEqual(g4.numel(), 4)
        self.assertGreater(float(g4.std()), 0)
        matrix = ds.PCMPhysicalMatrix(torch.linspace(-1., 1., 96).reshape(8, 12), cfg, 88)
        current = matrix.read_conductances(ds.YEAR, 89)
        corrected = ds.apply_driftslice(matrix, current, g4, b4.edges)
        for target, now, actual in zip(matrix.g_target, current, corrected):
            torch.testing.assert_close(actual, now * g4[ds.class_indices(target, b4.edges)])
        self.metrics['reference_banks'] = {'B1_gain': g1.tolist(), 'B4_gains': g4.tolist(),
                                         'fixed_device_budget': cfg.reference_cell_budget}

    def test_15_exact_retention_grid_and_protocol_counts(self):
        full = ds.Config(require_cuda=False).resolved()
        t = ue.exact_retention_times(full)
        self.assertEqual((full.train_repeats, full.realizations, len(t)), (5, 30, 75))
        self.assertEqual(len(np.unique(t)), 75)
        self.assertEqual((t[0], t[-1]), (20., 5*ds.YEAR))
        anchors = [20., 60., 1800., 3600., ds.DAY, 5*ds.DAY, 30*ds.DAY,
                   180*ds.DAY, ds.YEAR, 2*ds.YEAR, 3*ds.YEAR, 5*ds.YEAR]
        self.assertTrue(all(a in t for a in anchors))
        self.assertEqual(len(ue.exact_retention_times(self.cfg)), 8)
        self.metrics['full_grid'] = {'count': len(t), 'first_s': t[0], 'last_s': t[-1], 'anchors_s': anchors}

    def test_16_resume_and_source_provenance(self):
        p = self.newdir('interrupted_attention')
        original_save = ds.atomic_torch_save

        class Interrupted(Exception):
            pass

        def stop_after_epoch_one(obj, path):
            original_save(obj, path)
            if str(path).endswith('_training.pt') and obj.get('next_epoch') == 1:
                raise Interrupted('Simulated interruption after durable checkpoint')

        digital_path = self.newdir('interrupted_digital')
        train, val = loaders()
        with patch.object(ds, 'atomic_torch_save', side_effect=stop_after_epoch_one):
            with self.assertRaises(Interrupted):
                ds.train_digital_pretrained(self.cfg, 0, train, val, self.device,
                                             digital_path, self.fingerprint)
        train, val = loaders()
        resumed_digital, _ = ds.train_digital_pretrained(
            self.cfg, 0, train, val, self.device, digital_path, self.fingerprint)
        self.assertEqual(ds.state_fingerprint(resumed_digital), ds.state_fingerprint(self.digital))
        digital_final = load(digital_path/'fold0_digital_final.pt')
        original_final = load(self.digital_dir/'fold0_digital_final.pt')
        for key, value in digital_final['model'].items():
            torch.testing.assert_close(value, original_final['model'][key], atol=0, rtol=0)
        self.assertEqual(digital_final['history'], original_final['history'])

        train, val = loaders()
        with patch.object(ds, 'atomic_torch_save', side_effect=stop_after_epoch_one):
            with self.assertRaises(Interrupted):
                ue.train_hwa_all(ue.ATT, self.cfg, 0, self.digital, train, val,
                                 self.device, p, self.fingerprint)
        train, val = loaders()
        ue.train_hwa_all(ue.ATT, self.cfg, 0, self.digital, train, val,
                         self.device, p, self.fingerprint)
        resumed = load(p / f'fold0_{ue.ATT}_final.pt')
        uninterrupted = load(self.paths[ue.ATT] / f'fold0_{ue.ATT}_final.pt')
        for key, value in resumed['model'].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, uninterrupted['model'][key], atol=0, rtol=0)
        for key, value in resumed['gate'].items():
            torch.testing.assert_close(value, uninterrupted['gate'][key], atol=0, rtol=0)
        self.assertEqual(resumed['history'], uninterrupted['history'])
        train, val = loaders()
        with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
            ue.train_hwa_all(ue.ATT, self.cfg, 0, self.digital, train, val,
                             self.device, p, 'changed-code-fingerprint')
        base = self.newdir('manifest_source_guard')
        cfg = replace(self.cfg, output_root='runs', run_id='guard', resume_policy='error')
        out, first = ds.experiment_directory(cfg, base)
        manifest_before = (out/'manifest.json').read_bytes()
        source_before = ds.scientific_source_hash()
        fake = self.root/'changed_unified_experiment.py'
        fake.write_text(Path(ue.__file__).read_text()+'\n# changed scientific source\n')
        with patch.object(ue, '__file__', str(fake)):
            self.assertNotEqual(source_before, ds.scientific_source_hash())
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                ds.experiment_directory(cfg, base)
        with self.assertRaisesRegex(ValueError, 'contract changed'):
            ds.experiment_directory(replace(cfg, read_noise_scale=0.), base)
        with self.assertRaisesRegex(ValueError, 'contract changed'):
            ds.experiment_directory(cfg, base, data_identity={'changed_dataset': 'different_sha256'})
        self.assertEqual((out/'manifest.json').read_bytes(), manifest_before)
        self.metrics['resume'] = 'Interrupted digital and attention runs match uninterrupted final weights, selected FP32 state, gate, and histories exactly on CPU'


class ReportingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statuses = {}

    def addSuccess(self, test):
        super().addSuccess(test)
        self.statuses[test._testMethodName] = 'PASS'

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.statuses[test._testMethodName] = 'FAIL'

    def addError(self, test, err):
        super().addError(test, err)
        self.statuses[getattr(test, '_testMethodName', str(test))] = 'FAIL'

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.statuses[test._testMethodName] = 'NOT EXECUTED'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Must be a new directory')
    args = parser.parse_args()
    OUTPUT = args.out.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests)
    with (OUTPUT/'unittest.log').open('w') as log, contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2,
                                        resultclass=ReportingResult).run(suite)
    tests = []
    for name in unittest.defaultTestLoader.getTestCaseNames(PipelineTests):
        tests.append({'test': name, 'status': result.statuses.get(name, 'NOT EXECUTED')})
    report = {'tests': tests, 'tests_run': result.testsRun,
              'all_passed': result.wasSuccessful(), 'measurements': PipelineTests.metrics,
              'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'full_30_plus_154_training': 'NOT EXECUTED', 'scope': 'SMOKE TEST / small controlled fixtures'}
    (OUTPUT/'verification_results.json').write_text(json.dumps(report, indent=2))
    for row in tests:
        print(row['test'], row['status'])
    print('Evidence:', OUTPUT)
    sys.exit(0 if result.wasSuccessful() and len(tests) == result.testsRun else 1)
