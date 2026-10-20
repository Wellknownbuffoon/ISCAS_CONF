"""Lightweight regression tests for the uploaded Colab notebook; never runs setup/full cells.

python tests/test_notebook_gradients.py --out /tmp/a-new-output-folder
"""
import argparse
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogAdam
from aihwkit.optim.context import AnalogContext

NOTEBOOK = Path(__file__).resolve().parents[1] / 'DriftSlice_ISCAS_RESNET_AND_MLP_Full_Experiment_FIXED.ipynb'
OUTPUT = None
ORIGINAL = None


def load_runtime(notebook, destination):
    """Load definition cells only: no pip, Drive mount, datasets, full or smoke runners."""
    n = json.loads(Path(notebook).read_text())
    cells = [2, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 36, 40]
    sources = ['from __future__ import annotations\n']
    for i in cells:
        src = ''.join(n['cells'][i]['source']).replace('from __future__ import annotations\n', '')
        sources.append(src)
    destination.write_text('\n\n'.join(sources))
    name = 'notebook_gradient_test_' + destination.stem
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.LOCAL_CACHE_STR = str(destination.parent/'local-cache')
    return module


class GradientFixTests(unittest.TestCase):
    measurements = {}

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.r = load_runtime(NOTEBOOK, OUTPUT/'notebook_runtime.py')
        cls.r.seed_all(7171)
        cls.device = torch.device('cpu')
        cls.cfg = cls.r.Config(require_cuda=False, fast_mode=True, epochs=2,
                              digital_epochs=1, train_repeats=1, batch_size=2,
                              test_batch_size=2, validation_reads=1)

    def test_01_real_aihwkit_finite_gradient_overflow(self):
        r = self.r
        r.verify_gradient_norm_fix(self.cfg, self.device, 'CPU regression')
        layer = AnalogLinear(8, 3, bias=False, rpu_config=r.make_hwa_rpu(self.cfg))
        layer.set_weights(torch.full((3, 8), 1e-18))
        w = r.physical_weight_parameter(layer)
        with r.random_stream(1, self.device):
            loss = r.dar_penalty(w, w.detach().clone(), 1, 60., self.cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(w.grad).all())
        before = w.grad.clone()
        self.assertTrue(torch.isinf(w.grad.norm()))
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            torch.nn.utils.clip_grad_norm_(layer.parameters(), .8, error_if_nonfinite=True)
        torch.testing.assert_close(w.grad, before, atol=0, rtol=0)
        expected_norm = before.double().norm()
        expected = (before.double() * (.8/(expected_norm+1e-6)).clamp(max=1)).float()
        norm = r.clip_gradients_checked(layer, .8)
        torch.testing.assert_close(w.grad, expected, atol=0, rtol=0)
        self.assertLessEqual(w.grad.double().norm(), .800001)
        opt = AnalogAdam(layer.parameters(), lr=.001)
        opt.regroup_param_groups(layer)
        initial = w.detach().clone()
        opt.step()
        self.assertGreater(float((w.detach()-initial).abs().max()), 0)
        self.measurements['native_weak_scale_reproduction'] = {
            'loss': float(loss.detach()), 'all_gradient_entries_finite': True,
            'original_fp32_norm': 'inf', 'float64_norm': float(norm),
            'clipped_norm': float(w.grad.double().norm()), 'analogadam_updated_weights': True}

    def test_02_true_nan_inf_fail_without_modifying_gradients(self):
        for bad in (float('nan'), float('inf')):
            m = torch.nn.Linear(3, 1)
            m.weight.grad = torch.full_like(m.weight, bad)
            m.bias.grad = torch.ones_like(m.bias)
            before = m.bias.grad.clone()
            with self.assertRaisesRegex(FloatingPointError, 'weight.*NaN=.*Inf='):
                self.r.clip_gradients_checked(m, .8, context='controlled nonfinite test')
            torch.testing.assert_close(m.bias.grad, before)

    def test_03_regular_and_extreme_coefficient_clipping(self):
        m = torch.nn.Linear(3, 2)
        m.weight.grad = torch.tensor([[1., -2., 3.], [4., -5., 6.]])
        m.bias.grad = torch.tensor([2., 3.])
        before = [p.grad.clone() for p in m.parameters()]
        norm = torch.cat([g.flatten().double() for g in before]).norm()
        actual = self.r.clip_gradients_checked(m, .8)
        torch.testing.assert_close(actual, norm)
        for p, g in zip(m.parameters(), before):
            torch.testing.assert_close(p.grad, (g.double()*(.8/(norm+1e-6))).float(), atol=0, rtol=0)
        m.weight.grad.fill_(3e38); m.bias.grad.fill_(3e38)
        self.r.clip_gradients_checked(m, .8)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertTrue(all(p.grad.abs().max() > 0 for p in m.parameters()))

    def test_04_context_rebind_preserves_parameters_and_moments(self):
        r = self.r
        m = r.AnalogMLP(self.cfg, 42, self.device)
        opt = AnalogAdam(m.parameters(), lr=.001); opt.regroup_param_groups(m)
        x = torch.rand(2, 1, 28, 28)
        torch.nn.functional.cross_entropy(m(x), torch.tensor([1,2])).backward()
        r.clip_gradients_checked(m, .8); opt.step(); opt.zero_grad(set_to_none=True)
        old_weights = [r.physical_weight_parameter(l) for l in m.layers().values()]
        old_moments = [opt.state[w]['exp_avg'].clone() for w in old_weights]
        r.load_model_state(m, r.cpu_cloned_state(m), self.device)
        self.assertNotEqual({id(p) for p in m.parameters()}, {id(p) for g in opt.param_groups for p in g['params']})
        r.rebind_optimizer_after_analog_load(m, opt)
        grouped = [p for g in opt.param_groups for p in g['params']]
        self.assertEqual({id(p) for p in m.parameters()}, {id(p) for p in grouped})
        for w, old in zip(old_weights, old_moments):
            torch.testing.assert_close(opt.state[w]['exp_avg'], old, atol=0, rtol=0)
        self.assertEqual(sum(isinstance(p, AnalogContext) for p in grouped), 2)

    def test_05_notebook_actual_resnet_dar_resume_and_mlp(self):
        r = self.r
        g = torch.Generator().manual_seed(123)
        datasets = {}
        def loaders(arch):
            shape = (3, 32, 32) if arch == 'resnet32' else (1, 28, 28)
            if arch not in datasets:
                datasets[arch] = (TensorDataset(torch.randn(2, *shape, generator=g), torch.tensor([1,2])),
                                  TensorDataset(torch.randn(2, *shape, generator=g), torch.tensor([3,4])))
            train = DataLoader(datasets[arch][0], batch_size=2, shuffle=True,
                               generator=torch.Generator().manual_seed(124))
            val = DataLoader(datasets[arch][1], batch_size=2)
            train.split_name = 'train'; val.split_name = 'val'
            return train, val
        for arch in ('mlp', 'resnet32'):
            digital = (r.make_digital_resnet32(42) if arch == 'resnet32' else r.DigitalMLP()).eval()
            trainer = r.train_hwa_resnet if arch == 'resnet32' else r.train_hwa_all
            identity = r.state_dict_hash(digital.state_dict())
            a = OUTPUT/f'{arch}-resumed'; b = OUTPUT/f'{arch}-reference'
            train, val = loaders(arch)
            r._INTERRUPT_HOOK.update(stage=f'hwa:{r.DAR}', after_epoch=1)
            with self.assertRaises(r.SimulatedInterruption):
                trainer(r.DAR, self.cfg, 0, digital, identity, train, val, self.device, a, 'fixture-fp')
            train, val = loaders(arch)
            result = trainer(r.DAR, self.cfg, 0, digital, identity, train, val, self.device, a, 'fixture-fp')
            train, val = loaders(arch)
            reference = trainer(r.DAR, self.cfg, 0, digital, identity, train, val, self.device, b, 'fixture-fp')
            final_a, final_b = r.torch_load(a/f'{r.DAR}_final.pt'), r.torch_load(b/f'{r.DAR}_final.pt')
            self.assertEqual(r.state_dict_hash(final_a['model']), r.state_dict_hash(final_b['model']))
            self.assertEqual(result['optimizer_steps'], 2)
            self.assertGreater(result['history'][1]['train_extra'], 0)
            self.assertEqual(result['history'], reference['history'])
            self.measurements[arch+'_resume'] = {'completed_epochs': result['analog_epochs_completed'],
                'optimizer_steps': result['optimizer_steps'], 'epoch2_dar': result['history'][1]['train_extra'],
                'bit_identical_to_uninterrupted': True}

    def test_06_legacy_import_is_validated_read_only_and_idempotent(self):
        r = self.r
        root = OUTPUT/'migration'
        source = root/'full'/'old'; source.mkdir(parents=True)
        cfg = self.cfg
        data = {'fixture': 'test-data-sha'}
        legacy = r.stable_hash({'schema':r.SCHEMA,'code':r._GRADIENT_FIX_LEGACY_CODE['mlp'],
                               'env':r.environment_identity(),'data':data,
                               'const':{'DAY':r.DAY,'YEAR':r.YEAR,'T_REF':r.T_REF},
                               'cfg':r.config_subset(cfg,r.BOOKKEEPING_FIELDS|r.RETENTION_ONLY_FIELDS)})
        fp = {'train':'new-training-fingerprint'}
        manifest = {'arch':'mlp','fingerprints':{'train':legacy,'code_hashes':{'train':r._GRADIENT_FIX_LEGACY_CODE['mlp']}}}
        (source/'manifest.json').write_text(json.dumps(manifest))
        folder = source/'training'/legacy[:12]/'fold0'; folder.mkdir(parents=True)
        state = {'weight':torch.ones(2,3)}
        for suffix in ('final','best_validation'):
            pack = {'kind':'digital_'+suffix,'train_fp':legacy,'fold':0,
                    'history':[{'epoch':1}],'model':state,'model_identity':r.state_dict_hash(state)}
            torch.save(pack,folder/f'digital_{suffix}.pt')
        slot = {'kind':'hwa_training','train_fp':legacy,'fold':0,'method':r.DAR,
                'model':state,'best_model':state,'digital_identity':r.state_dict_hash(state),'history':[{'epoch':1}],
                'next_epoch':1,'optimizer_steps':1,'optimizer':{'state':{}},'scheduler':{'last_epoch':1}}
        torch.save(slot,folder/f'{r.DAR}_training.slotB.pt')
        torch.save({'digest':'split-fixture'},folder/'split.pt')
        before = {str(p):r.sha256_bytes(p.read_bytes()) for p in source.rglob('*') if p.is_file()}
        target = root/'full'/'new';target.mkdir()
        exp = SimpleNamespace(run_dir=target,train_dir=target/'training'/'new-training',
                              cfg=cfg,arch='mlp',fp=fp,data_hashes=data,methods=(r.DAR,),run_id='new')
        report = r.import_training_for_gradient_fix(exp,'old')
        self.assertEqual(len(report['imported']),3)
        self.assertEqual(report,r.import_training_for_gradient_fix(exp,'old'))
        self.assertEqual(before,{str(p):r.sha256_bytes(p.read_bytes()) for p in source.rglob('*') if p.is_file()})
        imported=r.torch_load(exp.train_dir/'fold0'/f'{r.DAR}_training.slotB.pt')
        self.assertEqual(imported['next_epoch'],1)
        self.assertEqual(imported['original_train_fp'],legacy)
        self.assertEqual(r.state_dict_hash(imported['model']),r.state_dict_hash(state))
        exp.data_hashes={'fixture':'different'};exp.run_dir=root/'full'/'rejected';exp.train_dir=exp.run_dir/'training'
        with self.assertRaisesRegex(r.ScientificCheckError,'config/data/runtime'):
            r.import_training_for_gradient_fix(exp,'old')
        self.assertFalse(exp.train_dir.exists())
        self.measurements['legacy_import']={'source_files_unchanged':True,'copied_files':3,
                                           'resume_epoch_preserved':1,'foreign_data_rejected':True}

    def test_08_real_original_resnet_checkpoint_import_and_continue(self):
        if ORIGINAL is None:
            self.skipTest('Original uploaded notebook not supplied; use --original-notebook')
        r = self.r
        old = load_runtime(ORIGINAL, OUTPUT/'original_runtime.py')
        code = old.component_source_hash({'config','data','model','train','rn_data','rn_model','rn_train'})
        self.assertEqual(code, r._GRADIENT_FIX_LEGACY_CODE['resnet32'])
        for i in (17, 19, 25, 27):
            original_json = json.loads(ORIGINAL.read_text())
            fixed_json = json.loads(NOTEBOOK.read_text())
            self.assertEqual(original_json['cells'][i]['source'], fixed_json['cells'][i]['source'])
        cfg = self.cfg
        data = {'synthetic_fixture':'same-inputs'}
        legacy_fp = old.compute_fingerprints_resnet(cfg, data, old.RESNET_SPEC)['train']
        new_fp = r.compute_fingerprints_resnet(cfg, data, r.RESNET_SPEC)['train']
        self.assertNotEqual(legacy_fp, new_fp)
        root = OUTPUT/'real-original-import';source=root/'full'/'old';source.mkdir(parents=True)
        (source/'manifest.json').write_text(json.dumps({'arch':'resnet32',
            'fingerprints':{'train':legacy_fp,'code_hashes':{'train':code}}}))
        folder=source/'training'/legacy_fp[:12]/'fold0'
        g=torch.Generator().manual_seed(615)
        train_ds=TensorDataset(torch.randn(2,3,32,32,generator=g),torch.tensor([1,2]))
        val_ds=TensorDataset(torch.randn(2,3,32,32,generator=g),torch.tensor([3,4]))
        def loaders():
            train=DataLoader(train_ds,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(616))
            val=DataLoader(val_ds,batch_size=2);train.split_name='train';val.split_name='val'
            return train,val
        train,val=loaders()
        digital,_,digital_pack=old.train_digital_resnet(cfg,0,train,val,self.device,folder,legacy_fp)
        identity=digital_pack['model_identity']
        train,val=loaders()
        old.train_hwa_resnet(old.HWA,cfg,0,digital,identity,train,val,self.device,folder,legacy_fp)
        train,val=loaders();old._INTERRUPT_HOOK.update(stage=f'hwa:{old.DAR}',after_epoch=1)
        with self.assertRaises(old.SimulatedInterruption):
            old.train_hwa_resnet(old.DAR,cfg,0,digital,identity,train,val,self.device,folder,legacy_fp)
        before={str(p):r.sha256_bytes(p.read_bytes()) for p in source.rglob('*') if p.is_file()}
        target=root/'full'/'fixed';target.mkdir()
        exp=SimpleNamespace(run_dir=target,train_dir=target/'training'/new_fp[:12],
                            cfg=cfg,arch='resnet32',fp={'train':new_fp},data_hashes=data,methods=(r.HWA,r.DAR),run_id='fixed')
        imported=r.import_training_for_gradient_fix(exp,'old')
        self.assertEqual(len(imported['imported']),5)
        from unittest.mock import patch
        train,val=loaders()
        with patch.object(r, 'transfer_pretrained_resnet_to_analog', side_effect=AssertionError('Completed HWA must not retrain')):
            completed=r.train_hwa_resnet(r.HWA,cfg,0,digital,identity,train,val,self.device,exp.train_dir/'fold0',new_fp)
        self.assertEqual(completed['analog_epochs_completed'],2)
        train,val=loaders()
        pack=r.train_hwa_resnet(r.DAR,cfg,0,digital,identity,train,val,self.device,
                                exp.train_dir/'fold0',new_fp)
        self.assertEqual(pack['analog_epochs_completed'],2)
        self.assertEqual(pack['optimizer_steps'],2)
        self.assertGreater(pack['history'][1]['train_extra'],0)
        self.assertEqual(before,{str(p):r.sha256_bytes(p.read_bytes()) for p in source.rglob('*') if p.is_file()})
        self.measurements['real_original_resnet_import']={'old_source_hash_verified':True,
            'old_checkpoint_imported_without_changing_source_files':True,
            'continued_from_epoch':2,'epochs_completed':2,'steps_total':2,
            'completed_fp32_and_hwa_reused':True}

    def test_07_gpu_regression_is_not_claimed_on_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available here; notebook preflight runs the regression on the actual GPU')
        self.r.verify_gradient_norm_fix(self.cfg, torch.device('cuda'), 'regression')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--original-notebook',type=Path)
    args=p.parse_args();OUTPUT=args.out.resolve();OUTPUT.mkdir(parents=True,exist_ok=False)
    ORIGINAL=args.original_notebook.resolve() if args.original_notebook else None
    with (OUTPUT/'test.log').open('w') as log, contextlib.redirect_stdout(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GradientFixTests))
    report={'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
            'failed':len(result.failures)+len(result.errors),'not_executed':len(result.skipped),
            'skipped':[(str(t),reason) for t,reason in result.skipped],
            'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
            'full_training':'NOT EXECUTED','data':'small synthetic fixtures',
            'measurements':GradientFixTests.measurements}
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
