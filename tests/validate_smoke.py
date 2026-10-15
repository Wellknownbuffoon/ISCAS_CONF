"""Validate an existing NEW smoke run without modifying any experiment artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image


def validate(path):
    def load(name):
        return torch.load(path/name, map_location='cpu', weights_only=False)

    manifest = json.loads((path/'manifest.json').read_text())
    cfg = manifest['config']
    assert cfg['fast_mode'] and manifest['experiment_label'] == 'SMOKE TEST'
    assert (cfg['digital_epochs'], cfg['epochs'], cfg['train_repeats'],
            cfg['realizations'], cfg['retention_points']) == (2, 2, 1, 2, 8)
    split = load('fold0_split.pt')
    assert len(split['train_indices']) == 1024 and len(split['val_indices']) == 256
    assert set(split['train_indices'].tolist()).isdisjoint(split['val_indices'].tolist())
    assert torch.equal(split['test_indices'], torch.arange(512))
    digital = load('fold0_digital_pretrained.pt')
    assert digital['epochs_completed'] == 2 and digital['optimizer_steps'] > 0
    direct = load('fold0_FP32-Direct_analog_checkpoint.pt')
    assert direct['analog_epochs'] == 0 and direct['optimizer_steps'] == 0
    identity = digital['selected_state_sha256']
    assert direct['pretrained_provenance']['selected_state_sha256'] == identity
    updates, transfer = {}, direct['transfer_errors']
    for method in cfg['methods']:
        final = load(f'fold0_{method}_final.pt')
        best = load(f'fold0_{method}_best_validation.pt')
        assert final['analog_epochs_completed'] == 2 and final['checkpoint_epoch'] == 2
        assert 1 <= best['checkpoint_epoch'] <= 2
        assert final['selected_best_validation_epoch'] == best['checkpoint_epoch']
        assert best['optimizer_steps'] == 8 and len(best['history']) == 2
        assert best['pretrained_provenance']['selected_state_sha256'] == identity
        assert all(v > 0 for v in best['first_weight_update'].values())
        updates[method] = best['first_weight_update']
        assert best['transfer_errors'] == transfer
    retention = pd.read_csv(path/'retention_all_methods.csv')
    expected = (len(cfg['methods']) + 1) * 5 * 8 * 2
    assert len(retention) == expected
    assert set(retention.method) == {'FP32-Direct', *cfg['methods']}
    assert set(retention.compensation) == {'none', 'DriftSlice', 'GDC-legacy-ones',
                                        'GDC-safe-onehot', 'GDC-ideal-onehot'}
    assert (retention.experiment_label == 'SMOKE TEST').all()
    assert np.isfinite(retention[['loss', 'accuracy', 'error', 'gain_fc1', 'gain_fc2']]).all().all()
    assert retention.accuracy.between(0, 100).all()
    assert (retention.groupby(['method', 'compensation', 'realization']).size() == 8).all()
    diagnostic = pd.read_csv(path/'gdc_calibration_diagnostics.csv')
    clip_columns = [c for c in diagnostic if c.endswith('_clip_fraction')]
    safe = diagnostic[diagnostic.compensation == 'GDC-safe-onehot']
    legacy = diagnostic[diagnostic.compensation == 'GDC-legacy-ones']
    assert (safe[clip_columns] == 0).all().all()
    for fmt in ('png', 'pdf'):
        assert (path/f'retention_all_methods.{fmt}').stat().st_size > 100
    with Image.open(path/'retention_all_methods.png') as image:
        image.verify()
    assert (path/'retention_all_methods.pdf').read_bytes().startswith(b'%PDF')
    return {'status': 'PASS', 'label': 'SMOKE TEST', 'run_directory': str(path),
            'retention_rows': len(retention), 'methods_including_direct': retention.method.nunique(),
            'compensation_strategies': retention.compensation.nunique(),
            'train_validation_indices_disjoint': True, 'test_examples': 512,
            'fp32_optimizer_steps': digital['optimizer_steps'],
            'fp32_selected_epoch': digital['selected_epoch'],
            'first_analogadam_max_updates': updates, 'transfer_errors': transfer,
            'safe_max_clip_fractions': safe[clip_columns].max().to_dict(),
            'legacy_mean_clip_fractions': legacy[clip_columns].mean().to_dict(),
            'legacy_max_clip_fractions': legacy[clip_columns].max().to_dict(),
            'figures_validated': ['PNG', 'PDF'], 'fingerprint': manifest['fingerprint'],
            'source_hash': manifest['source_hash']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', required=True, type=Path, help='A NEW report path')
    args = parser.parse_args()
    result = validate(args.run.resolve())
    # Exclusive create protects prior evidence just as the test runner does.
    with args.report.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result, indent=2))
