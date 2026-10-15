"""Small native AIHWKit v1.1.0 comparison; no package patches or installations."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from aihwkit.nn import AnalogLinear
from aihwkit.inference.compensation.drift import (
    GlobalDriftCompensation, GlobalDriftCompensationWithExactReference,
)
import driftslice_core as ds


def native_layer(weight, bias, cfg, compensation=None):
    r = ds.make_hwa_rpu(cfg)
    r.forward.is_perfect = True
    r.pre_post.input_range.enable = False
    r.mapping.weight_scaling_omega = 0.0
    r.mapping.learn_out_scaling = False
    r.noise_model, _ = ds.make_pcm_noise_model(cfg)
    r.drift_compensation = compensation
    layer = AnalogLinear(weight.shape[1], weight.shape[0], bias=True, rpu_config=r).eval()
    layer.set_weights(weight, bias)
    return layer


def native_comparison():
    """Controlled equality plus noisy descriptive comparison, not a benchmark."""
    device = torch.device('cpu')
    cfg = ds.Config(fast_mode=True, require_cuda=False)
    w = torch.linspace(-0.8, 0.8, 96).reshape(8, 12)
    bias = torch.linspace(-0.1, 0.1, 8)
    controlled = replace(cfg, programming_noise_scale=0., read_noise_scale=0.)
    layer = native_layer(w, bias, controlled)
    tile, = list(layer.analog_tiles())
    with ds.random_stream(321, device):
        layer.program_analog_weights()
    custom = ds.PCMPhysicalMatrix(w, controlled, 321)
    # Match sampled drift coefficients; with programming noise off, converter
    # round-tripping preserves the differential device state in this fixture.
    custom.nu = tuple(n.detach().clone() for n in tile.drift_noise_parameters)
    rows = []
    x = torch.linspace(-1., 1., 36).reshape(3, 12)
    for t in (ds.T_REF, 40., ds.YEAR, 5 * ds.YEAR):
        with ds.random_stream(444, device):
            layer.drift_analog_weights(ds.elapsed_pcm_time(t))
        actual, actual_bias = layer.get_weights(apply_weight_scaling=False)
        expected = custom.logical_weight(custom.read_conductances(t, 444))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(actual_bias, bias, atol=0, rtol=0)
        with torch.inference_mode():
            y = layer(x)
        torch.testing.assert_close(y, F.linear(x, expected, bias), atol=1e-6, rtol=2e-6)
        rows.append({'absolute_time_s': t, 'weight_max_error': float((actual - expected).abs().max()),
                     'output_max_error': float((y - F.linear(x, expected, bias)).abs().max())})

    # No exact equality is expected here: native converts signed programmed
    # weights back to a new pair; custom retains both physical conductances.
    noisy = []
    for seed in range(12):
        native = native_layer(w, bias, cfg)
        with ds.random_stream(seed, device):
            native.program_analog_weights()
        physical = ds.PCMPhysicalMatrix(w, cfg, seed)
        for t in (ds.T_REF, ds.YEAR):
            with ds.random_stream(1000 + seed, device):
                native.drift_analog_weights(ds.elapsed_pcm_time(t))
            nw, nb = native.get_weights(apply_weight_scaling=False)
            cw = physical.logical_weight(physical.read_conductances(t, 1000 + seed))
            assert torch.isfinite(nw).all() and torch.isfinite(cw).all()
            torch.testing.assert_close(nb, bias, atol=0, rtol=0)
            noisy.append({'seed': seed, 'absolute_time_s': t,
                          'native_mean_abs_weight': float(nw.abs().mean()),
                          'custom_mean_abs_weight': float(cw.abs().mean()),
                          'native_custom_rmse': float((nw - cw).square().mean().sqrt())})

    compensation = []
    for cls in (GlobalDriftCompensation, GlobalDriftCompensationWithExactReference):
        native = native_layer(w, bias, cfg, cls())
        nt, = list(native.analog_tiles())
        with ds.random_stream(777, device):
            native.program_analog_weights()
        reference = (nt.reference_combined_weights if cls is GlobalDriftCompensationWithExactReference
                     else nt.programmed_weights)
        with ds.random_stream(888, device):
            native.drift_analog_weights(ds.elapsed_pcm_time(ds.YEAR))
        current, _ = native.get_weights(apply_weight_scaling=False)
        expected_alpha = reference.abs().mean().clamp_min(0.0001) / current.abs().mean().clamp_min(0.0001)
        torch.testing.assert_close(nt.alpha, expected_alpha, atol=1e-6, rtol=1e-6)
        with torch.inference_mode():
            y = native(x)
        torch.testing.assert_close(y, F.linear(x, current * nt.alpha, bias), atol=2e-6, rtol=2e-6)
        compensation.append({'class': cls.__name__, 'alpha': float(nt.alpha),
                             'independent_alpha': float(expected_alpha)})
    return {'controlled_shared_nu_no_program_or_read_noise': rows,
            'noisy_unmatched_samples_12_seeds': noisy,
            'native_compensation_perfect_io': compensation,
            'limitations': 'Small CPU fixtures; no claim of distributional equivalence or GPU verification.'}
