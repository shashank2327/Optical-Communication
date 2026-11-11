"""
test_stage1.py — Validation of Physical Models, Units, and Receiver DSP
========================================================================
Validates Gates A through G from the TRD and Implementation Specification:
- Gate A: QPSK & 16-QAM Gray mapping round-trip and average power
- Gate B: Back-to-back zero BER for both QPSK and 16-QAM
- Gate C: Launch power conversion (|A|² in Watts)
- Gate D: Fiber physics (Attenuation, Chromatic Dispersion, Kerr Nonlinearity)
- Gate E: EDFA amplification and ASE noise statistics
- Gate F: SSFM step-size convergence and guard symbol protection
- Gate G: Receiver alignment and conventional DSP (FD-CDC) verification
"""
import os
import sys

# Ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tx import (generate_bits, modulate, rrc_taps, pulse_shape, set_launch_power,
                add_guard_symbols, SPS, SYMBOL_RATE_BD,
                _QPSK_TABLE, _16QAM_TABLE, QPSK_SYMBOLS, QAM16_SYMBOLS)
from rx import (matched_filter, downsample, normalize_symbols,
                phase_correct, demodulate, compute_ber, receive)
from fiber import (ssfm_propagate, edfa_amplify, propagate_link,
                   DEFAULT_FIBER, DEFAULT_EDFA, _alpha_power_km, _beta2_s2_km, _H_PLANCK, _C_LIGHT)
from dsp import compensate_chromatic_dispersion, calculate_evm, calculate_ser


def test_gate_a_mapping_roundtrip():
    """Gate A: Verify Gray mapping round-trip and unit energy for QPSK and 16-QAM."""
    print("\n--- Gate A: Mapping Round-Trip & Unit Energy ---")

    # 1. QPSK
    for bits, sym in _QPSK_TABLE.items():
        bit_arr = np.array(bits, dtype=np.uint8)
        mod_sym = modulate(bit_arr, 'QPSK')[0]
        assert np.isclose(mod_sym, sym), f"QPSK mapping mismatch for {bits}"
        demod_bits = demodulate(np.array([mod_sym]), 'QPSK')
        assert np.array_equal(demod_bits, bit_arr), f"QPSK round-trip failed for {bits}"
    qpsk_avg_pwr = np.mean(np.abs(QPSK_SYMBOLS) ** 2)
    assert np.isclose(qpsk_avg_pwr, 1.0, atol=1e-12), f"QPSK avg power = {qpsk_avg_pwr}"
    print(f"  [PASS] QPSK: all 4 states round-trip exactly, avg power = {qpsk_avg_pwr:.6f}")

    # 2. 16-QAM
    for bits, sym in _16QAM_TABLE.items():
        bit_arr = np.array(bits, dtype=np.uint8)
        mod_sym = modulate(bit_arr, '16-QAM')[0]
        assert np.isclose(mod_sym, sym), f"16-QAM mapping mismatch for {bits}"
        demod_bits = demodulate(np.array([mod_sym]), '16-QAM')
        assert np.array_equal(demod_bits, bit_arr), f"16-QAM round-trip failed for {bits}"
    qam_avg_pwr = np.mean(np.abs(QAM16_SYMBOLS) ** 2)
    assert np.isclose(qam_avg_pwr, 1.0, atol=1e-12), f"16-QAM avg power = {qam_avg_pwr}"
    print(f"  [PASS] 16-QAM: all 16 states round-trip exactly, avg power = {qam_avg_pwr:.6f}")


def test_gate_b_back_to_back_zero_ber():
    """Gate B: Back-to-back Tx/Rx with impairments disabled must yield BER = 0.0."""
    print("\n--- Gate B: Back-to-Back Zero BER ---")
    for mod in ['QPSK', '16-QAM']:
        for pwr_dBm in [-3.0, 0.0, 3.0]:
            n_syms = 2000
            bits = generate_bits(n_syms, mod, seed=123)
            syms = modulate(bits, mod)
            tx_sig = set_launch_power(pulse_shape(syms), pwr_dBm)

            res = receive(tx_sig, syms, bits, modulation=mod)
            assert res['ber'] == 0.0, f"Back-to-back BER != 0 for {mod} at {pwr_dBm} dBm: got {res['ber']}"
            assert res['ser'] == 0.0, f"Back-to-back SER != 0 for {mod} at {pwr_dBm} dBm: got {res['ser']}"
            # Truncated RRC filter (rolloff=0.1, span=10) has intrinsic ~6.3% EVM floor
            assert res['evm'] < 10.0, f"Back-to-back EVM unexpectedly high: {res['evm']}%"
            print(f"  [PASS] {mod:<6} at {pwr_dBm:+2.0f} dBm: BER = {res['ber']:.1e}, SER = {res['ser']:.1e}, EVM = {res['evm']:.2f}%")


def test_gate_c_launch_power():
    """Gate C: Physical launch power conversion in Watts."""
    print("\n--- Gate C: Launch Power Conversion ---")
    for p_dBm in [-5.0, 0.0, 5.0]:
        expected_W = 1e-3 * (10.0 ** (p_dBm / 10.0))
        bits = generate_bits(1000, 'QPSK', seed=42)
        sig = set_launch_power(pulse_shape(modulate(bits, 'QPSK')), p_dBm)
        actual_W = float(np.mean(np.abs(sig) ** 2))
        rel_err = abs(actual_W - expected_W) / expected_W
        assert rel_err < 1e-4, f"Power mismatch at {p_dBm} dBm: expected {expected_W}, got {actual_W}"
        print(f"  [PASS] Launch power {p_dBm:+2.0f} dBm: Expected {expected_W*1e3:.4f} mW, Actual {actual_W*1e3:.4f} mW (rel error: {rel_err:.2e})")


def test_gate_d_fiber_physics():
    """Gate D: Fiber attenuation, dispersion broadening, and Kerr nonlinear phase shift."""
    print("\n--- Gate D: Fiber Physics Tests ---")
    fs = SPS * SYMBOL_RATE_BD

    # 1. Attenuation-only: power decay must match exp(-alpha * L)
    fiber_att = {'alpha_dB_km': 0.2, 'D_ps_nm_km': 0.0, 'gamma_W_km': 0.0, 'wavelength_nm': 1550.0}
    L_km = 80.0
    cw_field = np.ones(1024, dtype=complex) * np.sqrt(1e-3)
    p_in = np.mean(np.abs(cw_field) ** 2)
    out_att = ssfm_propagate(cw_field, L_km, fiber_att, fs, dz_km=1.0)
    p_out = np.mean(np.abs(out_att) ** 2)
    expected_decay = 10.0 ** (-0.2 * L_km / 10.0)
    assert np.isclose(p_out / p_in, expected_decay, rtol=1e-3)
    print(f"  [PASS] Attenuation-only: Expected decay = {expected_decay:.4f}, Actual = {p_out/p_in:.4f}")

    # 2. Kerr-only: nonlinear phase shift Δφ = γ * P * L
    fiber_kerr = {'alpha_dB_km': 0.0, 'D_ps_nm_km': 0.0, 'gamma_W_km': 1.3, 'wavelength_nm': 1550.0}
    p_kerr = 10e-3  # 10 mW
    field_kerr = np.ones(512, dtype=complex) * np.sqrt(p_kerr)
    out_kerr = ssfm_propagate(field_kerr, L_km, fiber_kerr, fs, dz_km=1.0)
    expected_dphi = 1.3 * p_kerr * L_km  # radians
    actual_dphi = float(np.angle(out_kerr[0]) - np.angle(field_kerr[0]))
    assert np.isclose(actual_dphi, expected_dphi, rtol=1e-3)
    print(f"  [PASS] Kerr-only: Expected dphi = {expected_dphi:.4f} rad, Actual = {actual_dphi:.4f} rad")

    # 3. Dispersion-only: spectral phase must be exp(-j * beta2/2 * omega^2 * L)
    fiber_disp = {'alpha_dB_km': 0.0, 'D_ps_nm_km': 17.0, 'gamma_W_km': 0.0, 'wavelength_nm': 1550.0}
    pulse = np.zeros(1024, dtype=complex)
    pulse[512] = 1.0  # impulse
    out_disp = ssfm_propagate(pulse, L_km, fiber_disp, fs, dz_km=1.0)
    beta2 = _beta2_s2_km(17.0, 1550.0)
    omega = 2.0 * np.pi * np.fft.fftfreq(1024, d=1.0/fs)
    expected_fft = np.fft.fft(pulse) * np.exp(-1j * beta2 / 2.0 * (omega ** 2) * L_km)
    assert np.allclose(np.fft.fft(out_disp), expected_fft, atol=1e-10)
    print(f"  [PASS] Dispersion-only: FFT matches analytical phase filter within 1e-10")


def test_gate_e_edfa_noise():
    """Gate E: EDFA gain compensation and ASE noise variance."""
    print("\n--- Gate E: EDFA Gain & ASE Noise ---")
    fs = SPS * SYMBOL_RATE_BD
    gain_dB = 16.0
    G = 10.0 ** (gain_dB / 10.0)
    NF = 10.0 ** (DEFAULT_EDFA['noise_figure_dB'] / 10.0)
    n_sp = (G * NF - 1.0) / (2.0 * (G - 1.0))
    nu = _C_LIGHT / 1550e-9
    s_ase = (G - 1.0) * n_sp * _H_PLANCK * nu
    expected_sigma2_quad = s_ase * fs / 2.0

    # Feed zero input to measure pure ASE noise
    zero_in = np.zeros(200000, dtype=complex)
    rng = np.random.default_rng(42)
    noise_out = edfa_amplify(zero_in, gain_dB, DEFAULT_EDFA, fs, rng)
    actual_sigma2_real = np.var(noise_out.real)
    actual_sigma2_imag = np.var(noise_out.imag)

    rel_diff_r = abs(actual_sigma2_real - expected_sigma2_quad) / expected_sigma2_quad
    rel_diff_i = abs(actual_sigma2_imag - expected_sigma2_quad) / expected_sigma2_quad
    assert rel_diff_r < 0.05, f"Real noise variance mismatch: {actual_sigma2_real} vs {expected_sigma2_quad}"
    assert rel_diff_i < 0.05, f"Imag noise variance mismatch: {actual_sigma2_imag} vs {expected_sigma2_quad}"
    print(f"  [PASS] ASE variance per quad: Expected {expected_sigma2_quad:.4e}, Measured I={actual_sigma2_real:.4e}, Q={actual_sigma2_imag:.4e}")


def test_gate_f_ssfm_convergence():
    """Gate F: SSFM step size convergence (0.5 km vs 0.25 km vs 1.0 km)."""
    print("\n--- Gate F: SSFM Convergence ---")
    fs = SPS * SYMBOL_RATE_BD
    bits = generate_bits(1000, 'QPSK', seed=42)
    sig = set_launch_power(pulse_shape(modulate(bits, 'QPSK')), 3.0)

    # Propagate 80 km with dz = 1.0 km, 0.5 km, 0.25 km
    f_params = DEFAULT_FIBER
    out_100 = ssfm_propagate(sig, 80.0, f_params, fs, dz_km=1.0)
    out_050 = ssfm_propagate(sig, 80.0, f_params, fs, dz_km=0.5)
    out_025 = ssfm_propagate(sig, 80.0, f_params, fs, dz_km=0.25)

    diff_100_050 = np.linalg.norm(out_100 - out_050) / np.linalg.norm(out_050)
    diff_050_025 = np.linalg.norm(out_050 - out_025) / np.linalg.norm(out_025)
    print(f"  Relative difference (1.0 km vs 0.5 km):  {diff_100_050:.4e}")
    print(f"  Relative difference (0.5 km vs 0.25 km): {diff_050_025:.4e}")
    assert diff_050_025 < diff_100_050, "SSFM should converge monotonically with smaller dz"
    assert diff_050_025 < 1e-2, "0.5 km step size must be within 1% of 0.25 km"
    print("  [PASS] SSFM converges smoothly with decreasing step size dz")


def test_gate_g_fd_cdc_baseline():
    """Gate G: Verify linear FD-CDC recovers QPSK across 80 km with zero noise."""
    print("\n--- Gate G: Conventional DSP (FD-CDC) Recovery ---")
    fs = SPS * SYMBOL_RATE_BD
    bits = generate_bits(2000, 'QPSK', seed=42)
    syms = modulate(bits, 'QPSK')
    sig = set_launch_power(pulse_shape(syms), 0.0)

    # Propagate 80 km with 0 noise
    edfa_no_noise = {'noise_figure_dB': -100}
    rx_sig = propagate_link(sig, 1, 80.0, DEFAULT_FIBER, edfa_no_noise, fs, dz_km=0.5, seed=0)

    # 1. Uncompensated: dispersion destroys eye
    res_raw = receive(rx_sig, syms, bits, modulation='QPSK')
    print(f"  Raw uncompensated BER at 80 km: {res_raw['ber']:.4f} (EVM = {res_raw['evm']:.1f}%)")

    # 2. FD-CDC compensated: perfectly recovers eye
    rx_cdc = compensate_chromatic_dispersion(rx_sig, 80.0, DEFAULT_FIBER, fs)
    res_cdc = receive(rx_cdc, syms, bits, modulation='QPSK')
    print(f"  FD-CDC compensated BER at 80 km: {res_cdc['ber']:.4e} (EVM = {res_cdc['evm']:.1f}%)")
    assert res_cdc['ber'] == 0.0, f"FD-CDC must produce zero BER on linear 80 km link, got {res_cdc['ber']}"
    print("  [PASS] FD-CDC completely eliminates chromatic dispersion on linear link")


if __name__ == '__main__':
    test_gate_a_mapping_roundtrip()
    test_gate_b_back_to_back_zero_ber()
    test_gate_c_launch_power()
    test_gate_d_fiber_physics()
    test_gate_e_edfa_noise()
    test_gate_f_ssfm_convergence()
    test_gate_g_fd_cdc_baseline()
    print("\n==========================================")
    print("  ALL STAGE 1 ACCEPTANCE GATES PASSED!    ")
    print("==========================================")
