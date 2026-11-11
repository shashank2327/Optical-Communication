"""
generate_ber_curve.py — Offline Training and Rigorous Benchmark Sweep Script
=============================================================================
Executes Phases 3, 4, and 5 of the project specification:
1. Generates leakage-free multi-burst training and validation datasets.
2. Trains dedicated QPSK and 16-QAM neural network equalizers with feature standardizers.
3. Saves models with rich metadata to:
       models/equalizer_qpsk.pt
       models/equalizer_16qam.pt
       models/equalizer.pt (default QPSK copy for backward compatibility)
4. Conducts fair, paired evaluation on independent test sequences:
   - Evaluates (1) No Equalization, (2) Conventional Linear DSP (FD-CDC), and (3) Neural Network
   - Uses identical test bits, noise realizations, and exact common central bit indices.
5. Sweeps distances (80 km to 1600 km) and optical launch powers (-2 to +6 dBm) for BOTH QPSK and 16-QAM.
6. Generates publication-ready figures for BER, EVM, and Constellation Grids.
7. Logs detailed CSV records with exact error counts and empirical resolution bounds.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tx import (generate_bits, modulate, pulse_shape, set_launch_power,
                SYMBOL_RATE_BD, SPS, QPSK_SYMBOLS, QAM16_SYMBOLS)
from fiber import propagate_link, DEFAULT_FIBER, DEFAULT_EDFA
from rx import matched_filter, normalize_symbols, demodulate, count_bit_errors
from dsp import compensate_chromatic_dispersion, calculate_evm, carrier_phase_recovery
from ml import (NNEqualizer, FeatureScaler, build_dataset, build_multiburst_dataset,
                train as train_model, equalize, save_model, WINDOW_DEFAULT)
from plots import (plot_ber_vs_distance, plot_ber_vs_power, plot_evm_vs_distance,
                   plot_evm_vs_power, plot_training_curve, plot_constellation,
                   plot_constellation_grid, save_figure)

# ─── Configuration ────────────────────────────────────────────────────────────
SPAN_KM          = 80.0
TRAIN_DISTANCE   = 400.0        # 5 spans × 80 km
TRAIN_POWER_DBM  = 4.0          # In the nonlinear regime where Kerr effect creates distortion
N_BURSTS_TRAIN   = 3            # Multi-burst training set
BURST_LEN_TRAIN  = 2048         # Total training symbols = 3 × 2048 = 6,144
BURST_LEN_VAL    = 2048         # Independent validation burst
BURST_LEN_TEST   = 4096         # Independent test burst per evaluation point
DZ_KM_OFFLINE    = 0.5          # High-accuracy SSFM step size
WINDOW_SIZE      = 11           # Context window (5 past, center, 5 future)

DISTANCES_KM     = [80, 160, 240, 320, 400, 480, 560, 640, 800, 960, 1120, 1280, 1440, 1600]
POWERS_DBM       = [-2.0, 0.0, 2.0, 4.0, 6.0]

FS = SPS * SYMBOL_RATE_BD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_burst(n_symbols: int,
                   modulation: str,
                   distance_km: float,
                   power_dBm: float,
                   data_seed: int,
                   noise_seed: int) -> dict:
    """
    Simulate a complete transmission burst through the optical link.
    Returns transmitted symbols/bits, raw received field, and CDC-compensated field.
    """
    n_spans = max(1, int(round(distance_km / SPAN_KM)))
    bits = generate_bits(n_symbols, modulation, seed=data_seed)
    syms = modulate(bits, modulation)
    tx_sig = set_launch_power(pulse_shape(syms), power_dBm)

    # SSFM propagation through multi-span amplified fiber
    rx_field = propagate_link(tx_sig, n_spans, SPAN_KM, DEFAULT_FIBER, DEFAULT_EDFA,
                              fs=FS, dz_km=DZ_KM_OFFLINE, seed=noise_seed)

    # Conventional Linear DSP (FD-CDC)
    rx_field_cdc = compensate_chromatic_dispersion(rx_field, distance_km, DEFAULT_FIBER, fs=FS)

    return {
        'bits': bits,
        'syms': syms,
        'rx_field': rx_field,
        'rx_field_cdc': rx_field_cdc,
    }


def extract_symbols_and_normalize(rx_field: np.ndarray,
                                  n_expected: int) -> np.ndarray:
    """Run matched filter, deterministic downsampling, and RMS power normalization."""
    mf = matched_filter(rx_field)
    delay = 2 * 10 * SPS
    down = mf[delay : delay + n_expected * SPS : SPS]
    down = down[:n_expected]
    norm_syms, _ = normalize_symbols(down, method='rms')
    return norm_syms


def train_equalizer_for_modulation(modulation: str) -> tuple[NNEqualizer, FeatureScaler, dict]:
    """Train a dedicated neural network equalizer for the specified modulation."""
    print(f"\n[{modulation}] Building multi-burst training dataset at {TRAIN_DISTANCE:.0f} km, {TRAIN_POWER_DBM:+.1f} dBm...", flush=True)
    train_bursts = []
    for b in range(N_BURSTS_TRAIN):
        burst = simulate_burst(
            n_symbols=BURST_LEN_TRAIN,
            modulation=modulation,
            distance_km=TRAIN_DISTANCE,
            power_dBm=TRAIN_POWER_DBM,
            data_seed=100 + b,
            noise_seed=1000 + b
        )
        rx_norm = extract_symbols_and_normalize(burst['rx_field_cdc'], BURST_LEN_TRAIN)
        rx_cpr, _ = carrier_phase_recovery(rx_norm, modulation=modulation, pilot_tx=burst['syms'])
        # Discard 15 boundary transient symbols from filter edges
        train_bursts.append((rx_cpr[15:-15], burst['syms'][15:-15]))

    X_train, Y_train = build_multiburst_dataset(train_bursts, window=WINDOW_SIZE)

    # Independent validation burst (completely separate seed)
    val_burst = simulate_burst(
        n_symbols=BURST_LEN_VAL,
        modulation=modulation,
        distance_km=TRAIN_DISTANCE,
        power_dBm=TRAIN_POWER_DBM,
        data_seed=555,
        noise_seed=5555
    )
    val_rx_norm = extract_symbols_and_normalize(val_burst['rx_field_cdc'], BURST_LEN_VAL)
    val_rx_cpr, _ = carrier_phase_recovery(val_rx_norm, modulation=modulation, pilot_tx=val_burst['syms'])
    X_val, Y_val = build_dataset(val_rx_cpr[15:-15], val_burst['syms'][15:-15], window=WINDOW_SIZE)

    model = NNEqualizer(window=WINDOW_SIZE, hidden=64)
    print(f"[{modulation}] Training NNEqualizer (Window={WINDOW_SIZE}, 64 hidden, Adam lr=2e-3)...", flush=True)
    history, scaler = train_model(
        model, X_train, Y_train, X_val=X_val, Y_val=Y_val,
        epochs=50, batch_size=128, lr=0.002, verbose=True
    )

    metadata = {
        'modulation': modulation,
        'window': WINDOW_SIZE,
        'train_distance_km': TRAIN_DISTANCE,
        'train_power_dBm': TRAIN_POWER_DBM,
        'val_loss': float(history['val_loss'][-1]),
    }
    ckpt_path = os.path.join(MODELS_DIR, f'equalizer_{modulation.lower().replace("-", "")}.pt')
    save_model(model, scaler, modulation=modulation, metadata=metadata, path=ckpt_path)
    if modulation == 'QPSK':
        # Default copy for legacy dashboard compatibility
        save_model(model, scaler, modulation=modulation, metadata=metadata, path=os.path.join(MODELS_DIR, 'equalizer.pt'))

    return model, scaler, history


def evaluate_modulation_sweep(modulation: str,
                              model: NNEqualizer,
                              scaler: FeatureScaler,
                              eval_distance_power: float,
                              eval_power_distance: float = 400.0) -> dict:
    """
    Perform paired distance sweep and power sweep for a single modulation format.
    Guarantees fair common central index slicing and reports exact error counts and EVM.
    """
    bps = 2 if modulation == 'QPSK' else 4
    half = WINDOW_SIZE // 2
    res = {
        'modulation': modulation,
        'distances': np.array(DISTANCES_KM),
        'powers': np.array(POWERS_DBM),
        'dist_no_eq_ber': [],
        'dist_cdc_raw_ber': [],
        'dist_cdc_cpr_ber': [],
        'dist_ml_ber': [],
        'dist_cdc_cpr_evm': [],
        'dist_ml_evm': [],
        'dist_cpr_phase_deg': [],
        'pwr_no_eq_ber': [],
        'pwr_cdc_raw_ber': [],
        'pwr_cdc_cpr_ber': [],
        'pwr_ml_ber': [],
        'pwr_cdc_cpr_evm': [],
        'pwr_ml_evm': [],
        'pwr_cpr_phase_deg': [],
        'csv_rows': [],
        'sample_constellations': {}
    }

    # 1. Distance Sweep
    print(f"\n[{modulation}] Distance Sweep ({len(DISTANCES_KM)} points) at {eval_distance_power:+.1f} dBm...")
    print(f"  {'Dist (km)':<10} {'No-EQ BER':<13} {'CDC (Raw)':<13} {'CDC (+CPR)':<13} {'NN-EQ BER':<13} {'CDC EVM%':<10} {'NN EVM%':<10}")
    print("  " + "-" * 84)

    for dist_km in DISTANCES_KM:
        burst = simulate_burst(
            n_symbols=BURST_LEN_TEST,
            modulation=modulation,
            distance_km=dist_km,
            power_dBm=eval_distance_power,
            data_seed=9000 + dist_km,
            noise_seed=80000 + dist_km
        )
        tx_bits = burst['bits']
        tx_syms = burst['syms']

        # Extract normalized symbols
        rx_raw_no_eq    = extract_symbols_and_normalize(burst['rx_field'], BURST_LEN_TEST)
        rx_norm_cdc_raw = extract_symbols_and_normalize(burst['rx_field_cdc'], BURST_LEN_TEST)
        rx_norm_cdc_cpr, phi_rad = carrier_phase_recovery(rx_norm_cdc_raw, modulation=modulation, pilot_tx=tx_syms)

        # ML Equalization
        rx_ml_eq = equalize(model, rx_norm_cdc_cpr, scaler=scaler, window=WINDOW_SIZE)
        rx_ml_norm, _ = normalize_symbols(rx_ml_eq)

        # Common central symbol slicing
        n_eval = len(rx_ml_norm)
        tx_eval_syms = tx_syms[half : half + n_eval]
        tx_eval_bits = tx_bits[half * bps : (half + n_eval) * bps]

        no_eq_center   = rx_raw_no_eq[half : half + n_eval]
        cdc_raw_center = rx_norm_cdc_raw[half : half + n_eval]
        cdc_cpr_center = rx_norm_cdc_cpr[half : half + n_eval]

        bits_no_eq   = demodulate(no_eq_center, modulation)
        bits_cdc_raw = demodulate(cdc_raw_center, modulation)
        bits_cdc_cpr = demodulate(cdc_cpr_center, modulation)
        bits_ml      = demodulate(rx_ml_norm, modulation)

        err_no_eq, n_b, ber_no_eq   = count_bit_errors(tx_eval_bits, bits_no_eq)
        err_cdc_raw, _, ber_cdc_raw = count_bit_errors(tx_eval_bits, bits_cdc_raw)
        err_cdc_cpr, _, ber_cdc_cpr = count_bit_errors(tx_eval_bits, bits_cdc_cpr)
        err_ml, _, ber_ml           = count_bit_errors(tx_eval_bits, bits_ml)

        evm_cdc = calculate_evm(cdc_cpr_center, tx_eval_syms)
        evm_ml  = calculate_evm(rx_ml_norm, tx_eval_syms)
        phi_deg = np.degrees(phi_rad)

        res['dist_no_eq_ber'].append(ber_no_eq)
        res['dist_cdc_raw_ber'].append(ber_cdc_raw)
        res['dist_cdc_cpr_ber'].append(ber_cdc_cpr)
        res['dist_ml_ber'].append(ber_ml)
        res['dist_cdc_cpr_evm'].append(evm_cdc)
        res['dist_ml_evm'].append(evm_ml)
        res['dist_cpr_phase_deg'].append(phi_deg)

        print(f"  {dist_km:<10} {ber_no_eq:<13.4f} {ber_cdc_raw:<13.4e} {ber_cdc_cpr:<13.4e} {ber_ml:<13.4e} {evm_cdc:<10.2f} {evm_ml:<10.2f}", flush=True)
        res['csv_rows'].append(f"{modulation},{dist_km},{eval_distance_power},{n_b},{err_no_eq},{ber_no_eq:.4e},{err_cdc_raw},{ber_cdc_raw:.4e},{err_cdc_cpr},{ber_cdc_cpr:.4e},{evm_cdc:.2f},{err_ml},{ber_ml:.4e},{evm_ml:.2f},{phi_deg:.2f}")

        if dist_km == 400:
            res['sample_constellations'] = {
                'tx': tx_eval_syms,
                'no_eq': no_eq_center,
                'cdc_raw': cdc_raw_center,
                'cdc_cpr': cdc_cpr_center,
                'ml': rx_ml_norm,
                'phi_deg': phi_deg,
            }

    # 2. Power Sweep
    print(f"\n[{modulation}] Power Sweep ({len(POWERS_DBM)} points) at {eval_power_distance:.0f} km...")
    print(f"  {'Power (dBm)':<12} {'No-EQ BER':<13} {'CDC (Raw)':<13} {'CDC (+CPR)':<13} {'NN-EQ BER':<13} {'CDC EVM%':<10} {'NN EVM%':<10}")
    print("  " + "-" * 84)

    for p_dBm in POWERS_DBM:
        burst = simulate_burst(
            n_symbols=BURST_LEN_TEST,
            modulation=modulation,
            distance_km=eval_power_distance,
            power_dBm=p_dBm,
            data_seed=7000 + int(p_dBm * 10),
            noise_seed=90000 + int(p_dBm * 10)
        )
        tx_bits = burst['bits']
        tx_syms = burst['syms']

        rx_raw_no_eq    = extract_symbols_and_normalize(burst['rx_field'], BURST_LEN_TEST)
        rx_norm_cdc_raw = extract_symbols_and_normalize(burst['rx_field_cdc'], BURST_LEN_TEST)
        rx_norm_cdc_cpr, phi_rad = carrier_phase_recovery(rx_norm_cdc_raw, modulation=modulation, pilot_tx=tx_syms)

        rx_ml_eq = equalize(model, rx_norm_cdc_cpr, scaler=scaler, window=WINDOW_SIZE)
        rx_ml_norm, _ = normalize_symbols(rx_ml_eq)

        n_eval = len(rx_ml_norm)
        tx_eval_syms = tx_syms[half : half + n_eval]
        tx_eval_bits = tx_bits[half * bps : (half + n_eval) * bps]

        no_eq_center   = rx_raw_no_eq[half : half + n_eval]
        cdc_raw_center = rx_norm_cdc_raw[half : half + n_eval]
        cdc_cpr_center = rx_norm_cdc_cpr[half : half + n_eval]

        bits_no_eq   = demodulate(no_eq_center, modulation)
        bits_cdc_raw = demodulate(cdc_raw_center, modulation)
        bits_cdc_cpr = demodulate(cdc_cpr_center, modulation)
        bits_ml      = demodulate(rx_ml_norm, modulation)

        err_no_eq, n_b, ber_no_eq   = count_bit_errors(tx_eval_bits, bits_no_eq)
        err_cdc_raw, _, ber_cdc_raw = count_bit_errors(tx_eval_bits, bits_cdc_raw)
        err_cdc_cpr, _, ber_cdc_cpr = count_bit_errors(tx_eval_bits, bits_cdc_cpr)
        err_ml, _, ber_ml           = count_bit_errors(tx_eval_bits, bits_ml)

        evm_cdc = calculate_evm(cdc_cpr_center, tx_eval_syms)
        evm_ml  = calculate_evm(rx_ml_norm, tx_eval_syms)
        phi_deg = np.degrees(phi_rad)

        res['pwr_no_eq_ber'].append(ber_no_eq)
        res['pwr_cdc_raw_ber'].append(ber_cdc_raw)
        res['pwr_cdc_cpr_ber'].append(ber_cdc_cpr)
        res['pwr_ml_ber'].append(ber_ml)
        res['pwr_cdc_cpr_evm'].append(evm_cdc)
        res['pwr_ml_evm'].append(evm_ml)
        res['pwr_cpr_phase_deg'].append(phi_deg)

        print(f"  {p_dBm:<12.1f} {ber_no_eq:<13.4f} {ber_cdc_raw:<13.4e} {ber_cdc_cpr:<13.4e} {ber_ml:<13.4e} {evm_cdc:<10.2f} {evm_ml:<10.2f}", flush=True)
        res['csv_rows'].append(f"{modulation},400,{p_dBm},{n_b},{err_no_eq},{ber_no_eq:.4e},{err_cdc_raw},{ber_cdc_raw:.4e},{err_cdc_cpr},{ber_cdc_cpr:.4e},{evm_cdc:.2f},{err_ml},{ber_ml:.4e},{evm_ml:.2f},{phi_deg:.2f}")

        if p_dBm == 4.0:
            res['sample_constellations'] = {
                'tx': tx_eval_syms,
                'no_eq': no_eq_center,
                'cdc_raw': cdc_raw_center,
                'cdc_cpr': cdc_cpr_center,
                'ml': rx_ml_norm,
                'phi_deg': phi_deg,
            }

    # Convert numeric lists to arrays
    for k in list(res.keys()):
        if isinstance(res[k], list) and k != 'csv_rows':
            res[k] = np.array(res[k])

    res['n_bits'] = n_b
    return res


def run_benchmark():
    t_start = time.time()
    print("=" * 84, flush=True)
    print(" AI-Based Optical Link Impairment Compensation: Comprehensive Dual-Modulation Benchmark", flush=True)
    print("=" * 84, flush=True)

    # ─── 1. Train models for QPSK and 16-QAM ───────────────────────────────────
    qpsk_model, qpsk_scaler, qpsk_history = train_equalizer_for_modulation('QPSK')
    qam_model, qam_scaler, qam_history = train_equalizer_for_modulation('16-QAM')

    # Save training curves & histories
    fig_tr_qpsk = plot_training_curve(qpsk_history, 'Neural Network Training Curve (QPSK)')
    save_figure(fig_tr_qpsk, os.path.join(RESULTS_DIR, 'training_curve_qpsk.png'))
    save_figure(fig_tr_qpsk, os.path.join(RESULTS_DIR, 'training_curve.png'))  # Default copy
    np.save(os.path.join(RESULTS_DIR, 'training_history_qpsk.npy'), qpsk_history)
    np.save(os.path.join(RESULTS_DIR, 'training_history.npy'), qpsk_history)

    fig_tr_qam = plot_training_curve(qam_history, 'Neural Network Training Curve (16-QAM)')
    save_figure(fig_tr_qam, os.path.join(RESULTS_DIR, 'training_curve_16qam.png'))
    np.save(os.path.join(RESULTS_DIR, 'training_history_16qam.npy'), qam_history)

    # ─── 2. Evaluate Sweeps ───────────────────────────────────────────────────
    # QPSK evaluated at +2.0 dBm distance sweep (standard benchmark)
    qpsk_data = evaluate_modulation_sweep('QPSK', qpsk_model, qpsk_scaler, eval_distance_power=2.0)

    # 16-QAM evaluated at +4.0 dBm distance sweep (nonlinear distortion benchmark)
    qam_data = evaluate_modulation_sweep('16-QAM', qam_model, qam_scaler, eval_distance_power=4.0)

    # ─── 3. Save Figures ──────────────────────────────────────────────────────
    # QPSK Figures
    fig_dist_qpsk = plot_ber_vs_distance(
        qpsk_data['distances'], qpsk_data['dist_no_eq_ber'],
        bers_cdc=qpsk_data['dist_cdc_cpr_ber'], bers_ml=qpsk_data['dist_ml_ber'],
        n_bits=qpsk_data['n_bits'], title='BER vs Distance — QPSK (+2.0 dBm)'
    )
    save_figure(fig_dist_qpsk, os.path.join(RESULTS_DIR, 'ber_vs_distance_qpsk.png'))

    fig_pwr_qpsk = plot_ber_vs_power(
        qpsk_data['powers'], qpsk_data['pwr_no_eq_ber'],
        qpsk_data['pwr_cdc_cpr_ber'], qpsk_data['pwr_ml_ber'],
        distance_km=400.0, n_bits=qpsk_data['n_bits'], title='BER vs Power — QPSK (400 km)'
    )
    save_figure(fig_pwr_qpsk, os.path.join(RESULTS_DIR, 'ber_vs_power_qpsk.png'))

    fig_evm_dist_qpsk = plot_evm_vs_distance(
        qpsk_data['distances'], qpsk_data['dist_cdc_cpr_evm'], qpsk_data['dist_ml_evm'],
        title='Constellation EVM vs Distance — QPSK (+2.0 dBm)'
    )
    save_figure(fig_evm_dist_qpsk, os.path.join(RESULTS_DIR, 'evm_vs_distance_qpsk.png'))

    fig_evm_pwr_qpsk = plot_evm_vs_power(
        qpsk_data['powers'], qpsk_data['pwr_cdc_cpr_evm'], qpsk_data['pwr_ml_evm'],
        distance_km=400.0, title='Constellation EVM vs Power — QPSK (400 km)'
    )
    save_figure(fig_evm_pwr_qpsk, os.path.join(RESULTS_DIR, 'evm_vs_power_qpsk.png'))

    # 16-QAM Figures (Demonstrating significant nonlinear BER and EVM separation)
    fig_dist_qam = plot_ber_vs_distance(
        qam_data['distances'], qam_data['dist_no_eq_ber'],
        bers_cdc=qam_data['dist_cdc_cpr_ber'], bers_ml=qam_data['dist_ml_ber'],
        n_bits=qam_data['n_bits'], title='BER vs Distance — 16-QAM (+4.0 dBm Nonlinear Regime)'
    )
    save_figure(fig_dist_qam, os.path.join(RESULTS_DIR, 'ber_vs_distance_16qam.png'))
    save_figure(fig_dist_qam, os.path.join(RESULTS_DIR, 'ber_vs_distance.png'))  # Primary highlight

    fig_pwr_qam = plot_ber_vs_power(
        qam_data['powers'], qam_data['pwr_no_eq_ber'],
        qam_data['pwr_cdc_cpr_ber'], qam_data['pwr_ml_ber'],
        distance_km=400.0, n_bits=qam_data['n_bits'], title='BER vs Power — 16-QAM (400 km Nonlinear Penalty)'
    )
    save_figure(fig_pwr_qam, os.path.join(RESULTS_DIR, 'ber_vs_power_16qam.png'))
    save_figure(fig_pwr_qam, os.path.join(RESULTS_DIR, 'ber_vs_power.png'))      # Primary highlight

    fig_evm_dist_qam = plot_evm_vs_distance(
        qam_data['distances'], qam_data['dist_cdc_cpr_evm'], qam_data['dist_ml_evm'],
        title='Constellation EVM vs Distance — 16-QAM (+4.0 dBm)'
    )
    save_figure(fig_evm_dist_qam, os.path.join(RESULTS_DIR, 'evm_vs_distance_16qam.png'))
    save_figure(fig_evm_dist_qam, os.path.join(RESULTS_DIR, 'evm_vs_distance.png'))

    fig_evm_pwr_qam = plot_evm_vs_power(
        qam_data['powers'], qam_data['pwr_cdc_cpr_evm'], qam_data['pwr_ml_evm'],
        distance_km=400.0, title='Constellation EVM vs Power — 16-QAM (400 km)'
    )
    save_figure(fig_evm_pwr_qam, os.path.join(RESULTS_DIR, 'evm_vs_power_16qam.png'))
    save_figure(fig_evm_pwr_qam, os.path.join(RESULTS_DIR, 'evm_vs_power.png'))

    # Constellation Plots (16-QAM and QPSK at 400 km, +4 dBm)
    if 'sample_constellations' in qam_data and qam_data['sample_constellations']:
        sc = qam_data['sample_constellations']
        phi_rot = sc.get('phi_deg', 40.0)

        fig_grid_16 = plot_constellation_grid(
            sc['tx'], sc['cdc_raw'], sc['cdc_cpr'], sc['ml'],
            title=f'16-QAM Constellation Comparison (400 km, +4 dBm, Kerr Rotation={phi_rot:+.1f}°)'
        )
        save_figure(fig_grid_16, os.path.join(RESULTS_DIR, 'constellation_grid_16qam.png'))
        save_figure(fig_grid_16, os.path.join(RESULTS_DIR, 'constellation_grid.png'))

        fig_after_16 = plot_constellation(
            [sc['tx'], sc['cdc_raw'], sc['cdc_cpr'], sc['ml']],
            ['Ideal Transmitted', f'Linear FD-CDC (Raw, {phi_rot:+.1f}°)', 'Linear FD-CDC (After CPR)', 'Neural Network Equalizer'],
            title='16-QAM Constellation Stages (400 km, +4 dBm)'
        )
        save_figure(fig_after_16, os.path.join(RESULTS_DIR, 'constellation_after_16qam.png'))
        save_figure(fig_after_16, os.path.join(RESULTS_DIR, 'constellation_after.png'))

        fig_before_16 = plot_constellation(
            [sc['tx'], sc['no_eq']],
            ['Ideal Transmitted', 'Uncompensated Received'],
            title='16-QAM Constellation Before Equalization (400 km, +4 dBm)'
        )
        save_figure(fig_before_16, os.path.join(RESULTS_DIR, 'constellation_before.png'))

    if 'sample_constellations' in qpsk_data and qpsk_data['sample_constellations']:
        sc_q = qpsk_data['sample_constellations']
        phi_rot_q = sc_q.get('phi_deg', 40.0)
        fig_grid_qpsk = plot_constellation_grid(
            sc_q['tx'], sc_q['cdc_raw'], sc_q['cdc_cpr'], sc_q['ml'],
            title=f'QPSK Constellation Comparison (400 km, +4 dBm, Kerr Rotation={phi_rot_q:+.1f}°)'
        )
        save_figure(fig_grid_qpsk, os.path.join(RESULTS_DIR, 'constellation_grid_qpsk.png'))

    # ─── 4. Save Comprehensive CSV and NumPy Results ──────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'benchmark_results.csv')
    csv_header = "modulation,distance_km,power_dBm,bits_compared,no_eq_errors,no_eq_ber,cdc_raw_errors,cdc_raw_ber,cdc_cpr_errors,cdc_cpr_ber,cdc_cpr_evm,ml_errors,ml_ber,ml_evm,cpr_phase_deg"
    all_csv_rows = [csv_header] + qpsk_data['csv_rows'] + qam_data['csv_rows']
    with open(csv_path, 'w') as f:
        f.write("\n".join(all_csv_rows) + "\n")
    print(f"\n  Comprehensive CSV saved -> {csv_path}")

    ber_curve_data = {
        # Top-level backwards compatibility for dashboard:
        'distances': qam_data['distances'],
        'ber_noml': qam_data['dist_no_eq_ber'],
        'ber_cdc': qam_data['dist_cdc_cpr_ber'],
        'ber_ml': qam_data['dist_ml_ber'],
        'powers': qam_data['powers'],
        'power_bers_no_eq': qam_data['pwr_no_eq_ber'],
        'power_bers_cdc': qam_data['pwr_cdc_cpr_ber'],
        'power_bers_ml': qam_data['pwr_ml_ber'],
        # Full per-modulation dictionaries:
        'QPSK': {
            'distances': qpsk_data['distances'],
            'dist_no_eq_ber': qpsk_data['dist_no_eq_ber'],
            'dist_cdc_raw_ber': qpsk_data['dist_cdc_raw_ber'],
            'dist_cdc_cpr_ber': qpsk_data['dist_cdc_cpr_ber'],
            'dist_ml_ber': qpsk_data['dist_ml_ber'],
            'dist_cdc_cpr_evm': qpsk_data['dist_cdc_cpr_evm'],
            'dist_ml_evm': qpsk_data['dist_ml_evm'],
            'powers': qpsk_data['powers'],
            'pwr_no_eq_ber': qpsk_data['pwr_no_eq_ber'],
            'pwr_cdc_raw_ber': qpsk_data['pwr_cdc_raw_ber'],
            'pwr_cdc_cpr_ber': qpsk_data['pwr_cdc_cpr_ber'],
            'pwr_ml_ber': qpsk_data['pwr_ml_ber'],
            'pwr_cdc_cpr_evm': qpsk_data['pwr_cdc_cpr_evm'],
            'pwr_ml_evm': qpsk_data['pwr_ml_evm'],
            'n_bits': qpsk_data['n_bits'],
        },
        '16-QAM': {
            'distances': qam_data['distances'],
            'dist_no_eq_ber': qam_data['dist_no_eq_ber'],
            'dist_cdc_raw_ber': qam_data['dist_cdc_raw_ber'],
            'dist_cdc_cpr_ber': qam_data['dist_cdc_cpr_ber'],
            'dist_ml_ber': qam_data['dist_ml_ber'],
            'dist_cdc_cpr_evm': qam_data['dist_cdc_cpr_evm'],
            'dist_ml_evm': qam_data['dist_ml_evm'],
            'powers': qam_data['powers'],
            'pwr_no_eq_ber': qam_data['pwr_no_eq_ber'],
            'pwr_cdc_raw_ber': qam_data['pwr_cdc_raw_ber'],
            'pwr_cdc_cpr_ber': qam_data['pwr_cdc_cpr_ber'],
            'pwr_ml_ber': qam_data['pwr_ml_ber'],
            'pwr_cdc_cpr_evm': qam_data['pwr_cdc_cpr_evm'],
            'pwr_ml_evm': qam_data['pwr_ml_evm'],
            'n_bits': qam_data['n_bits'],
        }
    }
    np.save(os.path.join(RESULTS_DIR, 'ber_curve.npy'), ber_curve_data)
    print(f"  Comprehensive NumPy archive saved -> {os.path.join(RESULTS_DIR, 'ber_curve.npy')}")

    print(f"\nBenchmark completed successfully in {time.time() - t_start:.1f} s!")


if __name__ == '__main__':
    run_benchmark()
