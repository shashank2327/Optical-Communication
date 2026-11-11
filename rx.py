"""
rx.py — Coherent Optical Receiver Module
=========================================
Implements the full coherent receiver DSP pipeline:
    Received optical field [√W]
        → Matched RRC filter           (Nyquist RC channel, SNR maximization)
        → Downsample & Timing Trim     (Deterministic group-delay compensation, 1 sps)
        → Amplitude/Gain Normalization (Physical [√W] → Dimensionless unit constellation)
        → Carrier Phase Alignment      (4-fold rotational ambiguity resolution)
        → Hard-Decision Demapping      (Voronoi slicing against Gray constellation)
        → Performance Evaluation       (BER, SER, EVM, error count accounting)

Units and Conventions:
- Input `signal` is in physical optical field units of [√W].
- Matched filtering with unit-energy RRC filter yields optical field samples.
- `normalize_symbols()` scales the received samples to unit average power (E[|s|²] = 1.0)
  so that Euclidean distance slicing matches the ideal constellation coordinates.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import upfirdn

from tx import (rrc_taps, SPS, ROLLOFF, SPAN,
                QPSK_SYMBOLS, QAM16_SYMBOLS,
                QPSK_BITS,   QAM16_BITS)


# ─── 1. Matched filter ────────────────────────────────────────────────────────

def matched_filter(signal: np.ndarray,
                   sps: int = SPS,
                   rolloff: float = ROLLOFF,
                   span: int = SPAN) -> np.ndarray:
    """
    Apply the RRC matched filter (identical taps to Tx filter).

    The combined Tx × Rx response is a raised-cosine filter with roll-off β,
    ensuring zero ISI at integer symbol sampling instants in back-to-back links.

    Parameters
    ----------
    signal : np.ndarray, complex
        Received optical field at rate sps × Rs [√W].
    sps : int
        Samples per symbol.
    rolloff : float
        RRC roll-off factor β.
    span : int
        Filter span in symbols.

    Returns
    -------
    filtered : np.ndarray, complex
        Filtered signal (same sampling rate as input).
    """
    h = rrc_taps(sps, rolloff, span)
    return upfirdn(h, signal, up=1, down=1).astype(complex)


# ─── 2. Downsampling & Timing ──────────────────────────────────────────────────

def downsample(signal: np.ndarray,
               sps: int = SPS,
               offset: int = 0,
               span: int = SPAN) -> np.ndarray:
    """
    Compensate the deterministic combined filter delay and downsample to 1 sample/symbol.

    Total deterministic group delay:
        delay = (Tx span + Rx span) × sps = 2 × SPAN × sps  samples.

    Parameters
    ----------
    signal : np.ndarray, complex
        Matched-filter output signal.
    sps : int
        Samples per symbol.
    offset : int
        Sampling phase within [0, sps). Default is 0 (exact Nyquist instant).
    span : int
        Filter span in symbols.

    Returns
    -------
    symbols : np.ndarray, complex
        Downsampled symbol sequence at 1 sample per symbol.
    """
    delay = 2 * span * sps
    if len(signal) <= delay:
        raise ValueError(f"Signal length ({len(signal)}) must exceed filter delay ({delay}).")

    sig = signal[delay:]
    return sig[offset::sps]


# ─── 3. Amplitude & Gain Normalization ─────────────────────────────────────────

def normalize_symbols(rx_symbols: np.ndarray,
                      method: str = 'rms',
                      pilot_rx: np.ndarray | None = None,
                      pilot_tx: np.ndarray | None = None) -> tuple[np.ndarray, float | complex]:
    """
    Normalize received symbol amplitude from physical [√W] scale to
    the dimensionless unit-average-power constellation scale (E[|s|²] = 1.0).

    This resolves the unit mismatch between physical optical power and
    dimensionless constellation decision boundaries.

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Unnormalized symbols in [√W].
    method : {'rms', 'pilot'}
        - 'rms': Blind RMS power normalization (divides by sample RMS magnitude).
                 Preserves unit power for both QPSK and normalized 16-QAM.
        - 'pilot': Data-aided estimation using a known pilot prefix.
    pilot_rx : np.ndarray or None
        Received pilot symbols (required if method='pilot').
    pilot_tx : np.ndarray or None
        Known transmitted pilot symbols (required if method='pilot').

    Returns
    -------
    normalized_symbols : np.ndarray, complex
        Symbols scaled such that average power is approximately 1.0.
    gain : float or complex
        Estimated channel gain factor applied: s_norm = s_raw / gain.
    """
    if method == 'pilot':
        if pilot_rx is None or pilot_tx is None or len(pilot_rx) == 0:
            raise ValueError("pilot_rx and pilot_tx must be non-empty for pilot normalization.")
        h_est = np.mean(pilot_rx / pilot_tx)
        if abs(h_est) < 1e-15:
            return rx_symbols.copy(), 1.0
        return rx_symbols / h_est, h_est
    else:
        # Blind RMS power normalization
        pwr = np.mean(np.abs(rx_symbols) ** 2)
        if pwr < 1e-30:
            return rx_symbols.copy(), 1.0
        rms = float(np.sqrt(pwr))
        return rx_symbols / rms, rms


# ─── 4. Carrier Phase Recovery ────────────────────────────────────────────────

def phase_correct(rx_symbols: np.ndarray,
                  modulation: str = 'QPSK',
                  pilot_rx: np.ndarray | None = None,
                  pilot_tx: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """
    Continuous Carrier Phase Recovery (CPR) and 4-fold rotational ambiguity resolution.

    Compensates for accumulated Kerr nonlinear phase rotation (SPM) and residual
    phase offsets. Supports:
    1. Pilot-aided / reference-aided phase estimation (using known preamble or pilot sequence)
    2. Blind Viterbi-Viterbi 4th-power phase estimation for QPSK
    3. Blind minimum-distance angular grid search for 16-QAM

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Normalized received symbols (unit average power).
    modulation : {'QPSK', '16-QAM'}
    pilot_rx : np.ndarray or None
        Received pilot symbols.
    pilot_tx : np.ndarray or None
        Known transmitted pilot symbols.

    Returns
    -------
    symbols_corrected : np.ndarray, complex
        Phase-aligned symbols.
    best_phase_rad : float
        The estimated rotation angle in radians (subtracted to align).
    """
    # 1. Pilot-aided / reference-aided phase estimation
    if pilot_tx is not None and len(pilot_tx) > 0:
        p_rx = pilot_rx if (pilot_rx is not None and len(pilot_rx) > 0) else rx_symbols
        n_p = min(len(p_rx), len(pilot_tx))
        corr = np.mean(p_rx[:n_p] * np.conj(pilot_tx[:n_p]))
        phi_est = float(np.angle(corr))
        return rx_symbols * np.exp(-1j * phi_est), phi_est

    ref = QPSK_SYMBOLS if modulation == 'QPSK' else QAM16_SYMBOLS

    # 2. Blind Carrier Phase Recovery
    if modulation == 'QPSK':
        # Viterbi-Viterbi 4th-power estimator
        v4 = np.mean(rx_symbols ** 4)
        base_phi = (np.angle(v4) - np.pi) / 4.0
        candidates = [(base_phi + k * np.pi / 2.0) for k in range(4)]
        candidates = [(p + np.pi) % (2.0 * np.pi) - np.pi for p in candidates]
        scores = []
        sample = rx_symbols[:min(len(rx_symbols), 500)]
        for c in candidates:
            rot = sample * np.exp(-1j * c)
            dists = np.min(np.abs(rot[:, None] - ref[None, :]) ** 2, axis=1)
            scores.append(np.mean(dists))
        best_phi = float(candidates[np.argmin(scores)])
        return rx_symbols * np.exp(-1j * best_phi), best_phi
    else:
        # Blind 16-QAM minimum distance angular grid search
        grid = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        sample = rx_symbols[:min(len(rx_symbols), 500)]
        scores = []
        for c in grid:
            rot = sample * np.exp(-1j * c)
            dists = np.min(np.abs(rot[:, None] - ref[None, :]) ** 2, axis=1)
            scores.append(np.mean(dists))
        best_phi = float(grid[np.argmin(scores)])
        return rx_symbols * np.exp(-1j * best_phi), best_phi


# ─── 5. Hard-decision Demodulation ────────────────────────────────────────────

def demodulate(symbols: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
    """
    Map each normalized symbol to the nearest constellation point (hard decision)
    and return the corresponding Gray-coded bit sequence.

    Parameters
    ----------
    symbols    : np.ndarray, complex (normalized to unit power)
    modulation : {'QPSK', '16-QAM'}

    Returns
    -------
    bits : np.ndarray, uint8
        Demodulated bit sequence of length = len(symbols) × bits_per_symbol.
    """
    if modulation == 'QPSK':
        ref_syms = QPSK_SYMBOLS
        ref_bits = QPSK_BITS
    elif modulation == '16-QAM':
        ref_syms = QAM16_SYMBOLS
        ref_bits = QAM16_BITS
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    # Nearest Euclidean neighbor
    idx = np.argmin(np.abs(symbols[:, None] - ref_syms[None, :]) ** 2, axis=1)
    return ref_bits[idx].flatten()


# ─── 6. Bit Error Rate ────────────────────────────────────────────────────────

def compute_ber(tx_bits: np.ndarray,
                rx_bits: np.ndarray,
                strict: bool = False) -> float:
    """
    Compute the Bit Error Rate (BER).

    Parameters
    ----------
    tx_bits : np.ndarray, uint8
        Transmitted reference bits.
    rx_bits : np.ndarray, uint8
        Demodulated received bits.
    strict : bool
        If True, raises ValueError when array lengths differ.

    Returns
    -------
    ber : float ∈ [0.0, 1.0]
    """
    if strict and len(tx_bits) != len(rx_bits):
        raise ValueError(f"Bit length mismatch: tx has {len(tx_bits)}, rx has {len(rx_bits)} bits.")

    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 1.0

    errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    return float(errors / n)


def count_bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> tuple[int, int, float]:
    """
    Return exact bit error count, compared bit count, and BER.

    Returns
    -------
    errors : int
    total : int
    ber : float
    """
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 0, 0, 1.0
    errors = int(np.sum(tx_bits[:n] != rx_bits[:n]))
    return errors, n, float(errors / n)


# ─── 7. Full Receiver Pipeline ────────────────────────────────────────────────

def receive(signal: np.ndarray,
            tx_symbols: np.ndarray,
            tx_bits: np.ndarray,
            modulation: str = 'QPSK',
            sps: int = SPS,
            rolloff: float = ROLLOFF,
            span: int = SPAN,
            norm_method: str = 'rms',
            guard_symbols: int = 0,
            pilot_tx: np.ndarray | None = None) -> dict:
    """
    Execute the full coherent receiver pipeline.

    Steps:
        1. Matched filtering
        2. Delay compensation & Downsampling
        3. Guard interval trimming (if present)
        4. Amplitude normalization (physical √W -> unit power)
        5. Carrier phase recovery (4-fold rotation)
        6. Hard-decision demodulation
        7. Performance metrics (BER, SER, EVM)

    Parameters
    ----------
    signal : np.ndarray, complex
        Received optical field [√W].
    tx_symbols : np.ndarray, complex
        Transmitted reference symbols (unit power).
    tx_bits : np.ndarray, uint8
        Transmitted reference bits.
    modulation : {'QPSK', '16-QAM'}
    sps : int
        Samples per symbol.
    rolloff : float
        RRC roll-off factor.
    span : int
        RRC filter span.
    norm_method : {'rms', 'pilot'}
        Normalization strategy.
    guard_symbols : int
        Number of guard symbols prepended and appended to trim.

    Returns
    -------
    result : dict
        Contains received symbols, bits, BER, SER, EVM, and diagnostics.
    """
    bps = 2 if modulation == 'QPSK' else 4

    # 1. Matched filter
    mf_out = matched_filter(signal, sps, rolloff, span)

    # 2. Downsample
    rx_raw = downsample(mf_out, sps, offset=0, span=span)

    # 3. Guard symbols / alignment trimming
    if guard_symbols > 0:
        rx_raw = rx_raw[guard_symbols : -guard_symbols if guard_symbols > 0 else None]

    n = min(len(rx_raw), len(tx_symbols))
    rx_raw = rx_raw[:n]
    tx_sym = tx_symbols[:n]
    tx_b = tx_bits[:n * bps]

    # 4. Amplitude normalization
    rx_norm, gain = normalize_symbols(rx_raw, method=norm_method)

    # 5. Phase recovery
    pilot_ref = pilot_tx if pilot_tx is not None else tx_sym[:64]
    rx_aligned, phase_rot = phase_correct(rx_norm, modulation, pilot_rx=rx_norm[:len(pilot_ref)], pilot_tx=pilot_ref)

    # 6. Demodulation
    rx_bits = demodulate(rx_aligned, modulation)

    # 7. Metrics
    err_count, total_bits, ber = count_bit_errors(tx_b, rx_bits)

    # EVM calculation
    err_vec = rx_aligned - tx_sym
    ref_pwr = np.mean(np.abs(tx_sym) ** 2)
    evm_pct = float(np.sqrt(np.mean(np.abs(err_vec) ** 2) / max(ref_pwr, 1e-30)) * 100.0)

    # SER calculation
    ref = QPSK_SYMBOLS if modulation == 'QPSK' else QAM16_SYMBOLS
    rx_dec = ref[np.argmin(np.abs(rx_aligned[:, None] - ref[None, :]) ** 2, axis=1)]
    ser = float(np.mean(rx_dec != tx_sym))

    return {
        'rx_symbols': rx_aligned,           # Normalized and phase-aligned
        'rx_symbols_raw': rx_raw,           # Unnormalized [√W]
        'rx_symbols_norm': rx_norm,         # Normalized before phase alignment
        'rx_bits': rx_bits,
        'ber': ber,
        'ser': ser,
        'evm': evm_pct,
        'bit_errors': err_count,
        'bits_compared': total_bits,
        'estimated_gain': gain,
        'phase_rotation_rad': phase_rot,
        'mf_signal': mf_out,
    }
