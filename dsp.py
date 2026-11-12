"""
dsp.py — Conventional Digital Signal Processing (DSP) Module
=============================================================
Provides classical DSP algorithms for optical communications:
- Frequency-domain chromatic dispersion compensation (FD-CDC)
- Linear adaptive LMS equalizer (FFE)
- Continuous waveform reconstruction from equalized symbols (for eye diagrams)
- Signal quality metrics: EVM (Error Vector Magnitude), SER (Symbol Error Rate)
"""
import numpy as np
from scipy.signal import upfirdn
from tx import rrc_taps, SPS, ROLLOFF, SPAN, QPSK_SYMBOLS, QAM16_SYMBOLS
from fiber import _beta2_s2_km


def compensate_chromatic_dispersion(signal: np.ndarray,
                                    distance_km: float,
                                    fiber_params: dict,
                                    fs: float) -> np.ndarray:
    """
    Frequency-Domain Chromatic Dispersion Compensation (FD-CDC).

    Inverts the linear group-velocity dispersion accumulated over `distance_km`:
        H_CDC(ω) = exp( + j (β₂ / 2) ω² L )

    Parameters
    ----------
    signal : np.ndarray, complex
        Optical field in [√W].
    distance_km : float
        Total fiber propagation distance [km].
    fiber_params : dict
        Contains 'D_ps_nm_km' and 'wavelength_nm'.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    compensated : np.ndarray, complex
        Dispersion-compensated optical field in [√W].
    """
    if distance_km <= 0 or abs(fiber_params.get('D_ps_nm_km', 0.0)) < 1e-12:
        return signal.copy()

    beta2 = _beta2_s2_km(fiber_params['D_ps_nm_km'], fiber_params['wavelength_nm'])  # s²/km
    n = len(signal)
    dt = 1.0 / fs
    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)  # rad/s

    # Invert the dispersion phase shift accumulated over distance L
    # H_CDC(ω) = exp(+j (β₂ / 2) ω² L)
    h_cdc = np.exp(1j * (beta2 / 2.0) * (omega ** 2) * distance_km)
    return np.fft.ifft(np.fft.fft(signal) * h_cdc)


def reconstruct_waveform(symbols: np.ndarray,
                         sps: int = SPS,
                         rolloff: float = ROLLOFF,
                         span: int = SPAN) -> np.ndarray:
    """
    Reconstruct a continuous-time waveform from discrete symbols by
    upsampling and pulse-shaping with the full Raised Cosine filter
    (RRC applied twice: Tx pulse shaping + Rx matched filter).

    The FD-CDC eye diagram path naturally passes through two RRC filters
    (Tx RRC → channel → Rx matched filter RRC = Raised Cosine), so the
    reconstruction must also apply two RRC filters for a fair comparison.

    Parameters
    ----------
    symbols : np.ndarray, complex
        Equalized discrete symbol sequence.
    sps : int
        Samples per symbol.
    rolloff : float
        RRC roll-off factor.
    span : int
        RRC filter span.

    Returns
    -------
    waveform : np.ndarray, complex
        Reconstructed analog waveform at rate sps × Rs with Raised Cosine
        pulse shape (zero ISI at symbol sampling instants).
    """
    h = rrc_taps(sps, rolloff, span)
    # 1st RRC: Tx pulse shaping (upsample + filter)
    tx_shaped = upfirdn(h, symbols, up=sps).astype(complex)
    # 2nd RRC: Rx matched filter (no resampling, just filter)
    return upfirdn(h, tx_shaped, up=1, down=1).astype(complex)


def calculate_evm(rx_symbols: np.ndarray, tx_symbols: np.ndarray) -> float:
    """
    Calculate Error Vector Magnitude (EVM) in percent [%].

        EVM_RMS = sqrt( mean( |rx - tx|² ) / mean( |tx|² ) ) * 100%

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Received or equalized symbols (normalized).
    tx_symbols : np.ndarray, complex
        Ideal transmitted reference symbols.

    Returns
    -------
    evm_pct : float
        RMS EVM percentage.
    """
    n = min(len(rx_symbols), len(tx_symbols))
    if n == 0:
        return 100.0
    err = rx_symbols[:n] - tx_symbols[:n]
    ref_pwr = np.mean(np.abs(tx_symbols[:n]) ** 2)
    if ref_pwr < 1e-30:
        return 100.0
    return float(np.sqrt(np.mean(np.abs(err) ** 2) / ref_pwr) * 100.0)


def calculate_ser(rx_symbols: np.ndarray,
                  tx_symbols: np.ndarray,
                  modulation: str = 'QPSK') -> float:
    """
    Calculate Symbol Error Rate (SER).

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Received/equalized symbols (normalized).
    tx_symbols : np.ndarray, complex
        Ideal transmitted symbols.
    modulation : {'QPSK', '16-QAM'}

    Returns
    -------
    ser : float ∈ [0, 1]
    """
    ref = QPSK_SYMBOLS if modulation == 'QPSK' else QAM16_SYMBOLS
    n = min(len(rx_symbols), len(tx_symbols))
    if n == 0:
        return 1.0

    rx_dec = ref[np.argmin(np.abs(rx_symbols[:n, None] - ref[None, :]) ** 2, axis=1)]
    tx_dec = ref[np.argmin(np.abs(tx_symbols[:n, None] - ref[None, :]) ** 2, axis=1)]
    return float(np.mean(rx_dec != tx_dec))


def carrier_phase_recovery(rx_symbols: np.ndarray,
                           modulation: str = 'QPSK',
                           pilot_tx: np.ndarray | None = None,
                           n_pilots: int = 64) -> tuple[np.ndarray, float]:
    """
    Continuous Carrier Phase Recovery (CPR) for coherent optical communications.

    Compensates for accumulated Kerr nonlinear phase rotation (SPM) and residual
    phase offsets. Supports:
    1. Pilot-aided / reference-aided phase estimation (using known preamble or pilot sequence)
    2. Blind Viterbi-Viterbi 4th-power phase estimation for QPSK
    3. Blind minimum-distance angular grid search for 16-QAM

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Normalized received symbols (dimensionless, unit average power).
    modulation : {'QPSK', '16-QAM'}
        Modulation format.
    pilot_tx : np.ndarray or None
        Known transmitted reference symbols. If provided, the first `n_pilots`
        symbols are used to compute the optimal phase angle and resolve quadrant ambiguity.
    n_pilots : int
        Number of pilot symbols to use for phase estimation.

    Returns
    -------
    symbols_cpr : np.ndarray, complex
        Phase-corrected symbol sequence: symbols_cpr = rx_symbols * exp(-j * phi_est).
    phi_est : float
        Estimated carrier phase offset in radians.
    """
    ref = QPSK_SYMBOLS if modulation == 'QPSK' else QAM16_SYMBOLS

    # 1. Pilot-aided / reference-aided phase estimation
    if pilot_tx is not None and len(pilot_tx) > 0:
        n_p = min(n_pilots, len(pilot_tx), len(rx_symbols))
        corr = np.mean(rx_symbols[:n_p] * np.conj(pilot_tx[:n_p]))
        phi_est = float(np.angle(corr))
        return rx_symbols * np.exp(-1j * phi_est), phi_est

    # 2. Blind Carrier Phase Recovery
    if modulation == 'QPSK':
        # Viterbi-Viterbi 4th-power estimator
        # E[r^4] = E[(s * exp(j*phi))^4] = E[s^4] * exp(j*4*phi)
        # For QPSK: s^4 = exp(j*pi) = -1. Thus angle(r^4) = 4*phi + pi.
        v4 = np.mean(rx_symbols ** 4)
        base_phi = (np.angle(v4) - np.pi) / 4.0
        # 4 ambiguity candidates
        candidates = [(base_phi + k * np.pi / 2.0) for k in range(4)]
        # Wrap to (-pi, pi]
        candidates = [(p + np.pi) % (2.0 * np.pi) - np.pi for p in candidates]
        # Select candidate that minimizes Euclidean distance to QPSK constellation
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


def lms_equalizer(rx_symbols: np.ndarray,
                  tx_symbols: np.ndarray,
                  n_taps: int = 15,
                  mu: float = 0.01) -> np.ndarray:
    """
    Linear Least Mean Squares (LMS) adaptive FIR equalizer.
    Serves as an adaptive linear DSP benchmark.

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Received symbols (normalized).
    tx_symbols : np.ndarray, complex
        Reference transmitted symbols for supervised training.
    n_taps : int
        Number of equalizer FIR taps (odd integer).
    mu : float
        LMS step-size (adaptation rate).

    Returns
    -------
    eq_symbols : np.ndarray, complex
        Equalized output symbol sequence.
    """
    if n_taps % 2 == 0:
        n_taps += 1
    half = n_taps // 2

    n = min(len(rx_symbols), len(tx_symbols))
    # Initialize center-tap to 1.0, others to 0.0
    w = np.zeros(n_taps, dtype=complex)
    w[half] = 1.0

    out = np.zeros(n - 2 * half, dtype=complex)
    for i in range(len(out)):
        c = i + half
        x = rx_symbols[c - half : c + half + 1][::-1]  # tap vector
        y = np.dot(w, x)
        d = tx_symbols[c]
        err = d - y
        w += mu * err * np.conj(x)
        out[i] = y

    return out

