"""
tx.py — Transmitter Module
==========================
Generates random bits, maps them to Gray-coded QPSK or 16-QAM symbols,
applies root-raised-cosine (RRC) pulse shaping, and sets the launch power.

Units and Conventions:
- Transmitted symbols s[k] are dimensionless complex numbers normalized to
  unit average power: E[|s[k]|²] = 1.0.
- Transmit waveform A(t) has physical optical field units of [√W].
- Optical power P(t) = |A(t)|² in [Watts].
- Launch power P_dBm sets the mean sample power: mean(|A(t)|²) = 1 mW × 10^(P_dBm / 10).

Pipeline:
    generate_bits → modulate → (add_guard_symbols) → pulse_shape → set_launch_power
"""
import numpy as np
from scipy.signal import upfirdn

# ─── Global signal parameters ─────────────────────────────────────────────────
SYMBOL_RATE_BD = 32e9   # Symbol rate  [Baud] (32 GBd)
SPS            = 8      # Samples per symbol
ROLLOFF        = 0.1    # RRC roll-off factor β
SPAN           = 10     # RRC filter span [symbols]; filter length = 2*SPAN*SPS + 1


# ─── Modulation look-up tables ────────────────────────────────────────────────

def _qpsk_table() -> dict:
    """
    Gray-coded QPSK: 2 bits/symbol, unit average power.
    Mapping:
        (0, 0) -> (+1 + 1j) / sqrt(2)
        (0, 1) -> (-1 + 1j) / sqrt(2)
        (1, 1) -> (-1 - 1j) / sqrt(2)
        (1, 0) -> (+1 - 1j) / sqrt(2)
    Adjacent constellation points differ by exactly 1 bit.
    Average symbol power = 1.0.
    """
    s = np.sqrt(2.0)
    return {
        (0, 0): ( 1.0 + 1.0j) / s,
        (0, 1): (-1.0 + 1.0j) / s,
        (1, 1): (-1.0 - 1.0j) / s,
        (1, 0): ( 1.0 - 1.0j) / s,
    }


def _16qam_table() -> dict:
    """
    Gray-coded 16-QAM: 4 bits/symbol, unit average power.

    Bit mapping: (b3, b2, b1, b0)
    - b3, b2 determine the In-phase (I) coordinate (Gray code 00->-3, 01->-1, 11->+3, 10->+1)
    - b1, b0 determine the Quadrature (Q) coordinate (Gray code)
    Levels: [-3, -1, +1, +3] / sqrt(10)
    Average symbol power = (2 * (9 + 1 + 1 + 9) / 4) / 10 = 1.0.
    """
    gray_order = [0, 1, 3, 2]               # 2-bit Gray code indices
    levels     = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
    table = {}
    for i, gi in enumerate(gray_order):
        for j, gj in enumerate(gray_order):
            bits = ((gi >> 1) & 1, gi & 1, (gj >> 1) & 1, gj & 1)
            table[bits] = complex(levels[i], levels[j])
    return table


_QPSK_TABLE  = _qpsk_table()
_16QAM_TABLE = _16qam_table()

# Constellation arrays exposed for rx.py and dsp.py
QPSK_SYMBOLS  = np.array(list(_QPSK_TABLE.values()),  dtype=complex)
QAM16_SYMBOLS = np.array(list(_16QAM_TABLE.values()), dtype=complex)
QPSK_BITS     = np.array(list(_QPSK_TABLE.keys()),    dtype=np.uint8)
QAM16_BITS    = np.array(list(_16QAM_TABLE.keys()),   dtype=np.uint8)


# ─── Core transmitter functions ───────────────────────────────────────────────

def generate_bits(n_symbols: int,
                  modulation: str = 'QPSK',
                  seed: int = 42) -> np.ndarray:
    """
    Generate a pseudo-random bit stream for the specified modulation format.

    Parameters
    ----------
    n_symbols : int
        Number of symbols to generate.
    modulation : {'QPSK', '16-QAM'}
        Modulation format (2 bits/symbol for QPSK, 4 bits/symbol for 16-QAM).
    seed : int
        NumPy random seed (ensures reproducibility).

    Returns
    -------
    bits : np.ndarray, shape (n_symbols * bits_per_symbol,), dtype uint8
    """
    bps = 2 if modulation == 'QPSK' else 4
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_symbols * bps, dtype=np.uint8)


def modulate(bits: np.ndarray, modulation: str = 'QPSK') -> np.ndarray:
    """
    Map a bit stream to complex IQ symbols using the Gray-coded constellation table.

    Parameters
    ----------
    bits : np.ndarray, dtype uint8
        Bit stream; length must be n_symbols × bits_per_symbol.
    modulation : {'QPSK', '16-QAM'}

    Returns
    -------
    symbols : np.ndarray, complex128, shape (n_symbols,)
        Dimensionless complex symbols with unit average power (E[|s|²] = 1.0).
    """
    if modulation == 'QPSK':
        bps, table = 2, _QPSK_TABLE
    elif modulation == '16-QAM':
        bps, table = 4, _16QAM_TABLE
    else:
        raise ValueError(f"Unsupported modulation: {modulation}. Choose 'QPSK' or '16-QAM'.")

    if len(bits) % bps != 0:
        raise ValueError(f"Bit stream length ({len(bits)}) must be a multiple of bits-per-symbol ({bps}).")

    rows = bits.reshape(-1, bps)
    return np.array([table[tuple(r)] for r in rows], dtype=complex)


def add_guard_symbols(symbols: np.ndarray,
                      n_guard: int = 64,
                      modulation: str = 'QPSK',
                      seed: int = 0) -> tuple[np.ndarray, int]:
    """
    Prepend and append guard symbols around payload symbols.
    Prevents FFT circular-boundary wraparound in SSFM from contaminating the payload.

    Parameters
    ----------
    symbols : np.ndarray, complex
        Payload symbols to protect.
    n_guard : int
        Number of guard symbols prepended and appended.
    modulation : str
        Modulation format for generating guard symbols.
    seed : int
        Seed for guard symbol generation.

    Returns
    -------
    symbols_with_guard : np.ndarray, complex
        Padded symbol array of length N + 2 * n_guard.
    n_guard : int
        The guard length used.
    """
    if n_guard <= 0:
        return symbols, 0

    guard_bits_pre  = generate_bits(n_guard, modulation, seed=seed)
    guard_bits_post = generate_bits(n_guard, modulation, seed=seed + 1)
    guard_pre  = modulate(guard_bits_pre, modulation)
    guard_post = modulate(guard_bits_post, modulation)

    return np.concatenate([guard_pre, symbols, guard_post]), n_guard


def rrc_taps(sps: int = SPS,
             rolloff: float = ROLLOFF,
             span: int = SPAN) -> np.ndarray:
    """
    Compute root-raised-cosine (RRC) filter taps.

    The RRC filter is the square root of the raised-cosine (Nyquist) filter.
    Applying the same filter twice (at Tx and Rx) gives the full RC response,
    ensuring zero ISI at integer symbol-period sampling points.

    Parameters
    ----------
    sps     : int   – samples per symbol
    rolloff : float – roll-off factor β ∈ [0, 1]
    span    : int   – filter span in symbols (half-length = span * sps)

    Returns
    -------
    h : np.ndarray, real, length = 2 * span * sps + 1
        Unit-energy RRC impulse response (sum(h²) = 1.0).
    """
    N    = span * sps
    t    = np.arange(-N, N + 1) / sps   # normalised time (symbol periods)
    beta = rolloff
    h    = np.zeros(len(t), dtype=float)

    for k, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[k] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 1e-10 and abs(abs(ti) - 1.0 / (2.0 * beta)) < 1e-6:
            h[k] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[k] = num / den if abs(den) > 1e-10 else 0.0

    h /= np.sqrt(np.sum(h ** 2))        # normalise to unit energy: sum(h²) = 1.0
    return h


def pulse_shape(symbols: np.ndarray,
                sps: int     = SPS,
                rolloff: float = ROLLOFF,
                span: int    = SPAN) -> np.ndarray:
    """
    Upsample discrete symbols by `sps` and apply the RRC transmit filter.

    Parameters
    ----------
    symbols : np.ndarray, complex
        Complex symbol sequence (length N).
    sps     : int   – upsample factor
    rolloff : float – RRC roll-off factor
    span    : int   – RRC filter span in symbols

    Returns
    -------
    signal : np.ndarray, complex
        Pulse-shaped signal at rate sps × Rs  (length ≈ N * sps).
    """
    h = rrc_taps(sps, rolloff, span)
    return upfirdn(h, symbols, up=sps).astype(complex)


def set_launch_power(signal: np.ndarray, power_dBm: float = 0.0) -> np.ndarray:
    """
    Scale signal amplitude so that the average optical launch power equals `power_dBm`.

    The signal is converted from dimensionless normalized amplitude to physical
    optical field envelope A(t) in units of [√W].
        P_launch [W] = 10^((power_dBm - 30) / 10)
        mean(|A(t)|²) = P_launch [W]

    Parameters
    ----------
    signal    : np.ndarray, complex – input optical field
    power_dBm : float               – target average launch power [dBm]

    Returns
    -------
    signal_scaled : np.ndarray, complex – optical field [√W]
    """
    power_W = 1e-3 * (10.0 ** (power_dBm / 10.0))
    current_power = np.mean(np.abs(signal) ** 2)
    if current_power < 1e-30:
        return signal
    return signal * np.sqrt(power_W / current_power)
