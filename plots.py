"""
plots.py — Visualization Module
================================
Generates all figures for the optical communication link project:
- plot_eye_diagram: Overlaid signal traces (pre-equalizer or reconstructed post-equalizer)
- plot_constellation: IQ scatter plots on shared normalized scale
- plot_ber_vs_distance: Log-scale BER curves comparing No-EQ, Conventional DSP (CDC), and ML
- plot_ber_vs_power: Optical launch power sweep curves showing Kerr nonlinear penalty
- plot_training_curve: MSE loss vs epoch with baseline zero/identity loss benchmarks
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

# ─── Accessible Colour Palette ────────────────────────────────────────────────
_C_BLUE   = '#2563EB'   # Ideal Transmitted / Train loss
_C_RED    = '#DC2626'   # Uncompensated (No EQ)
_C_PURPLE = '#7C3AED'   # Conventional DSP (Linear FD-CDC)
_C_GREEN  = '#059669'   # Neural Network Equalizer (ML)
_C_ORANGE = '#D97706'   # Validation loss
_C_GREY   = '#6B7280'   # Grid / Baselines / Thresholds


# ─── Eye Diagram ──────────────────────────────────────────────────────────────

def plot_eye_diagram(signal: np.ndarray,
                     sps: int,
                     title: str = 'Eye Diagram',
                     n_traces: int = 200,
                     stage_label: str = 'Pre-Equalizer') -> plt.Figure:
    """
    Overlay multiple 2-symbol-period segments to form an eye diagram.

    Parameters
    ----------
    signal : np.ndarray, complex
        Signal sampled at `sps` samples/symbol.
    sps : int
        Samples per symbol.
    title : str
        Figure title.
    n_traces : int
        Number of overlaid traces.
    stage_label : str
        Explicit label of the processing stage (e.g. 'Pre-Equalizer' or 'Post-ML Reconstructed').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    seg_len = 2 * sps
    max_traces = min(n_traces, (len(signal) - sps) // sps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    for ax, component, label in zip(axes,
                                    [signal.real, signal.imag],
                                    ['In-Phase (I)', 'Quadrature (Q)']):
        for k in range(max_traces):
            seg = component[k * sps : k * sps + seg_len]
            if len(seg) < seg_len:
                break
            ax.plot(seg, color=_C_BLUE, alpha=0.15, linewidth=0.7)

        ax.set_title(f'{label} — {stage_label}', fontsize=11, fontweight='semibold')
        ax.set_xlabel('Sample within 2T window', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.35)

        # Vertical markers at the ideal Nyquist sampling instants
        for ts in [sps // 2, sps + sps // 2]:
            ax.axvline(ts, color=_C_GREY, linewidth=0.8, linestyle=':')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ─── Constellation Diagram ────────────────────────────────────────────────────

def plot_constellation(symbols_list: list[np.ndarray],
                       labels: list[str],
                       title: str = 'Constellation Diagram',
                       max_points: int = 1200,
                       colours: list[str] | None = None,
                       markers: list[str] | None = None) -> plt.Figure:
    """
    Scatter plot of symbol clouds on a shared, normalized IQ plane.

    Parameters
    ----------
    symbols_list : list of complex np.ndarray
        Symbols normalized to unit average power.
    labels : list of str
        Legend labels corresponding to symbols_list.
    title : str
    max_points : int
    colours : list of str, optional
        Custom color hex strings for each trace.
    markers : list of str, optional
        Custom marker symbols for each trace.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    default_colours = [_C_BLUE, _C_RED, _C_PURPLE, _C_ORANGE, _C_GREEN]
    default_markers = ['o', 'x', '^', 'D', 's']
    sizes   = [10,  6,   8,   7,   7]

    use_cols = colours if colours is not None else default_colours
    use_marks = markers if markers is not None else default_markers

    fig, ax = plt.subplots(figsize=(6, 6))

    for syms, lbl, col, mk, sz in zip(symbols_list, labels, use_cols, use_marks, sizes):
        n = min(len(syms), max_points)
        ax.scatter(syms[:n].real, syms[:n].imag,
                   s=sz, alpha=0.50, color=col, marker=mk, label=lbl)

    ax.axhline(0, color=_C_GREY, linewidth=0.8, linestyle='--')
    ax.axvline(0, color=_C_GREY, linewidth=0.8, linestyle='--')
    ax.set_xlabel('In-Phase (I)  [Normalized]', fontsize=11)
    ax.set_ylabel('Quadrature (Q)  [Normalized]', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(markerscale=2.5, fontsize=9, loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    return fig


def plot_constellation_grid(tx_syms: np.ndarray,
                            cdc_raw_syms: np.ndarray,
                            cdc_cpr_syms: np.ndarray,
                            ml_syms: np.ndarray,
                            title: str = 'Optical Constellation Diagram Comparison (400 km, +4 dBm)',
                            max_points: int = 1000) -> plt.Figure:
    """
    2x2 Subplot Grid comparing:
    - (a) Ideal Transmitted Symbols
    - (b) Linear FD-CDC Before Phase Correction (showing physical Kerr rotation)
    - (c) Linear FD-CDC After Carrier Phase Recovery (de-rotated onto quadrants)
    - (d) Neural Network Equalizer (nonlinear compression & EVM reduction)
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    panels = [
        (axes[0, 0], tx_syms, 'Ideal Transmitted', _C_BLUE, 'o'),
        (axes[0, 1], cdc_raw_syms, 'Linear FD-CDC (Before CPR)', _C_RED, 'x'),
        (axes[1, 0], cdc_cpr_syms, 'Linear FD-CDC (After CPR)', _C_PURPLE, '^'),
        (axes[1, 1], ml_syms, 'Neural Network Equalizer', _C_GREEN, 's'),
    ]
    for ax, syms, label, col, mk in panels:
        if syms is None:
            ax.text(0, 0, 'Equalizer Disabled / Unavailable',
                    ha='center', va='center', fontsize=11, color=_C_GREY, fontstyle='italic')
            ax.set_title(label, fontsize=11, fontweight='semibold')
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.3)
            continue
        n = min(len(syms), max_points)
        ax.scatter(syms[:n].real, syms[:n].imag, s=12, alpha=0.55, color=col, marker=mk)
        ax.axhline(0, color=_C_GREY, linewidth=0.8, linestyle='--')
        ax.axvline(0, color=_C_GREY, linewidth=0.8, linestyle='--')
        ax.set_title(label, fontsize=11, fontweight='semibold')
        ax.set_xlabel('In-Phase (I)', fontsize=9)
        ax.set_ylabel('Quadrature (Q)', fontsize=9)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout()
    return fig


# ─── BER vs Distance ──────────────────────────────────────────────────────────

def plot_ber_vs_distance(distances: np.ndarray,
                         bers_no_eq: np.ndarray,
                         bers_cdc: np.ndarray | None = None,
                         bers_ml: np.ndarray | None = None,
                         fec_limit: float = 3.8e-3,
                         n_bits: int = 8172,
                         title: str = 'Bit Error Rate vs Transmission Distance',
                         highlight_distance: float | None = None) -> plt.Figure:
    """
    Log-scale BER vs transmission distance comparing:
    - Baseline 1: No Equalization (Raw)
    - Baseline 2: Conventional Linear DSP (FD-CDC)
    - Proposed: Neural Network Equalizer (ML)

    Parameters
    ----------
    distances : np.ndarray
        Distances in [km].
    bers_no_eq : np.ndarray
        Raw uncompensated BER.
    bers_cdc : np.ndarray or None
        Conventional Linear FD-CDC BER.
    bers_ml : np.ndarray or None
        Neural network equalized BER.
    fec_limit : float
        Standard 7% hard-decision FEC threshold (3.8 × 10⁻³).
    n_bits : int
        Number of bits compared (determines statistical resolution limit 1/N).
    title : str
        Figure title.
    highlight_distance : float or None
        Current operating distance [km] to highlight with an indicator line.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    res_limit = 1.0 / max(n_bits, 1000)
    y_min = res_limit * 0.4

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    def _plot_series(x_vals, y_vals, col, mk, ls, label):
        y_arr = np.array(y_vals, dtype=float)
        has_err = y_arr > 0
        zero_err = y_arr == 0

        # Plot line connecting measured non-zero BERs
        if np.any(has_err):
            ax.semilogy(x_vals[has_err], y_arr[has_err],
                        linestyle=ls, color=col, linewidth=1.8, marker=mk, markersize=6,
                        label=label)
        # Plot zero-error points with open/hollow markers at the resolution limit
        if np.any(zero_err):
            ax.scatter(x_vals[zero_err], [res_limit] * np.sum(zero_err),
                       s=45, facecolors='none', edgecolors=col, linewidth=1.5,
                       label=f'{label} (0 errors: < {3.0/n_bits:.1e})' if not np.any(has_err) else None)

    _plot_series(distances, bers_no_eq, _C_RED, 'o', '--', 'No Equalization (Raw)')

    if bers_cdc is not None:
        _plot_series(distances, bers_cdc, _C_PURPLE, '^', '-.', 'Linear FD-CDC (+ CPR)')

    if bers_ml is not None:
        _plot_series(distances, bers_ml, _C_GREEN, 's', '-', 'Neural Network Equalizer (ML)')

    # Resolution limit line (1/N)
    ax.axhline(res_limit, color=_C_GREY, linestyle='--', linewidth=1.0, alpha=0.7,
               label=f'Resolution Limit: 1/N ({res_limit:.1e})')

    # Standard 7% Overhead HD-FEC limit threshold
    ax.axhline(fec_limit, color='crimson', linestyle=':', linewidth=1.3,
               label=f'7% HD-FEC Limit ({fec_limit:.1e})')

    # Current operating point visual indicator
    if highlight_distance is not None:
        ax.axvline(highlight_distance, color='#F59E0B', linestyle=':', linewidth=2.2, alpha=0.9,
                   label=f'Current Link ({highlight_distance:.0f} km)')

    ax.set_xlabel('Transmission Distance  [km]', fontsize=11)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=y_min, top=1.2)
    fig.tight_layout()
    return fig


# ─── BER vs Launch Power ──────────────────────────────────────────────────────

def plot_ber_vs_power(powers_dBm: np.ndarray,
                      bers_no_eq: np.ndarray,
                      bers_cdc: np.ndarray,
                      bers_ml: np.ndarray,
                      distance_km: float = 400.0,
                      fec_limit: float = 3.8e-3,
                      n_bits: int = 8172,
                      title: str | None = None,
                      highlight_power: float | None = None) -> plt.Figure:
    """
    Plot BER vs Optical Launch Power showing Kerr nonlinear penalty at high powers.
    Zero-error points are represented honestly with hollow markers at the resolution limit.
    """
    res_limit = 1.0 / max(n_bits, 1000)
    y_min = res_limit * 0.4

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    def _plot_series(x_vals, y_vals, col, mk, ls, label):
        y_arr = np.array(y_vals, dtype=float)
        has_err = y_arr > 0
        zero_err = y_arr == 0
        if np.any(has_err):
            ax.semilogy(x_vals[has_err], y_arr[has_err],
                        linestyle=ls, color=col, linewidth=1.8, marker=mk, markersize=6,
                        label=label)
        if np.any(zero_err):
            ax.scatter(x_vals[zero_err], [res_limit] * np.sum(zero_err),
                       s=45, facecolors='none', edgecolors=col, linewidth=1.5,
                       label=f'{label} (0 errors: < {3.0/n_bits:.1e})' if not np.any(has_err) else None)

    _plot_series(powers_dBm, bers_no_eq, _C_RED, 'o', '--', 'No Equalization (Raw)')
    _plot_series(powers_dBm, bers_cdc, _C_PURPLE, '^', '-.', 'Linear FD-CDC (+ CPR)')
    _plot_series(powers_dBm, bers_ml, _C_GREEN, 's', '-', 'Neural Network Equalizer (ML)')

    ax.axhline(res_limit, color=_C_GREY, linestyle='--', linewidth=1.0, alpha=0.7,
               label=f'Resolution Limit: 1/N ({res_limit:.1e})')
    ax.axhline(fec_limit, color='crimson', linestyle=':', linewidth=1.3,
               label=f'7% HD-FEC Limit ({fec_limit:.1e})')

    # Current operating point visual indicator
    if highlight_power is not None:
        ax.axvline(highlight_power, color='#F59E0B', linestyle=':', linewidth=2.2, alpha=0.9,
                   label=f'Current Power ({highlight_power:+.1f} dBm)')

    ax.set_xlabel('Optical Launch Power  [dBm]', fontsize=11)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=11)
    if title is None:
        title = f'BER vs Launch Power at {distance_km:.0f} km (Nonlinear Regime)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    ax.set_ylim(bottom=y_min, top=1.2)
    fig.tight_layout()
    return fig


# ─── EVM Plots ────────────────────────────────────────────────────────────────

def plot_evm_vs_distance(distances: np.ndarray,
                         evm_cdc: np.ndarray,
                         evm_ml: np.ndarray,
                         evm_raw: np.ndarray | None = None,
                         title: str = 'Constellation EVM vs Transmission Distance',
                         highlight_distance: float | None = None) -> plt.Figure:
    """
    Plot Error Vector Magnitude (EVM) vs distance, showing continuous analog signal quality.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if evm_raw is not None:
        ax.plot(distances, evm_raw, 'o--', color=_C_RED, linewidth=1.5, markersize=5, label='No Equalization (Raw)')
    ax.plot(distances, evm_cdc, '^-.', color=_C_PURPLE, linewidth=1.8, markersize=6, label='Linear FD-CDC (+ CPR)')
    ax.plot(distances, evm_ml, 's-', color=_C_GREEN, linewidth=2.2, markersize=6, label='Neural Network Equalizer (ML)')

    if highlight_distance is not None:
        ax.axvline(highlight_distance, color='#F59E0B', linestyle=':', linewidth=2.2, alpha=0.9,
                   label=f'Current Link ({highlight_distance:.0f} km)')

    ax.set_xlabel('Transmission Distance  [km]', fontsize=11)
    ax.set_ylabel('Error Vector Magnitude (EVM)  [%]', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    return fig


def plot_evm_vs_power(powers_dBm: np.ndarray,
                      evm_cdc: np.ndarray,
                      evm_ml: np.ndarray,
                      distance_km: float = 400.0,
                      title: str | None = None,
                      highlight_power: float | None = None) -> plt.Figure:
    """
    Plot EVM vs Optical Launch Power showing Kerr nonlinear variance reduction.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(powers_dBm, evm_cdc, '^-.', color=_C_PURPLE, linewidth=1.8, markersize=6, label='Linear FD-CDC (+ CPR)')
    ax.plot(powers_dBm, evm_ml, 's-', color=_C_GREEN, linewidth=2.2, markersize=6, label='Neural Network Equalizer (ML)')

    if highlight_power is not None:
        ax.axvline(highlight_power, color='#F59E0B', linestyle=':', linewidth=2.2, alpha=0.9,
                   label=f'Current Power ({highlight_power:+.1f} dBm)')

    ax.set_xlabel('Optical Launch Power  [dBm]', fontsize=11)
    ax.set_ylabel('Error Vector Magnitude (EVM)  [%]', fontsize=11)
    if title is None:
        title = f'Constellation EVM vs Launch Power at {distance_km:.0f} km'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    return fig


# ─── Training Convergence Curve ───────────────────────────────────────────────

def plot_training_curve(history: dict,
                        title: str = 'Neural Network Training Curve',
                        log_scale: bool = True) -> plt.Figure:
    """
    Plot training and validation MSE loss vs epoch along with baseline loss benchmarks.

    Parameters
    ----------
    history : dict
        Contains 'train_loss', 'val_loss', 'baseline_zero', 'baseline_identity'.
    title : str
        Figure title.
    log_scale : bool
        If True, use logarithmic Y-scale (semilogy) to expand the convergence region
        and prevent high baseline values from squishing the loss curves.
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    if log_scale:
        ax.semilogy(epochs, history['train_loss'],
                    color=_C_BLUE, linewidth=2.0, label='Train MSE')
        ax.semilogy(epochs, history['val_loss'],
                    color=_C_ORANGE, linewidth=2.0, linestyle='--', label='Validation MSE')
    else:
        ax.plot(epochs, history['train_loss'],
                color=_C_BLUE, linewidth=2.0, label='Train MSE')
        ax.plot(epochs, history['val_loss'],
                color=_C_ORANGE, linewidth=2.0, linestyle='--', label='Validation MSE')

    b_zero = history.get('baseline_zero')
    b_ident = history.get('baseline_identity')

    if log_scale:
        # In log scale, both baselines display cleanly without dominating the dynamic range
        if b_zero is not None:
            ax.axhline(b_zero, color=_C_GREY, linestyle=':',
                       linewidth=1.2, label=f"Zero Predictor Baseline ({b_zero:.3f})")
        if b_ident is not None:
            ax.axhline(b_ident, color='#9333EA', linestyle='-.',
                       linewidth=1.1, label=f"Identity Baseline ({b_ident:.3f})")
        ax.set_ylabel('Mean Squared Error (MSE)  [Log Scale]', fontsize=11)
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
    else:
        # Linear scale: zoom in to max loss region so 0.497 doesn't flatten 90% of the graph
        max_loss = max(max(history['train_loss']), max(history['val_loss']))
        y_top = min(max_loss * 1.35, 0.08)

        if b_ident is not None:
            ax.axhline(b_ident, color='#9333EA', linestyle='-.',
                       linewidth=1.1, label=f"Identity Baseline ({b_ident:.3f})")

        if b_zero is not None:
            if b_zero <= y_top:
                ax.axhline(b_zero, color=_C_GREY, linestyle=':',
                           linewidth=1.2, label=f"Zero Predictor Baseline ({b_zero:.3f})")
            else:
                # Add informative text annotation instead of stretching Y-axis
                ax.text(0.98, 0.94, f"Zero Predictor Baseline: {b_zero:.3f} (off-scale)",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                        color=_C_GREY, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=_C_GREY))

        ax.set_ylim(bottom=0.0, top=y_top)
        ax.set_ylabel('Mean Squared Error (MSE)  [Zoomed Linear]', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.35)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='center right' if log_scale else 'upper right')
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save figure to disk and close."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure saved -> {path}')
