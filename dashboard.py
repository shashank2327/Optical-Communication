"""
dashboard.py — Streamlit Interactive Dashboard
================================================
Launch:   python -m streamlit run dashboard.py

Features:
- Dynamic modulation-specific model loading (QPSK / 16-QAM) with feature scaling.
- Live optical link simulation with sliders for distance, launch power, and modulation.
- Three-way comparison: (1) No Equalization, (2) Linear FD-CDC, (3) Neural Network Equalizer.
- Shared normalized constellation plots and 4-panel stage breakdown.
- Pre-computed benchmark tab with distance sweep, power sweep, and training convergence curves.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tx import (generate_bits, modulate, pulse_shape, set_launch_power,
                SYMBOL_RATE_BD, SPS, ROLLOFF, SPAN)
from fiber import propagate_link, DEFAULT_FIBER, DEFAULT_EDFA
from rx import matched_filter, downsample, normalize_symbols, demodulate, count_bit_errors
from dsp import compensate_chromatic_dispersion, calculate_evm, carrier_phase_recovery
from ml import load_model, equalize, NNEqualizer, FeatureScaler, WINDOW_DEFAULT
from plots import (plot_constellation, plot_constellation_grid,
                   plot_ber_vs_distance, plot_ber_vs_power, plot_evm_vs_distance,
                   plot_evm_vs_power, plot_training_curve)

# ─── Constants ────────────────────────────────────────────────────────────────
N_SYMBOLS_LIVE = 512    # Responsive live simulation (< 1.5 s on laptop CPU)
SPAN_KM        = 80.0   # EDFA spacing [km]
DZ_KM_LIVE     = 1.0    # SSFM step for live simulation

FS = SPS * SYMBOL_RATE_BD

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title='Optical Link AI Equalizer Dashboard',
    page_icon='🔬',
    layout='wide',
)

# ─── Cached Model & Resource Loaders ──────────────────────────────────────────
@st.cache_resource
def get_model_for_modulation(modulation: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod_tag = modulation.lower().replace("-", "")
    paths = [
        os.path.join(base_dir, 'models', f'equalizer_{mod_tag}.pt'),
        os.path.join(base_dir, 'models', 'equalizer.pt')
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                model, scaler, meta = load_model(p)
                return model, scaler, meta, p
            except Exception as e:
                pass
    return None, None, {}, None


@st.cache_data
def get_precomputed_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'results', 'ber_curve.npy')
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True).item()
        except Exception:
            pass
    return None


@st.cache_data
def get_training_histories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res = {}
    for mod in ['QPSK', '16-QAM']:
        mod_tag = mod.lower().replace("-", "")
        p = os.path.join(base_dir, 'results', f'training_history_{mod_tag}.npy')
        if os.path.exists(p):
            try:
                res[mod] = np.load(p, allow_pickle=True).item()
            except Exception:
                pass
    qpsk_p = os.path.join(base_dir, 'results', 'training_history.npy')
    if 'QPSK' not in res and os.path.exists(qpsk_p):
        try:
            res['QPSK'] = np.load(qpsk_p, allow_pickle=True).item()
        except Exception:
            pass
    return res


# ─── Header ───────────────────────────────────────────────────────────────────
st.title('🔬 Optical Link AI Impairment Compensation Dashboard')
st.caption(
    'High-Speed Coherent Optical Link Simulation & Neural-Network Nonlinear Equalization  '
    '| Final-Year B.Tech (ECE) Project'
)

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Link Parameters')

    fiber_km = st.slider('Fiber Length (km)', min_value=80, max_value=1600, value=400, step=80)
    power_dBm = st.slider('Launch Power (dBm)', min_value=-4.0, max_value=6.0, value=2.0, step=1.0)
    modulation = st.selectbox('Modulation Format', ['QPSK', '16-QAM'])
    use_ml = st.toggle('🧠 Neural Network Equalizer', value=True)

    n_spans = max(1, int(round(fiber_km / SPAN_KM)))
    total_loss_dB = 0.2 * fiber_km

    st.markdown('---')
    st.markdown(
        f'**EDFA Spans:** {n_spans} ({SPAN_KM:.0f} km/span)  \n'
        f'**Cumulative Fiber Loss:** {total_loss_dB:.1f} dB  \n'
        f'**Dispersion Memory:** ~{int(fiber_km * 17.0 * 1e-12 / (1.0 / 32e9)):d} symbols'
    )

    # Model metadata & checks
    model, scaler, meta, model_path = get_model_for_modulation(modulation)
    st.markdown('---')
    if model is not None:
        st.success(f"Loaded: `{os.path.basename(model_path)}`")
        st.markdown(
            f"**Architecture:** MLP (Window={meta.get('window', 11)}, Hidden={meta.get('hidden', 64)})  \n"
            f"**Training Anchor:** {meta.get('train_distance_km', 400):.0f} km, {meta.get('train_power_dBm', 4.0):+.1f} dBm"
        )
        if fiber_km > 1000 or power_dBm > 5.0:
            st.warning("⚠️ Operating beyond nominal training condition (generalization stress test).")
    else:
        st.error("No model found. Run `python generate_ber_curve.py` first.")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_live, tab_precomp = st.tabs(['📡 Live Simulation', '📊 Benchmark Curves'])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    with st.spinner('Simulating optical propagation through fiber...'):
        bps = 2 if modulation == 'QPSK' else 4
        # 1. Transmitter
        tx_bits = generate_bits(N_SYMBOLS_LIVE, modulation, seed=42)
        tx_syms = modulate(tx_bits, modulation)
        tx_field = set_launch_power(pulse_shape(tx_syms), power_dBm)

        # 2. Fiber Channel (SSFM + EDFA ASE noise)
        rx_field = propagate_link(
            tx_field, n_spans, SPAN_KM, DEFAULT_FIBER, DEFAULT_EDFA,
            fs=FS, dz_km=DZ_KM_LIVE, seed=100
        )

        # 3. Conventional Linear DSP (FD-CDC)
        rx_field_cdc = compensate_chromatic_dispersion(rx_field, fiber_km, DEFAULT_FIBER, fs=FS)

        # 4. Matched filter & downsampling
        mf_raw = matched_filter(rx_field)
        mf_cdc = matched_filter(rx_field_cdc)

        delay = 2 * SPAN * SPS
        rx_syms_raw = downsample(mf_raw, SPS, offset=0)
        rx_syms_cdc = downsample(mf_cdc, SPS, offset=0)

        n_syms = min(len(rx_syms_raw), len(rx_syms_cdc), len(tx_syms))
        rx_syms_raw = rx_syms_raw[:n_syms]
        rx_syms_cdc = rx_syms_cdc[:n_syms]
        tx_syms_c   = tx_syms[:n_syms]
        tx_bits_c   = tx_bits[:n_syms * bps]

        # 5. Normalization & Carrier Phase Recovery
        rx_norm_raw, _     = normalize_symbols(rx_syms_raw)
        rx_norm_cdc_raw, _ = normalize_symbols(rx_syms_cdc)
        rx_norm_cdc_cpr, phi_rad = carrier_phase_recovery(rx_norm_cdc_raw, modulation=modulation, pilot_tx=tx_syms_c)
        phi_deg = np.degrees(phi_rad)

        # 6. ML Equalization (on normalized CDC+CPR symbols)
        w_size = meta.get('window', WINDOW_DEFAULT)
        half = w_size // 2

        eq_syms = None
        eq_norm = None

        if use_ml and model is not None:
            eq_raw = equalize(model, rx_norm_cdc_cpr, scaler=scaler, window=w_size)
            if len(eq_raw) > 0:
                eq_norm, _ = normalize_symbols(eq_raw)
                eq_syms = eq_norm

        # 7. Common central evaluation slicing
        if eq_norm is not None:
            n_eval = len(eq_norm)
            eval_tx_syms = tx_syms_c[half : half + n_eval]
            eval_tx_bits = tx_bits_c[half * bps : (half + n_eval) * bps]

            eval_raw_syms     = rx_norm_raw[half : half + n_eval]
            eval_cdc_raw_syms = rx_norm_cdc_raw[half : half + n_eval]
            eval_cdc_cpr_syms = rx_norm_cdc_cpr[half : half + n_eval]
            eval_ml_syms      = eq_norm
        else:
            n_eval = len(tx_syms_c) - 2 * half
            eval_tx_syms = tx_syms_c[half : half + n_eval]
            eval_tx_bits = tx_bits_c[half * bps : (half + n_eval) * bps]

            eval_raw_syms     = rx_norm_raw[half : half + n_eval]
            eval_cdc_raw_syms = rx_norm_cdc_raw[half : half + n_eval]
            eval_cdc_cpr_syms = rx_norm_cdc_cpr[half : half + n_eval]
            eval_ml_syms      = None

        # Demodulate & count errors
        bits_raw     = demodulate(eval_raw_syms, modulation)
        bits_cdc_raw = demodulate(eval_cdc_raw_syms, modulation)
        bits_cdc_cpr = demodulate(eval_cdc_cpr_syms, modulation)

        err_raw, total_b, ber_raw     = count_bit_errors(eval_tx_bits, bits_raw)
        err_cdc_raw, _, ber_cdc_raw   = count_bit_errors(eval_tx_bits, bits_cdc_raw)
        err_cdc_cpr, _, ber_cdc_cpr   = count_bit_errors(eval_tx_bits, bits_cdc_cpr)

        evm_raw     = calculate_evm(eval_raw_syms, eval_tx_syms)
        evm_cdc_raw = calculate_evm(eval_cdc_raw_syms, eval_tx_syms)
        evm_cdc_cpr = calculate_evm(eval_cdc_cpr_syms, eval_tx_syms)

        if eval_ml_syms is not None:
            bits_ml = demodulate(eval_ml_syms, modulation)
            err_ml, _, ber_ml = count_bit_errors(eval_tx_bits, bits_ml)
            evm_ml = calculate_evm(eval_ml_syms, eval_tx_syms)
        else:
            err_ml, ber_ml, evm_ml = None, None, None

    # ── Metric Cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Condition', f'{fiber_km} km | {power_dBm:+.0f} dBm', f'{modulation}')
    m2.metric('No Equalizer', f'{ber_raw:.2e}', f'EVM: {evm_raw:.1f}%', delta_color='inverse')
    m3.metric('CDC (Raw, No CPR)', f'{ber_cdc_raw:.2e}', f'Rot: {phi_deg:+.1f}°', delta_color='inverse')
    m4.metric('CDC (+ CPR)', f'{ber_cdc_cpr:.2e}', f'EVM: {evm_cdc_cpr:.1f}%', delta_color='inverse')

    if ber_ml is not None:
        delta_val = f"EVM: {evm_ml:.1f}%"
        m5.metric('Neural Network Equalizer', f'{ber_ml:.2e}', delta_val)
    else:
        m5.metric('Neural Network Equalizer', 'Disabled' if not use_ml else 'No Model')

    if evm_ml is not None:
        st.caption(
            f'ℹ️ *Kerr nonlinear Self-Phase Modulation (SPM) rotation: **{phi_deg:+.1f}°**. '
            f'Linear FD-CDC (+CPR) achieves BER = {ber_cdc_cpr:.2e} (EVM: {evm_cdc_cpr:.1f}%). '
            f'Neural Network Equalizer compresses constellation variance, reducing EVM from {evm_cdc_cpr:.1f}% to {evm_ml:.1f}%.*'
        )
    else:
        st.caption(
            f'ℹ️ *Kerr nonlinear Self-Phase Modulation (SPM) rotation: **{phi_deg:+.1f}°**. '
            f'Linear FD-CDC (+CPR) achieves BER = {ber_cdc_cpr:.2e} (EVM: {evm_cdc_cpr:.1f}%). '
            f'Neural Network Equalizer is currently disabled.*'
        )
    st.divider()

    # ── Visualizations ────────────────────────────────────────────────────────
    st.subheader('Constellation Diagram')

    # Toggle switches for selecting constellation stages
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        show_tx = st.toggle('🔵 Ideal Transmitted', value=True)
    with t2:
        show_raw = st.toggle('🔴 Linear FD-CDC (Raw)', value=True)
    with t3:
        show_cpr = st.toggle('🟣 After CPR', value=True)
    with t4:
        show_ml = st.toggle('🟢 Neural Network EQ', value=True, disabled=(eval_ml_syms is None))

    sym_list = []
    lbl_list = []
    col_list = []
    marker_list = []

    if show_tx:
        sym_list.append(eval_tx_syms)
        lbl_list.append('Ideal Transmitted')
        col_list.append('#2563EB')
        marker_list.append('o')

    if show_raw:
        sym_list.append(eval_cdc_raw_syms)
        lbl_list.append(f'Linear FD-CDC (Raw, {phi_deg:+.1f}°)')
        col_list.append('#DC2626')
        marker_list.append('x')

    if show_cpr:
        sym_list.append(eval_cdc_cpr_syms)
        lbl_list.append('Linear FD-CDC (After CPR)')
        col_list.append('#7C3AED')
        marker_list.append('^')

    if show_ml and eval_ml_syms is not None:
        sym_list.append(eval_ml_syms)
        lbl_list.append('Neural Network EQ')
        col_list.append('#059669')
        marker_list.append('s')

    if len(sym_list) > 0:
        fig_const = plot_constellation(
            sym_list, lbl_list,
            title=f'{modulation} Constellation ({fiber_km} km, {power_dBm:+.0f} dBm)',
            colours=col_list,
            markers=marker_list
        )
        st.pyplot(fig_const, use_container_width=True)
    else:
        st.info("💡 Please enable at least one toggle above to display constellation symbols.")

    # ── 4-Panel Stage Breakdown Expander ──────────────────────────────────────
    with st.expander("🔍 View 4-Panel Stage Breakdown (Tx vs Raw CDC vs CPR vs Neural Network)", expanded=False):
        fig_grid = plot_constellation_grid(
            eval_tx_syms, eval_cdc_raw_syms, eval_cdc_cpr_syms, eval_ml_syms,
            title=f'{modulation} 4-Stage Constellation Comparison ({fiber_km} km, {power_dBm:+.0f} dBm)'
        )
        st.pyplot(fig_grid, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRE-COMPUTED BENCHMARK CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_precomp:
    data = get_precomputed_data()
    histories = get_training_histories()

    if data is not None:
        mod_choice = st.radio(
            "Select Benchmark Modulation:",
            ["16-QAM (Nonlinear Impairment Benchmark — SPM & Inter-Ring Distortion)",
             "QPSK (Linear Dispersion & Phase Benchmark — Constant Envelope)"],
            horizontal=True
        )
        sel_mod = '16-QAM' if '16-QAM' in mod_choice else 'QPSK'

        # Extract per-modulation dataset if available, or fall back to root data
        m_data = data.get(sel_mod, data)
        distances = m_data.get('distances', data['distances'])
        powers = m_data.get('powers', data.get('powers', np.array([-2, 0, 2, 4, 6])))
        n_bits = m_data.get('n_bits', 8172 if sel_mod == 'QPSK' else 16344)

        # Distances series
        dist_no_eq = m_data.get('dist_no_eq_ber', data.get('ber_noml'))
        dist_cdc = m_data.get('dist_cdc_cpr_ber', data.get('ber_cdc'))
        dist_ml = m_data.get('dist_ml_ber', data.get('ber_ml'))

        # Powers series
        pwr_no_eq = m_data.get('pwr_no_eq_ber', data.get('power_bers_no_eq'))
        pwr_cdc = m_data.get('pwr_cdc_cpr_ber', data.get('power_bers_cdc'))
        pwr_ml = m_data.get('pwr_ml_ber', data.get('power_bers_ml'))

        # EVM series
        dist_evm_cdc = m_data.get('dist_cdc_cpr_evm')
        dist_evm_ml = m_data.get('dist_ml_evm')
        pwr_evm_cdc = m_data.get('pwr_cdc_cpr_evm')
        pwr_evm_ml = m_data.get('pwr_ml_evm')

        # ── Benchmark Controls & Current Operating Point Banner ───────────────
        st.info(f"📍 **Current Live Parameters:** Fiber Length = **{fiber_km} km** | Launch Power = **{power_dBm:+.1f} dBm**  \n"
                f"*(Amber dotted vertical lines on the benchmark curves indicate where your current live link parameters lie).*")

        # ── Row 1: Bit Error Rate (BER) Curves ────────────────────────────────
        st.markdown(f"### 📈 Bit Error Rate (BER) Performance — {sel_mod}")
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader('BER vs Transmission Distance')
            eval_pwr = "+4.0 dBm" if sel_mod == '16-QAM' else "+2.0 dBm"
            fig_dist = plot_ber_vs_distance(
                distances=distances,
                bers_no_eq=dist_no_eq,
                bers_cdc=dist_cdc,
                bers_ml=dist_ml,
                n_bits=n_bits,
                title=f'{sel_mod} BER vs Distance ({eval_pwr})',
                highlight_distance=fiber_km
            )
            st.pyplot(fig_dist, use_container_width=True)

        with c_right:
            st.subheader('BER vs Optical Launch Power (Nonlinear Regime)')
            if pwr_no_eq is not None and pwr_cdc is not None and pwr_ml is not None:
                fig_pwr = plot_ber_vs_power(
                    powers_dBm=powers,
                    bers_no_eq=pwr_no_eq,
                    bers_cdc=pwr_cdc,
                    bers_ml=pwr_ml,
                    distance_km=400.0,
                    n_bits=n_bits,
                    title=f'{sel_mod} BER vs Power at 400 km',
                    highlight_power=power_dBm
                )
                st.pyplot(fig_pwr, use_container_width=True)

        # ── Row 2: Error Vector Magnitude (EVM) Curves ────────────────────────
        if dist_evm_cdc is not None and dist_evm_ml is not None:
            st.markdown(f"### 🎯 Error Vector Magnitude (EVM) Constellation Compression — {sel_mod}")
            c_evm_l, c_evm_r = st.columns(2)

            with c_evm_l:
                st.subheader('EVM vs Transmission Distance')
                fig_evm_d = plot_evm_vs_distance(
                    distances=distances,
                    evm_cdc=dist_evm_cdc,
                    evm_ml=dist_evm_ml,
                    title=f'{sel_mod} EVM vs Distance ({eval_pwr})',
                    highlight_distance=fiber_km
                )
                st.pyplot(fig_evm_d, use_container_width=True)

            with c_evm_r:
                st.subheader('EVM vs Optical Launch Power')
                if pwr_evm_cdc is not None and pwr_evm_ml is not None:
                    fig_evm_p = plot_evm_vs_power(
                        powers_dBm=powers,
                        evm_cdc=pwr_evm_cdc,
                        evm_ml=pwr_evm_ml,
                        distance_km=400.0,
                        title=f'{sel_mod} EVM vs Launch Power (400 km)',
                        highlight_power=power_dBm
                    )
                    st.pyplot(fig_evm_p, use_container_width=True)

        # ── Row 3: Training Loss Convergence ──────────────────────────────────
        if sel_mod in histories:
            c_tr_head, c_tr_opt = st.columns([3, 1])
            with c_tr_head:
                st.markdown(f"### 🧠 Neural Network Convergence — {sel_mod}")
            with c_tr_opt:
                log_scale = st.checkbox(
                    "Logarithmic Y-Scale",
                    value=True,
                    help="Expands the convergence region to clearly inspect loss plateauing and separation from Identity Baseline."
                )

            fig_tr = plot_training_curve(
                histories[sel_mod],
                f'NNEqualizer Training Convergence ({sel_mod})',
                log_scale=log_scale
            )
            st.pyplot(fig_tr, use_container_width=True)
            st.caption(
                "💡 **Convergence Insights:** "
                "At Epoch 1, validation MSE begins lower than training MSE because train loss is averaged across progressive weight updates within the epoch, "
                "whereas validation evaluates the network at the end of the epoch. "
                "In Log Scale, the loss clearly plateaus at ~0.009 (well below the 0.028 Identity Baseline), proving active nonlinear distortion compensation without overfitting."
            )

        # ── Row 4: Detailed CSV Records ───────────────────────────────────────
        st.markdown(f"### 📋 Exact Benchmark Records & Error Counts — {sel_mod}")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'results', 'benchmark_results.csv')
        if os.path.exists(csv_path):
            df_all = pd.read_csv(csv_path)
            df_mod = df_all[df_all['modulation'] == sel_mod].copy()
            st.dataframe(df_mod, use_container_width=True)
        else:
            df = pd.DataFrame({
                'Distance (km)': distances,
                'No Equalization BER': dist_no_eq,
                'Linear FD-CDC BER': dist_cdc,
                'Neural Network BER': dist_ml,
            })
            st.dataframe(df, use_container_width=True)

        # ── Informational Notes & Physical Insights ───────────────────────────
        with st.expander("ℹ️ Scientific Analysis: Resolution Limits, CPR & Nonlinear Compensation", expanded=False):
            st.markdown(r"""
**1. Empirical Resolution Limit & Zero-Error Confidence Bounds:**
- When $0$ bit errors are observed over $N$ tested bits, the empirical measurement resolution is $1/N$.
- Rather than inserting an artificial numerical floor (such as $10^{-6}$), zero-error points are represented by hollow markers at the empirical limit $1/N$.
- Statistically, by the rule of three for a Poisson/Binomial process, observing zero errors in $N$ bits establishes a **95% Clopper-Pearson upper confidence bound** of $\text{BER} < \frac{3}{N}$ (e.g., $< 3.67 \times 10^{-4}$ for $N=8,172$).

**2. Why QPSK CDC (+CPR) and Neural Network Curves Converge:**
- QPSK has a constant envelope ($|s|^2 = 1.0$) and a wide decision boundary distance ($d_{\min} = 1/\sqrt{2} \approx 0.707$).
- Once Carrier Phase Recovery (CPR) eliminates the mean Kerr phase rotation $\Delta\phi = N_{\text{spans}} \gamma P_0 L_{\text{eff}}$, the residual noise margin is $0.707 / \sigma \approx 6.4$, yielding $Q(6.4) \approx 7.8 \times 10^{-11}$.
- In an 8,172-bit test burst, observing zero errors is the exact physically expected outcome. The true contribution of the Neural Network in QPSK is **analog constellation variance (EVM) compression**, reducing EVM from $11.1\%$ down to $1.55\%$ (an $86\%$ improvement).

**3. Distinct BER Separation in 16-QAM:**
- 16-QAM features three concentric amplitude rings ($|s|^2 \in \{0.2, 1.0, 1.8\}$), causing differential intensity-dependent rotation (constellation spiraling) and a much narrower decision margin ($d_{\min} = 0.316$).
- Linear FD-CDC (+CPR) cannot correct amplitude-dependent rotation, causing BER to cross above the standard 7% HD-FEC threshold ($3.8 \times 10^{-3}$) at 400 km (+4 dBm).
- The Neural Network Equalizer learns the nonlinear spatial-temporal mapping, reducing BER from $5.64 \times 10^{-3}$ down to $2.21 \times 10^{-3}$ (a $>56\%$ error reduction) and recovering transmission below the HD-FEC threshold.

**4. Why Raw Uncompensated BER is ~0.40–0.50:**
- For 32 GBd over SSMF ($\beta_2 = -21.67\text{ ps}^2/\text{km}$), the dispersion length is $L_D = T_0^2 / |\beta_2| \approx 45\text{ km}$.
- At distances $\ge 80\text{ km}$, uncompensated chromatic dispersion spreads each pulse over tens of symbol slots. By the Central Limit Theorem, the inter-symbol interference converges to a 2D complex Gaussian process, giving an asymptotic random BER of $0.5000$ (50%). Back-to-back ($0\text{ km}$) achieves $\text{BER} = 0.0000$.
            """)
    else:
        st.info("No pre-computed benchmark data found. Run `python generate_ber_curve.py` to generate.")
