# AI-Based Impairment Compensation for High-Speed Optical Communication Links

> **B.Tech (ECE) Final-Year Project** | Optical Communication + AI/ML (Photonics & DSP)  
> Repository: `TERRA2k5/Optical-Communication-Links`

---

## 1. Overview & Project Objectives

In 32 GBaud coherent optical communications, transmission reach is fundamentally limited by:
1. **Chromatic Dispersion (CD):** Broadens optical pulses across symbol periods, creating inter-symbol interference (ISI).
2. **Kerr Nonlinearities (Self-Phase Modulation - SPM):** Fiber refractive index variations ($n = n_0 + n_2 I$) introduce intensity-dependent nonlinear phase distortion.
3. **Amplified Spontaneous Emission (ASE) Noise:** Injected by periodic Erbium-Doped Fiber Amplifiers (EDFAs).

While linear frequency-domain chromatic dispersion compensation (FD-CDC) handles dispersion, it cannot invert Kerr nonlinear distortion. This project implements a **physically consistent optical simulation** and trains a **lightweight Neural Network Equalizer (MLP)** that compensates both residual dispersion and nonlinear phase noise, achieving a demonstrable BER reduction and reach extension compared to conventional linear DSP.

---

## 2. System Architecture & Multi-Stage Receiver Comparison

```
Transmitter (tx.py)
   Bits -> Gray Modulation (QPSK / 16-QAM) -> RRC Pulse Shaping (8 SPS) -> Launch Power Scaling [√W]
      ↓
Optical Channel (fiber.py)
   [SSFM Fiber Span (80 km) -> In-line EDFA + ASE Noise] × N spans
      ↓
Coherent Receiver Front-End (rx.py & dsp.py)
   Rx Matched Filter -> Delay Alignment -> RMS Normalization -> Carrier Phase Recovery (CPR)
      ↓
Four Evaluation Stages:
   ├── (1) No Equalizer (Raw uncompensated dispersive signal)
   ├── (2) Linear FD-CDC (Raw, No CPR — Subject to ~45° Kerr SPM rotation at high power)
   ├── (3) Linear FD-CDC (+ CPR — De-rotated onto diagonal decision quadrants)
   └── (4) Neural Network Equalizer (Nonlinear impairment compensator on CPR symbols, Window W=11)
      ↓
Hard-Decision Demapping -> Bit Error Rate (BER), Symbol Error Rate (SER), EVM [%]
```

---

## 3. Physical Parameters

| Parameter | Symbol | Nominal Value | Module / Control |
|:---|:---:|:---:|:---|
| Carrier Wavelength | $\lambda$ | $1550\text{ nm}$ | `fiber.py` |
| Symbol Rate | $R_s$ | $32\text{ GBaud}$ ($31.25\text{ ps}$ symbol period) | `tx.py` |
| Samples per Symbol | $\text{SPS}$ | $8$ ($256\text{ GSa/s}$) | `tx.py` |
| RRC Roll-Off Factor | $\beta$ | $0.1$ | `tx.py` |
| Modulation Formats | — | QPSK (2 bps) & 16-QAM (4 bps) | `tx.py` / Dashboard |
| Fiber Attenuation | $\alpha$ | $0.2\text{ dB/km}$ | `fiber.py` |
| Chromatic Dispersion | $D$ | $17\text{ ps/(nm}\cdot\text{km)}$ ($\beta_2 \approx -21.7\text{ ps}^2/\text{km}$) | `fiber.py` |
| Kerr Nonlinear Coeff. | $\gamma$ | $1.3\text{ W}^{-1}\text{km}^{-1}$ | `fiber.py` |
| EDFA Span Length | $L_{\text{span}}$ | $80\text{ km}$ ($16\text{ dB}$ loss restored per span) | `fiber.py` |
| EDFA Noise Figure | $\text{NF}$ | $5.0\text{ dB}$ | `fiber.py` |
| Launch Power | $P_0$ | $-4\text{ to }+6\text{ dBm}$ (Nominal $+2\text{ to }+4\text{ dBm}$) | `generate_ber_curve.py` / Dashboard |
| Equalizer Architecture | — | MLP: $\text{Input}(22) \to \text{FC}(64) \to \text{ReLU} \to \text{FC}(64) \to \text{ReLU} \to \text{FC}(2)$ | `ml.py` |

---

## 4. Quick Start: How to Run the Project

Ensure you have activated your Python environment:

```powershell
cd "d:\New folder\Optical-Communication-Links"
.\venv\Scripts\Activate.ps1
```

### Step 1 — Run the Acceptance Test Suite
Verifies physical units, Gray mapping roundtrips, back-to-back zero BER, and model persistence:

```powershell
python test_project.py
```
*(Optionally run `python test_stage1.py` and `python test_stage2.py` for granular Acceptance Gate checks).*

### Step 2 — Train Models & Generate Pre-Computed Benchmark Curves
Trains separate QPSK and 16-QAM models, conducts multi-burst evaluations across distances ($80$ to $1600\text{ km}$) and powers ($-2$ to $+6\text{ dBm}$), and logs results to `results/benchmark_results.csv`:

```powershell
python generate_ber_curve.py
```

### Step 3 — Launch the Interactive Streamlit Dashboard
Start the real-time interactive user interface:

```powershell
python -m streamlit run dashboard.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## 5. Verified Experimental Results

The benchmark evaluates all processing paths on identical transmitted bit sequences and optical noise realizations across the exact same common central symbol slice ($W=11$ edge-loss accounted for on all branches):

### 5.1 Reach Extension (Distance Sweep at $+2.0\text{ dBm}$ for QPSK)

| Distance (km) | No-EQ BER | CDC (Raw, No CPR) BER | CDC (+ CPR) BER | Neural Network Equalizer BER | CDC EVM [%] | NN EVM [%] |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **80** | .3993$ | .0000$ | .0000$ | **.0000$** | .71\%$ | **.13\%$** |
| **240** | .4574$ | .0000$ | .0000$ | **.0000$** | .65\%$ | **.14\%$** |
| **400** | .4781$ | .0000$ | .0000$ | **.0000$** | .13\%$ | **.39\%$** |
| **480** | .4831$ | .12 \times 10^{-4}$ | .0000$ | **.0000$** | .63\%$ | **.53\%$** |
| **560** | .4810$ | .30 \times 10^{-2}$ | .89 \times 10^{-4}$ | **.67 \times 10^{-4}$** | .75\%$ | **.19\%$** |
| **640** | .4952$ | .87 \times 10^{-2}$ | .89 \times 10^{-4}$ | **.67 \times 10^{-4}$** | .20\%$ | **.16\%$** |
| **800** | .5012$ | .57 \times 10^{-1}$ | .57 \times 10^{-4}$ | **.79 \times 10^{-4}$** | .81\%$ | **.41\%$** |
| **960** | .5146$ | .85 \times 10^{-1}$ | .34 \times 10^{-4}$ | **.34 \times 10^{-4}$** | .24\%$ | **.93\%$** |
| **1120** | .5050$ | .97 \times 10^{-1}$ | .34 \times 10^{-4}$ | **.34 \times 10^{-4}$** | .81\%$ | **.70\%$** |
| **1280** | .5180$ | .98 \times 10^{-1}$ | .47 \times 10^{-3}$ | **.71 \times 10^{-3}$** | .75\%$ | **.42\%$** |
| **1440** | .5138$ | .99 \times 10^{-1}$ | .57 \times 10^{-3}$ | **.57 \times 10^{-3}$** | .72\%$ | **.69\%$** |
| **1600** | .5181$ | .00 \times 10^{-1}$ | .08 \times 10^{-3}$ | **.84 \times 10^{-3}$** | .94\%$ | **.36\%$** |

> **QPSK Insights:** In QPSK, the large decision boundary ({\\min}=0.707$) ensures zero bit errors after CPR across moderate distances ((6.4) \approx 10^{-10}$). Zero-error points are represented honestly with hollow markers at the empirical resolution limit /N$ (.22 \times 10^{-4}$), with 95% confidence upper bound $< 3.67 \times 10^{-4}$. The Neural Network Equalizer provides dramatic **analog constellation variance (EVM) compression**, reducing EVM from .13\%$ to .39\%$ at \text{ km}$ (**.5\%$ variance reduction**).

---

### 5.2 Nonlinear Impairment Mitigation in 16-QAM (Distance Sweep at $+4.0\text{ dBm}$)

16-QAM possesses 3 concentric amplitude rings ($|s|^2 \in \{0.2, 1.0, 1.8\}$), causing differential Kerr self-phase modulation (constellation spiraling) and a much narrower decision margin ({\\min} = 0.316$).

| Distance (km) | No-EQ BER | CDC (Raw, No CPR) BER | CDC (+ CPR) BER | Neural Network Equalizer BER | BER Reduction [%] | 7% HD-FEC Status (.8 \times 10^{-3}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **80** | .4537$ | .12 \times 10^{-5}$ | .0000$ | **.0000$** | — | **Passed** |
| **160** | .4843$ | .44 \times 10^{-2}$ | .12 \times 10^{-5}$ | **.0000$** | **100%** | **Passed** |
| **240** | .4925$ | .41 \times 10^{-1}$ | .51 \times 10^{-4}$ | **.84 \times 10^{-4}$** | **66.7%** | **Passed** |
| **400** | .5004$ | .56 \times 10^{-1}$ | .97 \times 10^{-3}$ | **.41 \times 10^{-3}$** | **55.8%** | **Near FEC Threshold** |
| **480** | .4954$ | .17 \times 10^{-1}$ | .38 \times 10^{-3}$ | **.77 \times 10^{-3}$** | **43.1%** | **Passed** |
| **560** | .4992$ | .73 \times 10^{-1}$ | .25 \times 10^{-2}$ | **.32 \times 10^{-3}$** | **33.3%** | Improved |
| **640** | .5073$ | .13 \times 10^{-1}$ | .82 \times 10^{-2}$ | **.06 \times 10^{-2}$** | **41.6%** | Improved |
| **800** | .5086$ | .78 \times 10^{-1}$ | .50 \times 10^{-2}$ | **.83 \times 10^{-2}$** | **19.2%** | Improved |

---

### 5.3 Nonlinear Penalty & Power Sweep at \text{ km}$ (16-QAM)

| Launch Power | Kerr SPM Rotation | CDC (Raw, No CPR) BER | CDC (+ CPR) BER | NN Equalizer BER | BER Reduction [%] | 7% HD-FEC Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$-2.0\text{ dBm}$** | $+8.4^\circ$ | .28 \times 10^{-3}$ | .12 \times 10^{-5}$ | **.12 \times 10^{-5}$** | — | **Passed** |
| **.0\text{ dBm}$** | $+12.8^\circ$ | .47 \times 10^{-2}$ | .95 \times 10^{-4}$ | **.34 \times 10^{-4}$** | **7.7%** | **Passed** |
| **$+2.0\text{ dBm}$** | $+21.8^\circ$ | .61 \times 10^{-1}$ | .79 \times 10^{-4}$ | **.67 \times 10^{-4}$** | **62.5%** | **Passed** |
| **$+4.0\text{ dBm}$** | **$+34.9^\circ$** | .53 \times 10^{-1}$ | .24 \times 10^{-3}$ (Failed FEC) | **.73 \times 10^{-3}$** | **40.2%** | **Passed FEC Threshold!** |
| **$+6.0\text{ dBm}$** | **$+61.5^\circ$** | .99 \times 10^{-1}$ | .96 \times 10^{-2}$ | **.93 \times 10^{-2}$** | — | Deep Nonlinear |

---

## 6. Repository File Structure

```
Optical-Communication-Links/
├── tx.py                  # Transmitter: Gray QPSK/16-QAM mapping, RRC pulse shaping, [√W] scaling
├── fiber.py               # Optical channel: Symmetric SSFM solving NLSE, in-line EDFA + ASE noise
├── rx.py                  # Receiver: Matched filtering, deterministic timing, amplitude normalization
├── dsp.py                 # Conventional DSP: FD-CDC dispersion compensation, waveform reconstruction
├── ml.py                  # Neural network: Leakage-free dataset builder, MLP equalizer, FeatureScaler
├── plots.py               # Visualizations: Eye diagrams, constellations, BER curves, training loss
├── dashboard.py           # Streamlit dashboard: Real-time sliders, dual eye diagrams, metrics
├── generate_ber_curve.py  # Benchmark suite: Multi-burst training and evaluation sweeps
├── test_project.py        # Comprehensive automated test suite
├── test_stage1.py         # Physical model acceptance gates (A to G)
├── test_stage2.py         # ML pipeline acceptance gates (H to L)
├── models/
│   ├── equalizer_qpsk.pt  # Dedicated QPSK neural network checkpoint
│   ├── equalizer_16qam.pt # Dedicated 16-QAM neural network checkpoint
│   └── equalizer.pt       # Default copy for legacy dashboard compatibility
└── results/
    ├── benchmark_results.csv   # Exact numerical evaluation records
    ├── ber_curve.npy           # Benchmark arrays
    ├── ber_vs_distance.png     # Reach extension plot
    ├── ber_vs_power.png        # Kerr nonlinearity mitigation plot
    ├── constellation_before.png # Transmitted vs Uncompensated
    ├── constellation_after.png  # Multi-stage overlay: Tx, CDC (Raw), CDC (CPR), NN-EQ
    ├── constellation_grid.png   # 2x2 side-by-side subplot panel of all 4 stages
    ├── training_curve_qpsk.png
    └── training_curve_16qam.png
```

---

## 7. Research References

[1] E. Ip & J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," *J. Lightw. Technol.*, vol. 26, no. 20, pp. 3416–3425, 2008.  
[2] M. A. Jarajreh *et al.*, "Artificial neural network nonlinear equalizer for coherent optical OFDM," *IEEE Photon. Technol. Lett.*, vol. 27, no. 4, pp. 387–390, 2015.  
[3] P. J. Freire *et al.*, "Neural networks-based equalizers for coherent optical transmission: Caveats and pitfalls," *IEEE J. Sel. Topics Quantum Electron.*, vol. 28, no. 4, 2022.  
[4] J. Ding *et al.*, "Intra-channel nonlinearity mitigation using perturbation-based neural network," *J. Lightw. Technol.*, vol. 40, no. 21, 2022.  
[5] V. Neskorniuk *et al.*, "Neural-network-based nonlinearity equalizer for 128 GBaud coherent transceivers," *Proc. OFC*, 2021.
