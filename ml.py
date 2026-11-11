"""
ml.py — Neural-Network Equalizer Module
========================================
Implements a lightweight, window-based fully-connected equalizer that
compensates optical fiber impairments (residual chromatic dispersion and
Kerr nonlinear phase noise) using supervised learning.

Architecture:
    Input(2W) → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(2)

    W = context window width (odd integer, e.g. 11)
    Input:  real and imaginary parts of W consecutive normalized received symbols
    Output: (Re, Im) of the estimated transmitted center symbol

Engineering Rules Enforced:
- Supervised labels: X = window of received symbols, Y = center transmitted symbol.
- Strict train/val/test separation: bursts or non-overlapping time blocks (no leakage).
- Standardized feature scaling fitted on training set only and saved with model weights.
- Baseline loss reporting (zero predictor & identity predictor).
- Modulation-specific metadata and compatibility checking.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Default Hyper-parameters ─────────────────────────────────────────────────
WINDOW_DEFAULT   = 11     # context window width (odd); input dim = 2 × WINDOW
HIDDEN_DEFAULT   = 64     # hidden layer width
EPOCHS_DEFAULT   = 60     # training epochs
BATCH_DEFAULT    = 128    # mini-batch size
LR_DEFAULT       = 2e-3   # Adam initial learning rate
VAL_FRAC_DEFAULT = 0.2    # validation fraction


# ─── Feature Scaler ───────────────────────────────────────────────────────────

class FeatureScaler:
    """
    Standardizes input features (zero mean, unit variance).
    Statistics are fitted on training data only and stored in checkpoints.
    """
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> FeatureScaler:
        self.mean = np.mean(X, axis=0, keepdims=True).astype(np.float32)
        std = np.std(X, axis=0, keepdims=True).astype(np.float32)
        # Prevent division by zero
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return X.astype(np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_dict(cls, d: dict | None) -> FeatureScaler:
        if d is None or 'mean' not in d or 'std' not in d:
            return cls()
        return cls(mean=d['mean'], std=d['std'])


# ─── Model ────────────────────────────────────────────────────────────────────

class NNEqualizer(nn.Module):
    """
    Lightweight fully-connected neural equalizer.

    Input  : flattened real + imaginary parts of a window of received symbols
             → tensor shape (batch, 2 * window)
    Output : (Re, Im) estimate of the center transmitted symbol
             → tensor shape (batch, 2)
    """

    def __init__(self, window: int = WINDOW_DEFAULT, hidden: int = HIDDEN_DEFAULT):
        super().__init__()
        if window % 2 == 0:
            raise ValueError(f"window must be an odd integer, got {window}")
        self.window = window
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(2 * window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Dataset Construction ─────────────────────────────────────────────────────

def build_dataset(rx_symbols: np.ndarray,
                  tx_symbols: np.ndarray,
                  window: int = WINDOW_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a sliding-window (X, Y) dataset from a stream of symbol pairs.

    For each center index i ∈ [half, N − half):
        X[i] = [Re(rx[i-half : i+half+1]),  Im(rx[i-half : i+half+1])]
        Y[i] = [Re(tx[i]),                  Im(tx[i])]

    Parameters
    ----------
    rx_symbols : np.ndarray, complex
        Received symbols (normalized to unit power).
    tx_symbols : np.ndarray, complex
        Ideal transmitted symbols.
    window : int
        Odd integer context window.

    Returns
    -------
    X : np.ndarray, float32, shape (n_samples, 2*window)
    Y : np.ndarray, float32, shape (n_samples, 2)
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")

    half = window // 2
    N = min(len(rx_symbols), len(tx_symbols))
    n = N - 2 * half
    if n <= 0:
        raise ValueError(f"Symbol sequence length {N} too short for window {window}.")

    X = np.empty((n, 2 * window), dtype=np.float32)
    Y = np.empty((n, 2), dtype=np.float32)

    for i in range(n):
        c = i + half
        win = rx_symbols[c - half : c + half + 1]
        X[i] = np.concatenate([win.real.astype(np.float32),
                               win.imag.astype(np.float32)])
        Y[i, 0] = float(tx_symbols[c].real)
        Y[i, 1] = float(tx_symbols[c].imag)

    return X, Y


def build_multiburst_dataset(bursts: list[tuple[np.ndarray, np.ndarray]],
                             window: int = WINDOW_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dataset by combining multiple independent bursts without edge leakage.
    Each burst is windowed separately, then concatenated.
    """
    X_list, Y_list = [], []
    for rx_syms, tx_syms in bursts:
        x, y = build_dataset(rx_syms, tx_syms, window=window)
        X_list.append(x)
        Y_list.append(y)
    return np.vstack(X_list), np.vstack(Y_list)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model: NNEqualizer,
          X_train: np.ndarray,
          Y_train: np.ndarray,
          X_val: np.ndarray | None = None,
          Y_val: np.ndarray | None = None,
          epochs: int = EPOCHS_DEFAULT,
          batch_size: int = BATCH_DEFAULT,
          lr: float = LR_DEFAULT,
          val_frac: float = VAL_FRAC_DEFAULT,
          seed: int = 42,
          verbose: bool = True) -> tuple[dict, FeatureScaler]:
    """
    Train the neural equalizer with Adam optimizer, MSE loss, and learning rate decay.
    Guarantees no data leakage between train and validation.

    Parameters
    ----------
    model : NNEqualizer
    X_train : np.ndarray
    Y_train : np.ndarray
    X_val : np.ndarray or None
        If provided, used directly as independent validation set.
    Y_val : np.ndarray or None
    epochs : int
    batch_size : int
    lr : float
    val_frac : float
        Fraction reserved for validation if X_val is None (contiguous block split).
    seed : int
    verbose : bool

    Returns
    -------
    history : dict
        Contains 'train_loss', 'val_loss', 'baseline_zero', 'baseline_identity'.
    scaler : FeatureScaler
        Fitted on training features only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Independent validation split (temporal block split, NOT random shuffle)
    if X_val is None or Y_val is None:
        n_total = len(X_train)
        n_val = max(1, int(n_total * val_frac))
        n_tr = n_total - n_val
        # Discard context-window edge gap between train and val to eliminate leakage
        gap = model.window
        tr_slice = slice(0, max(1, n_tr - gap))
        val_slice = slice(n_tr, n_total)

        X_tr_raw, Y_tr = X_train[tr_slice], Y_train[tr_slice]
        X_val_raw, Y_val = X_train[val_slice], Y_train[val_slice]
    else:
        X_tr_raw, Y_tr = X_train, Y_train
        X_val_raw, Y_val = X_val, Y_val

    # 2. Fit feature standardizer ON TRAINING DATA ONLY
    scaler = FeatureScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_v = scaler.transform(X_val_raw)

    # 3. Compute baseline losses for physical context
    loss_zero = float(np.mean(Y_tr ** 2))
    # Identity predictor: center received symbol (window has W real, W imag)
    half = model.window // 2
    x_center_re = X_tr_raw[:, half]
    x_center_im = X_tr_raw[:, model.window + half]
    loss_identity = float(np.mean((Y_tr[:, 0] - x_center_re) ** 2 +
                                  (Y_tr[:, 1] - x_center_im) ** 2))

    if verbose:
        print(f"  Training samples: {len(X_tr)}  |  Validation samples: {len(X_v)}")
        print(f"  Baseline Loss - Zero Predictor:     {loss_zero:.6f}")
        print(f"  Baseline Loss - Identity Predictor: {loss_identity:.6f}")

    # 4. PyTorch DataLoaders
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(Y_val))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'baseline_zero': loss_zero,
        'baseline_identity': loss_identity
    }

    best_val_loss = np.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        tr_loss = running_loss / len(X_tr)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_v)

        scheduler.step(val_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and ((epoch + 1) % 15 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch+1:3d}/{epochs}  train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}")

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, scaler


# ─── Inference ────────────────────────────────────────────────────────────────

def equalize(model: NNEqualizer,
             rx_symbols: np.ndarray,
             scaler: FeatureScaler | None = None,
             window: int | None = None) -> np.ndarray:
    """
    Apply the trained equalizer to received symbols.

    Parameters
    ----------
    model : NNEqualizer
    rx_symbols : np.ndarray, complex
        Normalized received symbol sequence.
    scaler : FeatureScaler or None
        Feature standardizer saved from training.
    window : int or None
        Window size (defaults to model.window).

    Returns
    -------
    eq_symbols : np.ndarray, complex
        Equalized symbol estimates of length len(rx_symbols) - 2*(window//2).
    """
    model.eval()
    w = window or model.window
    half = w // 2
    N = len(rx_symbols)
    n = N - 2 * half
    if n <= 0:
        return np.array([], dtype=complex)

    X = np.empty((n, 2 * w), dtype=np.float32)
    for i in range(n):
        c = i + half
        win = rx_symbols[c - half : c + half + 1]
        X[i] = np.concatenate([win.real.astype(np.float32),
                               win.imag.astype(np.float32)])

    if scaler is not None:
        X = scaler.transform(X)

    with torch.no_grad():
        y_hat = model(torch.from_numpy(X)).numpy()

    return (y_hat[:, 0] + 1j * y_hat[:, 1]).astype(complex)


# ─── Model Checkpoints & I/O ──────────────────────────────────────────────────

def save_model(model: NNEqualizer,
               scaler: FeatureScaler | None = None,
               modulation: str = 'QPSK',
               metadata: dict | None = None,
               path: str = 'models/equalizer.pt') -> None:
    """Save model weights, scaler, and configuration metadata to a checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ckpt = {
        'state_dict': model.state_dict(),
        'window': model.window,
        'hidden': model.hidden,
        'modulation': modulation,
        'scaler': scaler.to_dict() if scaler else None,
        'metadata': metadata or {},
        'version': '2.0',
    }
    torch.save(ckpt, path)
    print(f"  Model successfully saved -> {path}")


def load_model(path: str = 'models/equalizer.pt') -> tuple[NNEqualizer, FeatureScaler, dict]:
    """
    Load equalizer model, scaler, and metadata from checkpoint.

    Returns
    -------
    model : NNEqualizer
    scaler : FeatureScaler
    metadata : dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    window = ckpt.get('window', WINDOW_DEFAULT)
    hidden = ckpt.get('hidden', HIDDEN_DEFAULT)

    model = NNEqualizer(window=window, hidden=hidden)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    scaler = FeatureScaler.from_dict(ckpt.get('scaler'))
    metadata = {
        'modulation': ckpt.get('modulation', 'QPSK'),
        'window': window,
        'hidden': hidden,
        'version': ckpt.get('version', '1.0'),
        **(ckpt.get('metadata') or {})
    }

    return model, scaler, metadata
