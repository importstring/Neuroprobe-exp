"""
npx/features.py  --  re-referencing, domain alignment, and spectrogram computation.

Replaces `build_laplacian_matrix`, `_laplacian_reference`, `_spectrogram`, and
`estimate_train_spec_stats` from the old notebook.

Three substantive changes from your version:

1. Laplacian is built from ELECTRODE LABELS (stem +/- 1 on the same shaft), not from a kNN
   graph over coordinates. That is what BrainBERT and the #1 leaderboard entry actually do.
   A kNN graph in cortical-projection space happily connects electrodes on different shafts
   in different lobes, which is not a Laplacian derivation -- it is noise injection.

2. Euclidean Alignment is applied per session. Each session's mean covariance R is whitened
   out: X <- R^{-1/2} X. This is the standard fix for cross-subject covariate shift and is
   reported at +6.5 to +7.4pp in the EEG transfer literature. It is cheap and it is the
   single change with the highest expected value in this pipeline.

3. Normalization statistics are GLOBAL per (electrode, frequency bin) across the session,
   not per sample. Per-sample z-scoring removes exactly the information the classifier needs:
   the absolute power level in a band. On this leaderboard the identical architecture scored
   0.522 with per-window-z and 0.547 with global-z. You were paying that 0.025 for free.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from .config import FeatureConfig, SAMPLING_RATE


# ---------------------------------------------------------------------------
# Electrode label parsing
# ---------------------------------------------------------------------------
_STEM_RE = re.compile(r"^(.*?)(\d+)$")


def stem_electrode_name(name: str):
    """'T1bIc12' -> ('T1bIc', 12).  Matches neuroprobe's own implementation."""
    m = _STEM_RE.match(name.strip())
    if not m:
        return name.strip(), -1
    return m.group(1), int(m.group(2))


def build_shaft_laplacian(electrode_labels) -> np.ndarray:
    """
    Return W [E, E] such that  x_lap = x - W @ x  is the Laplacian-re-referenced signal.

    For each electrode, neighbours are the electrodes with the SAME stem and index +/- 1.
    Electrodes without both neighbours get W row = 0 (i.e. they stay raw). That mirrors
    neuroprobe's `_get_all_laplacian_electrodes`, which simply excludes those electrodes;
    keeping them raw is strictly more information and is what the top entry does.
    """
    labels = [str(l) for l in electrode_labels]
    E = len(labels)
    index = {}
    for i, l in enumerate(labels):
        stem, num = stem_electrode_name(l)
        index[(stem, num)] = i

    W = np.zeros((E, E), dtype=np.float32)
    n_derived = 0
    for i, l in enumerate(labels):
        stem, num = stem_electrode_name(l)
        neigh = [index.get((stem, num - 1)), index.get((stem, num + 1))]
        neigh = [j for j in neigh if j is not None]
        if len(neigh) == 2:  # only a true bipolar-symmetric Laplacian
            for j in neigh:
                W[i, j] = 0.5
            n_derived += 1
    return W, n_derived


def build_knn_laplacian(coords: np.ndarray, k: int = 4) -> np.ndarray:
    """The old coordinate-kNN version. Kept ONLY so you can ablate it. Do not use as default."""
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float32)
    E = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, E), metric="euclidean").fit(coords)
    _, idx = nn.kneighbors(coords)
    W = np.zeros((E, E), dtype=np.float32)
    for e in range(E):
        neigh = [j for j in idx[e] if j != e]
        if not neigh:
            continue
        d = np.array([max(np.linalg.norm(coords[e] - coords[j]), 1e-6) for j in neigh], dtype=np.float32)
        w = (1.0 / d)
        w /= w.sum()
        for wi, j in zip(w, neigh):
            W[e, j] = wi
    return W


def apply_reref(x: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """x: [E, T] -> [E, T]"""
    if W is None:
        return x
    return x - W @ x


# ---------------------------------------------------------------------------
# Euclidean Alignment
# ---------------------------------------------------------------------------
def _matrix_inv_sqrt(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R.astype(np.float64))
    w = np.clip(w, eps * float(np.max(w)) if np.max(w) > 0 else eps, None)
    return (V @ np.diag(w ** -0.5) @ V.T).astype(np.float32)


def estimate_ea_matrix(windows: np.ndarray, max_windows: int = 512, shrink: float = 0.05) -> np.ndarray:
    """
    windows: [N, E, T] re-referenced voltage from ONE session.
    Returns A = R^{-1/2}, applied as x_aligned = A @ x.

    Ledoit-Wolf-style shrinkage toward a scaled identity keeps this stable when N is small
    relative to E (120 electrodes, so you want at least ~300 windows).
    """
    N = windows.shape[0]
    sel = np.linspace(0, N - 1, num=min(N, max_windows)).astype(int)
    E = windows.shape[1]
    R = np.zeros((E, E), dtype=np.float64)
    for i in sel:
        x = windows[i].astype(np.float64)
        R += (x @ x.T) / x.shape[1]
    R /= len(sel)
    mu = float(np.trace(R) / E)
    R = (1.0 - shrink) * R + shrink * mu * np.eye(E)
    return _matrix_inv_sqrt(R)


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------
def stft_features(x: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """
    x: [E, T] float32 -> [E, F, Tb] float32, frequency-cropped, log1p-magnitude, NOT normalized.
    Normalization happens later using session-global statistics.
    """
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    window = torch.hann_window(cfg.win_len, dtype=torch.float32)
    S = torch.stft(
        xt,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win_len,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mag = S.abs()
    if cfg.log1p:
        mag = torch.log1p(mag)
    F_keep = cfg.n_freq_bins()
    mag = mag[:, :F_keep, :]
    return mag.numpy().astype(np.float32)


def freq_axis_hz(cfg: FeatureConfig) -> np.ndarray:
    return np.arange(cfg.n_freq_bins()) * (SAMPLING_RATE / cfg.n_fft)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def estimate_global_stats(feats_memmap, batch: int = 256):
    """
    feats_memmap: [N, E, F, T] -> (mean [E,F,1], std [E,F,1]) as float32.
    Streamed in chunks so this works on a 6 GB cache with 16 GB of RAM.
    """
    N = feats_memmap.shape[0]
    E, Fb, Tb = feats_memmap.shape[1:]
    s = np.zeros((E, Fb), dtype=np.float64)
    ss = np.zeros((E, Fb), dtype=np.float64)
    n = 0
    for i in range(0, N, batch):
        chunk = np.asarray(feats_memmap[i:i + batch], dtype=np.float64)  # [b,E,F,T]
        s += chunk.sum(axis=(0, 3))
        ss += (chunk ** 2).sum(axis=(0, 3))
        n += chunk.shape[0] * Tb
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 1e-12)
    return mean.astype(np.float32)[:, :, None], np.sqrt(var).astype(np.float32)[:, :, None]
