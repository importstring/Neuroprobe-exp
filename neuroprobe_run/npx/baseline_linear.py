"""
npx/baseline_linear.py  --  the E0 gate. Run this BEFORE anything else.

There is no equivalent in the old notebook, which is the single biggest process problem you had:
you were tuning a 2M-parameter model with no reference point, so you could not tell a bug from
a null result.

The gate: reproduce the published cross-subject linear baseline, ~0.539 mean AUROC over
15 tasks x 10 sessions, within +/- 0.015. If you cannot, you have a pipeline bug -- wrong
window alignment, wrong normalization, mismatched coordinates, or leakage -- and every fancier
experiment you run on top of it is uninterpretable.

Method: electrodes are pooled into a fixed set of anatomical regions by k-means over MNI
coordinates pooled across ALL subjects (so region k means the same anatomy in every subject),
each region gets the mean log-power in 8 canonical bands over 3 time thirds, and logistic
regression is fit on the resulting fixed-length vector. Region pooling is what makes a linear
model transferable across montages at all.

Run:
    python -m scripts.00_gate_linear --tasks all
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .cache_io import SessionCache, val_test_ranges
from .config import ALL_TASKS, EVAL_SESSIONS, LOG_ROOT, SAMPLING_RATE, TRAIN_SESSION, FeatureConfig
from .features import freq_axis_hz

BANDS = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 55), (65, 95), (95, 125), (125, 150)]


def band_indices(cfg: FeatureConfig):
    hz = freq_axis_hz(cfg)
    return [np.where((hz >= lo) & (hz < hi))[0] for lo, hi in BANDS]


def fit_region_atlas(fcfg: FeatureConfig, n_regions=24, seed=0):
    """One k-means over all subjects' coordinates. Region k = same anatomy for everyone."""
    pts = []
    for sid, tid in [TRAIN_SESSION] + EVAL_SESSIONS:
        try:
            pts.append(SessionCache(fcfg, sid, tid).coords)
        except FileNotFoundError:
            pass
    allpts = np.concatenate(pts, 0)
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(allpts)
    return km


def session_features(cache: SessionCache, task: str, km, bidx, sample_range=None):
    rows, labels = cache.rows(task), cache.labels(task)
    if sample_range is not None:
        lo, hi = sample_range
        rows, labels = rows[lo:hi], labels[lo:hi]
    region = km.predict(cache.coords)                       # [E]
    n_reg = km.n_clusters
    feats = cache.feats
    mean, std = cache.norm_mean, np.maximum(cache.norm_std, 1e-6)

    Tb = feats.shape[-1]
    thirds = [slice(0, Tb // 3), slice(Tb // 3, 2 * Tb // 3), slice(2 * Tb // 3, Tb)]
    X = np.zeros((len(rows), n_reg, len(bidx), len(thirds)), dtype=np.float32)
    for n, r in enumerate(rows):
        x = (np.asarray(feats[int(r)], dtype=np.float32) - mean) / std   # [E,F,T]
        for bi, fsel in enumerate(bidx):
            if len(fsel) == 0:
                continue
            bp = x[:, fsel, :].mean(1)                                    # [E,T]
            for ti, sl in enumerate(thirds):
                v = bp[:, sl].mean(1)                                     # [E]
                for k in range(n_reg):
                    m = region == k
                    X[n, k, bi, ti] = v[m].mean() if m.any() else 0.0
    return X.reshape(len(rows), -1), labels


def run(fcfg: FeatureConfig, tasks, n_regions=24, C=1.0, verbose=True):
    km = fit_region_atlas(fcfg, n_regions=n_regions)
    bidx = band_indices(fcfg)
    train_cache = SessionCache(fcfg, *TRAIN_SESSION)
    results = {}

    for task in tasks:
        try:
            Xtr, ytr = session_features(train_cache, task, km, bidx)
        except FileNotFoundError:
            continue
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(Xtr), ytr)

        per_session = {}
        for sid, tid in EVAL_SESSIONS:
            try:
                c = SessionCache(fcfg, sid, tid)
                _, tr = val_test_ranges(c, task)   # report on the TEST half, as the board does
                Xte, yte = session_features(c, task, km, bidx, sample_range=tr)
            except (FileNotFoundError, AssertionError):
                continue
            p = clf.decision_function(scaler.transform(Xte))
            per_session[f"btbank{sid}_{tid}"] = float(roc_auc_score(yte, p))
        results[task] = per_session
        if verbose and per_session:
            print(f"  {task:18s} mean={np.mean(list(per_session.values())):.4f}  "
                  f"n_sessions={len(per_session)}")

    per_task_mean = {t: float(np.mean(list(v.values()))) for t, v in results.items() if v}
    overall = float(np.mean(list(per_task_mean.values()))) if per_task_mean else float("nan")

    out = {"overall": overall, "per_task": per_task_mean, "detail": results,
           "n_regions": n_regions, "feature_hash": fcfg.hash()}
    (LOG_ROOT / "gate_linear.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== E0 GATE ===\noverall linear cross-subject AUROC = {overall:.4f}")
    print("published reference = 0.539   tolerance = +/- 0.015")
    if np.isnan(overall):
        print("VERDICT: no results. Cache is not built.")
    elif abs(overall - 0.539) <= 0.015:
        print("VERDICT: PASS. Pipeline is sane. Proceed to E1.")
    elif overall < 0.524:
        print("VERDICT: FAIL LOW. Do not tune anything. Check, in this order: "
              "(1) window alignment -- are you reading 2048 samples from word onset; "
              "(2) normalization -- is norm_std nonzero everywhere; "
              "(3) coordinates -- is each session using ITS OWN coords.npy; "
              "(4) label order -- do rows_{task}.npy and labels_{task}.npy line up.")
    else:
        print("VERDICT: FAIL HIGH. A linear model beating 0.554 cross-subject means leakage. "
              "Check that you are not fitting on any part of the evaluation sessions and that "
              "normalization statistics for eval sessions were not computed with labels present.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--regions", type=int, default=24)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--reref", default="shaft")
    ap.add_argument("--align", default="ea")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    a = ap.parse_args()
    tasks = ALL_TASKS if a.tasks == ["all"] else a.tasks
    fcfg = FeatureConfig(reref=a.reref, align=a.align, n_fft=a.n_fft, hop=a.hop, win_len=a.n_fft)
    run(fcfg, tasks, n_regions=a.regions, C=a.C)


if __name__ == "__main__":
    main()
