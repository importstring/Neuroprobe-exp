"""
npx/cache_build.py  --  build the precomputed feature cache.

This is the file that buys back most of your four months of compute.

The old notebook recomputed the Laplacian matrix and the STFT inside `__getitem__`, once per
sample, once per task, once per epoch, on a single worker because `num_workers=0` is forced on
Windows notebooks. With 6 tasks x 20 epochs that is 120 recomputations of identical arithmetic
on identical neural windows.

All 15 tasks draw from the SAME pool of 1-second windows in the same session -- the tasks
differ only in how each window is labelled. So: enumerate the union of windows once, compute
features once, write them to a float16 memmap, and let every task and every epoch read the
same bytes.

Cost, per session, at n_fft=512 / hop=128 / 0-150 Hz / 120 electrodes:
    38 freq bins x 17 time bins = 646 floats per electrode
    120 electrodes x 646 x 2 bytes = 151 KB per window
    ~3,500 windows -> ~543 MB per session
    11 sessions -> ~6.0 GB total, which fits in your 16 GB.

Contrast with your current n_fft=64 / hop=16 spec: 33 x 129 = 4,257 floats per electrode,
998 KB per window -- 6.6x more expensive AND much worse, because 64-point FFT at 2048 Hz gives
32 Hz per bin, which folds delta, theta, alpha and beta into a single number.

Run:
    python -m scripts.01_build_cache --sessions eval train
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import (
    ALL_TASKS, CACHE_ROOT, EVAL_SESSIONS, PRETRAIN_SESSIONS, SAMPLING_RATE,
    TRAIN_SESSION, WINDOW_SAMPLES, FeatureConfig,
)
from .features import (
    apply_reref, build_knn_laplacian, build_shaft_laplacian, estimate_ea_matrix,
    estimate_global_stats, stft_features,
)


def session_dir(cfg: FeatureConfig, sid: int, tid: int) -> Path:
    d = cfg.cache_dir() / f"btbank{sid}_{tid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_built(cfg: FeatureConfig, sid: int, tid: int) -> bool:
    return (session_dir(cfg, sid, tid) / "DONE").exists()


def _load_subject(sid: int, coordinate_system: str):
    from neuroprobe import BrainTreebankSubject

    return BrainTreebankSubject(
        subject_id=sid,
        cache=True,
        dtype=torch.float32,
        coordinates_type=coordinate_system,
        allow_missing_coordinates=False,  # FAIL LOUDLY. See the fallback_coords note in README.
    )


def enumerate_windows(subject, tid: int, cfg: FeatureConfig, tasks=None):
    """
    Returns:
        windows   : [N, 2] int64, sorted unique (index_from, index_to)
        per_task  : {task: {"rows": [n_t] int64 into `windows`, "labels": [n_t] float32}}
        elec_labels, coords
    """
    from neuroprobe import BrainTreebankSubjectTrialBenchmarkDataset

    tasks = tasks or ALL_TASKS
    raw = {}
    elec_labels = coords = None

    for task in tasks:
        ds = BrainTreebankSubjectTrialBenchmarkDataset(
            subject, trial_id=tid, dtype=torch.float32, eval_name=task,
            output_indices=True, lite=cfg.lite, nano=cfg.nano, output_dict=True,
        )
        if elec_labels is None:
            elec_labels = [str(x) for x in ds.electrode_labels]
            coords = np.asarray(ds.electrode_coordinates, dtype=np.float32)
            assert np.isfinite(coords).all(), (
                f"Non-finite coordinates for subject {subject.subject_id}. "
                "With coordinates_type='cortical' neuroprobe silently returns NaN for "
                "electrodes it cannot project. Use 'mni152'."
            )
        idxs, labels = [], []
        for i in range(len(ds)):
            item = ds[i]
            a, b = item["data"]
            idxs.append((int(a), int(b)))
            labels.append(float(item["label"]))
        raw[task] = (idxs, np.asarray(labels, dtype=np.float32))
        del ds

    all_idx = sorted({t for idxs, _ in raw.values() for t in idxs})
    lookup = {t: i for i, t in enumerate(all_idx)}
    windows = np.asarray(all_idx, dtype=np.int64)

    per_task = {}
    for task, (idxs, labels) in raw.items():
        per_task[task] = {
            "rows": np.asarray([lookup[t] for t in idxs], dtype=np.int64),
            "labels": labels,
        }
    return windows, per_task, elec_labels, coords


def build_session(cfg: FeatureConfig, sid: int, tid: int, coordinate_system: str,
                  tasks=None, verbose: bool = True) -> None:
    out = session_dir(cfg, sid, tid)
    if (out / "DONE").exists():
        if verbose:
            print(f"  btbank{sid}_{tid}: already built, skipping")
        return

    t0 = time.time()
    subject = _load_subject(sid, coordinate_system)
    subject.set_electrode_subset(subject.get_lite_electrodes() if cfg.lite else subject.electrode_labels)

    windows, per_task, elec_labels, coords = enumerate_windows(subject, tid, cfg, tasks=tasks)
    N, E = len(windows), len(elec_labels)
    Fb, Tb = cfg.n_freq_bins(), cfg.n_time_bins()
    if verbose:
        print(f"  btbank{sid}_{tid}: {N} unique windows, {E} electrodes -> [{N},{E},{Fb},{Tb}] fp16 "
              f"({N * E * Fb * Tb * 2 / 1e9:.2f} GB)")

    # --- re-referencing operator -------------------------------------------------
    if cfg.reref == "shaft":
        W, n_derived = build_shaft_laplacian(elec_labels)
        if verbose:
            print(f"    shaft Laplacian: {n_derived}/{E} electrodes have both neighbours")
        assert n_derived >= E // 3, (
            f"Only {n_derived}/{E} electrodes are Laplacian-derivable. Electrode labels are "
            f"probably not in the expected 'STEM<int>' form. First few: {elec_labels[:5]}"
        )
    elif cfg.reref == "knn":
        W = build_knn_laplacian(coords, k=4)
    elif cfg.reref == "none":
        W = None
    else:
        raise ValueError(cfg.reref)

    subject.load_neural_data(tid)
    subject.cache_neural_data(tid)  # ~3-4 GB for a full trial; one session at a time

    def read(i):
        a, b = int(windows[i, 0]), int(windows[i, 1])
        x = subject.get_all_electrode_data(tid, window_from=a, window_to=b)
        x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        assert x.shape == (E, WINDOW_SAMPLES), f"expected ({E},{WINDOW_SAMPLES}), got {x.shape}"
        return apply_reref(x.astype(np.float32), W)

    # --- pass 1: Euclidean Alignment matrix from a subsample ---------------------
    A = None
    if cfg.align == "ea":
        sub = np.linspace(0, N - 1, num=min(N, 512)).astype(int)
        stack = np.stack([read(i) for i in sub], axis=0)
        A = estimate_ea_matrix(stack)
        del stack
        gc.collect()
        if verbose:
            print(f"    EA matrix cond={np.linalg.cond(A):.1f}")
    elif cfg.align != "none":
        raise ValueError(cfg.align)

    # --- pass 2: features -> memmap ---------------------------------------------
    feats = np.lib.format.open_memmap(
        out / "feats.npy", mode="w+", dtype=np.dtype(cfg.dtype), shape=(N, E, Fb, Tb)
    )
    for i in range(N):
        x = read(i)
        if A is not None:
            x = A @ x
        feats[i] = stft_features(x, cfg).astype(cfg.dtype)
        if verbose and (i + 1) % 500 == 0:
            print(f"    {i + 1}/{N}  ({time.time() - t0:.0f}s)")
    feats.flush()

    subject.clear_neural_data_cache(tid)
    del subject
    gc.collect()

    # --- global normalization statistics ----------------------------------------
    mean, std = estimate_global_stats(feats)
    np.save(out / "norm_mean.npy", mean)
    np.save(out / "norm_std.npy", std)

    np.save(out / "windows.npy", windows)
    np.save(out / "coords.npy", coords)
    if A is not None:
        np.save(out / "ea.npy", A)
    (out / "electrode_labels.json").write_text(json.dumps(elec_labels))
    for task, d in per_task.items():
        np.save(out / f"rows_{task}.npy", d["rows"])
        np.save(out / f"labels_{task}.npy", d["labels"])
    (out / "meta.json").write_text(json.dumps({
        "subject_id": sid, "trial_id": tid, "n_windows": int(N), "n_electrodes": int(E),
        "shape": [int(N), int(E), int(Fb), int(Tb)], "coordinate_system": coordinate_system,
        "tasks": sorted(per_task.keys()),
        "task_n_samples": {k: int(len(v["labels"])) for k, v in per_task.items()},
        "sampling_rate": SAMPLING_RATE, "build_seconds": round(time.time() - t0, 1),
    }, indent=2))
    (out / "DONE").write_text("ok")
    del feats
    gc.collect()
    if verbose:
        print(f"  btbank{sid}_{tid}: done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["train", "eval"],
                    choices=["train", "eval", "pretrain", "all"])
    ap.add_argument("--coordinate-system", default="mni152")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Default: all 15. Labels are cheap; build all 15 once.")
    ap.add_argument("--reref", default="shaft")
    ap.add_argument("--align", default="ea")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--freq-max", type=float, default=150.0)
    ap.add_argument("--norm", default="global_elec_freq")
    args = ap.parse_args()

    cfg = FeatureConfig(reref=args.reref, align=args.align, n_fft=args.n_fft, hop=args.hop,
                        win_len=args.n_fft, freq_max_hz=args.freq_max, norm=args.norm)
    print(f"FeatureConfig hash={cfg.hash()}  cache={cfg.cache_dir()}")
    print(f"  F={cfg.n_freq_bins()} bins @ {SAMPLING_RATE / cfg.n_fft:.2f} Hz, T={cfg.n_time_bins()}")

    todo = []
    if "train" in args.sessions or "all" in args.sessions:
        todo.append(TRAIN_SESSION)
    if "eval" in args.sessions or "all" in args.sessions:
        todo += EVAL_SESSIONS
    if "pretrain" in args.sessions or "all" in args.sessions:
        todo += PRETRAIN_SESSIONS
    seen, uniq = set(), []
    for s in todo:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    for sid, tid in uniq:
        try:
            build_session(cfg, sid, tid, args.coordinate_system, tasks=args.tasks)
        except Exception as e:  # keep going; log the failure
            print(f"  !! btbank{sid}_{tid} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
