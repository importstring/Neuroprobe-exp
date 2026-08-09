"""
Synthetic end-to-end smoke test. No braintreebank data required.

Fabricates a feature cache with the exact on-disk layout that cache_build.py produces, plants a
weak signal in a few electrodes, then runs the E0 gate, 2 epochs of training, and the full
evaluation sweep. If this passes on your Windows box, the plumbing works and you can start the
real 6-hour cache build with some confidence instead of discovering a collate bug at hour five.

    python scripts/99_synthetic_smoke.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", tempfile.gettempdir())
_work = os.path.join(tempfile.gettempdir(), "npx_synthetic")
shutil.rmtree(_work, ignore_errors=True)
os.environ["NPX_WORK_DIR"] = _work

import numpy as np

from npx.config import ALL_TASKS, EVAL_SESSIONS, TRAIN_SESSION, FeatureConfig, TrainConfig

N_ELEC, N_WIN = 40, 300
SIGNAL_ELEC = [3, 4, 5]


def fabricate(cfg, sid, tid, seed):
    rng = np.random.RandomState(seed)
    Fb, Tb = cfg.n_freq_bins(), cfg.n_time_bins()
    d = cfg.cache_dir() / f"btbank{sid}_{tid}"
    d.mkdir(parents=True, exist_ok=True)

    feats = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float16,
                                      shape=(N_WIN, N_ELEC, Fb, Tb))
    y = (rng.rand(N_WIN) > 0.5).astype(np.float32)
    base = rng.randn(N_WIN, N_ELEC, Fb, Tb).astype(np.float32)
    # plant a weak, anatomically localized, high-frequency, mid-window effect
    base[y == 1][:, SIGNAL_ELEC, Fb // 2:, Tb // 3:2 * Tb // 3] += 0.8
    for i in range(N_WIN):
        if y[i] == 1:
            base[i, SIGNAL_ELEC, Fb // 2:, Tb // 3:2 * Tb // 3] += 0.8
    feats[:] = base.astype(np.float16)
    feats.flush()

    x = np.asarray(feats, dtype=np.float32)
    np.save(d / "norm_mean.npy", x.mean(axis=(0, 3))[:, :, None].astype(np.float32))
    np.save(d / "norm_std.npy", np.maximum(x.std(axis=(0, 3)), 1e-3)[:, :, None].astype(np.float32))
    np.save(d / "windows.npy", np.stack([np.arange(N_WIN) * 2048,
                                         np.arange(N_WIN) * 2048 + 2048], 1).astype(np.int64))
    # subject-specific coordinate offset, so cross-subject transfer is actually being tested
    coords = (rng.randn(N_ELEC, 3) * 20 + np.array([0.0, -18.0, 12.0]) + sid * 2.0).astype(np.float32)
    np.save(d / "coords.npy", coords)
    labels = [f"S{sid}a{i}" for i in range(N_ELEC)]
    (d / "electrode_labels.json").write_text(json.dumps(labels))
    for t in ALL_TASKS:
        perm = rng.permutation(N_WIN)
        np.save(d / f"rows_{t}.npy", perm.astype(np.int64))
        np.save(d / f"labels_{t}.npy", y[perm] if t in ("onset", "speech")
                else (rng.rand(N_WIN) > 0.5).astype(np.float32))
    (d / "meta.json").write_text(json.dumps({"subject_id": sid, "trial_id": tid,
                                             "n_windows": N_WIN, "n_electrodes": N_ELEC,
                                             "synthetic": True}))
    (d / "DONE").write_text("ok")


def main():
    cfg = FeatureConfig()
    print(f"synthetic cache -> {cfg.cache_dir()}")
    fabricate(cfg, *TRAIN_SESSION, seed=0)
    for i, (sid, tid) in enumerate(EVAL_SESSIONS):
        fabricate(cfg, sid, tid, seed=100 + i)

    print("\n--- selfcheck ---")
    from npx import selfcheck
    selfcheck.PASS.clear(); selfcheck.FAIL.clear()
    try:
        selfcheck.main()
    except SystemExit:
        raise SystemExit("selfcheck failed; fix that before anything else")

    print("\n--- E0 gate (synthetic: the 0.539 target does NOT apply here) ---")
    from npx.baseline_linear import run as run_gate
    run_gate(cfg, ["onset", "speech", "pitch"], n_regions=6)

    print("\n--- train 2 epochs ---")
    from npx.train import train
    tcfg = TrainConfig(run_tag="synthetic_smoke", tasks=["onset", "speech", "pitch"],
                       num_epochs=2, batch_size=16, num_workers=0, warmup_epochs=1, patience=99)
    train(cfg, tcfg)

    print("\n--- eval sweep + submission scaffold ---")
    from npx.evaluate import sweep
    sweep("synthetic_smoke", cfg, ["onset", "speech", "pitch"], write_submission=True,
          meta={"model_name": "synthetic", "author": "Simon Bergeron"})
    print("\nSMOKE TEST COMPLETE. onset should be well above 0.5; pitch should be ~0.5 "
          "(its synthetic labels are random by construction).")


if __name__ == "__main__":
    main()
