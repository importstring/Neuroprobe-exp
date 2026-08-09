"""
npx/cache_io.py  --  torch Dataset / DataLoader over the precomputed feature cache.

Replaces `CrossSubjectLaplacianSpectrogramDataset`, `unpack_base_item`, `infer_subject_id`,
`infer_electrode_coordinates`, `get_fallback_coords` and `collate_cross_subject_batch`.

Three of those five functions existed only to work around not knowing what the base dataset
returned. `infer_subject_id` guessed among six key names. `infer_electrode_coordinates` guessed
among three and then SILENTLY FELL BACK to another subject's coordinates. Subjects 1, 4, 5, 6, 8
and 10 all have exactly 120 Lite electrodes, so your `coords.shape[0] != x.shape[0]` guard could
never catch that fallback -- you would have trained subject 10's signals against subject 1's
coordinate frame, on a benchmark whose entire difficulty is coordinate harmonization, and every
assert would have passed. That whole class of bug is gone here: coordinates are read from the
cache directory of the session they belong to, and there is no fallback path.

The val/test split reproduces neuroprobe exactly:
    val  = Subset(test_dataset, range(0, len // 2))
    test = Subset(test_dataset, range(len // 2, len))
Dataset order is deterministic under neuroprobe's fixed global seed, so cache row order is
the dataset order and these ranges are the same rows the official harness would pick.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import FeatureConfig


class SessionCache:
    """Lazy, memmap-backed handle on one built session. Cheap to construct in a worker."""

    def __init__(self, cfg: FeatureConfig, sid: int, tid: int):
        self.dir = cfg.cache_dir() / f"btbank{sid}_{tid}"
        if not (self.dir / "DONE").exists():
            raise FileNotFoundError(f"{self.dir} not built. Run scripts/01_build_cache.py first.")
        self.sid, self.tid, self.cfg = sid, tid, cfg
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.coords = np.load(self.dir / "coords.npy")
        self.electrode_labels = json.loads((self.dir / "electrode_labels.json").read_text())
        self.norm_mean = np.load(self.dir / "norm_mean.npy")  # [E,F,1]
        self.norm_std = np.load(self.dir / "norm_std.npy")
        self._feats = None

    @property
    def feats(self):
        if self._feats is None:  # opened per process, never pickled
            self._feats = np.load(self.dir / "feats.npy", mmap_mode="r")
        return self._feats

    def rows(self, task: str) -> np.ndarray:
        return np.load(self.dir / f"rows_{task}.npy")

    def labels(self, task: str) -> np.ndarray:
        return np.load(self.dir / f"labels_{task}.npy")

    def n_samples(self, task: str) -> int:
        return len(self.labels(task))

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_feats"] = None
        return s


class CachedTaskDataset(Dataset):
    """
    One (session, task, sample-range) view over the cache.

    sample_range: None for everything, or (lo, hi) to carve out val/test halves.
    """

    def __init__(self, cache: SessionCache, task: str, sample_range=None, train: bool = False,
                 tcfg=None):
        self.cache, self.task, self.train, self.tcfg = cache, task, train, tcfg
        rows, labels = cache.rows(task), cache.labels(task)
        lo, hi = (0, len(labels)) if sample_range is None else sample_range
        self.rows, self.labels = rows[lo:hi], labels[lo:hi]
        u = np.unique(self.labels)
        assert set(u.tolist()) <= {0.0, 1.0}, f"{task}: labels are not binary: {u[:8]}"
        assert len(u) == 2, f"{task}: only one class present in this slice ({u}); AUROC undefined"
        self.norm = cache.cfg.norm

    def __len__(self):
        return len(self.labels)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """x: [E,F,T]. SpecAugment + electrode dropout. Train only."""
        c = self.tcfg
        if c is None:
            return x
        E, Fb, Tb = x.shape
        rng = np.random
        if c.aug_freq_mask:
            for _ in range(c.aug_freq_mask):
                w = rng.randint(0, max(2, Fb // 8))
                if w:
                    f0 = rng.randint(0, Fb - w)
                    x[:, f0:f0 + w, :] = 0.0
        if c.aug_time_mask:
            for _ in range(c.aug_time_mask):
                w = rng.randint(0, max(2, Tb // 6))
                if w:
                    t0 = rng.randint(0, Tb - w)
                    x[:, :, t0:t0 + w] = 0.0
        if c.aug_gauss_noise:
            x = x + rng.randn(*x.shape).astype(np.float32) * c.aug_gauss_noise
        return x

    def __getitem__(self, i):
        row = int(self.rows[i])
        x = np.asarray(self.cache.feats[row], dtype=np.float32)  # [E,F,T]

        if self.norm == "global_elec_freq":
            x = (x - self.cache.norm_mean) / np.maximum(self.cache.norm_std, 1e-6)
        elif self.norm == "per_sample":  # reproduces the old notebook, for ablation only
            flat = x.reshape(x.shape[0], -1)
            m = flat.mean(1)[:, None, None]
            s = np.maximum(flat.std(1), 1e-6)[:, None, None]
            x = (x - m) / s
        elif self.norm != "none":
            raise ValueError(self.norm)

        mask = np.ones(x.shape[0], dtype=bool)
        if self.train and self.tcfg is not None:
            x = self._augment(x)
            p = self.tcfg.aug_electrode_dropout
            if p > 0:
                drop = np.random.rand(x.shape[0]) < p
                if drop.all():
                    drop[np.random.randint(x.shape[0])] = False
                mask = ~drop

        return {
            "x_spec": torch.from_numpy(np.ascontiguousarray(x)),
            "coords": torch.from_numpy(np.ascontiguousarray(self.cache.coords)),
            "electrode_mask": torch.from_numpy(mask),
            "y": torch.tensor(float(self.labels[i]), dtype=torch.float32),
            "session": torch.tensor([self.cache.sid, self.cache.tid], dtype=torch.long),
        }


def collate(batch):
    """Pads the electrode axis. Coordinates of padded slots are zeroed AND masked."""
    max_e = max(b["x_spec"].shape[0] for b in batch)
    xs, cs, ms, ys, ss = [], [], [], [], []
    for b in batch:
        x, c, m = b["x_spec"], b["coords"], b["electrode_mask"]
        pad = max_e - x.shape[0]
        if pad > 0:
            x = torch.cat([x, torch.zeros((pad, *x.shape[1:]), dtype=x.dtype)], 0)
            c = torch.cat([c, torch.zeros((pad, c.shape[1]), dtype=c.dtype)], 0)
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)], 0)
        xs.append(x); cs.append(c); ms.append(m); ys.append(b["y"]); ss.append(b["session"])
    out = {
        "x_spec": torch.stack(xs), "coords": torch.stack(cs),
        "electrode_mask": torch.stack(ms), "y": torch.stack(ys), "session": torch.stack(ss),
    }
    assert out["electrode_mask"].any(1).all()
    return out


def make_loader(ds: CachedTaskDataset, tcfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=tcfg.batch_size, shuffle=shuffle, drop_last=False,
        num_workers=tcfg.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=tcfg.num_workers > 0, collate_fn=collate,
        prefetch_factor=4 if tcfg.num_workers > 0 else None,
    )


def val_test_ranges(cache: SessionCache, task: str):
    """Exactly neuroprobe's split of the evaluation session."""
    n = cache.n_samples(task)
    return (0, n // 2), (n // 2, n)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    MNI152 mm -> roughly unit scale, with a FIXED transform shared by all subjects.
    Never normalize coordinates per subject: that erases exactly the anatomical correspondence
    the harmonizer exists to exploit.
    """
    center = np.array([0.0, -18.0, 12.0], dtype=np.float32)  # approx MNI152 brain centroid
    return (coords - center) / 70.0
