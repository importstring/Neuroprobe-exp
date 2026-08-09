"""
npx/config.py  --  single source of truth for paths, feature spec, and hyperparameters.

Replaces the first two cells of the old notebook.

Everything that changes the CONTENT of the feature cache lives in FeatureConfig and is
hashed into the cache directory name. If you change n_fft, you get a new cache -- you can
never accidentally train on features built with a different spec. Everything that only
changes TRAINING lives in TrainConfig and does not invalidate the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths.  Edit these two lines and nothing else.
# ---------------------------------------------------------------------------
ROOT_DIR_BRAINTREEBANK = Path(
    os.environ.get(
        "ROOT_DIR_BRAINTREEBANK",
        r"C:\Users\simon\PyCharmMiscProject\neuroprobe-dev\braintreebank",
    )
)
WORK_DIR = Path(os.environ.get("NPX_WORK_DIR", r"C:\Users\simon\PyCharmMiscProject\neuroprobe-dev\npx_work"))

# neuroprobe reads this env var at import time, so it must be set before `import neuroprobe`.
os.environ["ROOT_DIR_BRAINTREEBANK"] = str(ROOT_DIR_BRAINTREEBANK)

CACHE_ROOT = WORK_DIR / "cache"       # feature memmaps
CKPT_ROOT = WORK_DIR / "checkpoints"  # per-run, per-task checkpoints
LOG_ROOT = WORK_DIR / "logs"          # runs.csv + per-run json
SUBMIT_ROOT = WORK_DIR / "submissions"

for _p in (CACHE_ROOT, CKPT_ROOT, LOG_ROOT, SUBMIT_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 2048  # neuroprobe.config.SAMPLING_RATE; asserted at cache build time
WINDOW_SAMPLES = 2048  # 1.0 s from word onset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ---------------------------------------------------------------------------
# Benchmark constants (verified against neuroprobe/train_test_splits.py)
# ---------------------------------------------------------------------------
ALL_TASKS = [
    "onset", "speech", "volume", "delta_volume", "pitch",
    "word_index", "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
]

# The 4-task panel you iterate on. Two canaries with known signal, two known-dead tasks.
# If onset drops below 0.70 you have broken something. If pitch/frame_brightness move,
# you have found something nobody else on the leaderboard has.
DIAGNOSTIC_PANEL = ["onset", "speech", "pitch", "frame_brightness"]

# The single labelled training session the cross-subject split gives you.
TRAIN_SESSION = (2, 4)

# The 10 cross-subject evaluation sessions (subject_id, trial_id).
EVAL_SESSIONS = [
    (1, 1), (1, 2),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1),
]

# Unlabelled sessions legal for self-supervised pretraining, per SUBMIT.md.
# Note: every cross-subject test subject (1,3,4,7,10) appears here.
PRETRAIN_SESSIONS = [
    (1, 0), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (3, 2), (4, 2),
    (5, 0), (6, 0), (6, 1), (6, 4), (8, 0), (9, 0),
    (7, 100), (7, 101), (7, 102), (10, 100), (10, 101),
]


# ---------------------------------------------------------------------------
# Feature spec -- hashed into the cache path
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    # Re-referencing. "shaft" = BrainBERT/Chau-style Laplacian using electrode LABELS
    # (stem +/- 1 on the same depth shaft). "knn" = the old coordinate-kNN version, kept
    # only so you can ablate it and watch it lose. "none" = raw.
    reref: str = "shaft"

    # Domain alignment applied per session after re-referencing.
    # "ea" = Euclidean Alignment: X <- R^{-1/2} X where R is the session's mean covariance.
    # This is the single highest-expected-value change in the whole pipeline.
    align: str = "ea"

    # STFT. 512/128 at 2048 Hz gives 4 Hz resolution and T=17 frames over 1 s.
    n_fft: int = 512
    hop: int = 128
    win_len: int = 512
    freq_max_hz: float = 150.0     # crop; 0-150 Hz -> 38 bins at 4 Hz
    log1p: bool = True

    # Normalization. "global_elec_freq" = per (electrode, freq-bin) z-score using stats
    # from the whole session. This is the global-z that bought +0.025 on the leaderboard.
    # "per_sample" reproduces the old notebook's bug so you can ablate it.
    norm: str = "global_elec_freq"

    lite: bool = True
    nano: bool = False
    dtype: str = "float16"          # cache storage dtype

    def n_freq_bins(self) -> int:
        onesided = self.n_fft // 2 + 1
        hz_per_bin = SAMPLING_RATE / self.n_fft
        return min(onesided, int(self.freq_max_hz / hz_per_bin) + 1)

    def n_time_bins(self) -> int:
        # torch.stft with center=True
        return WINDOW_SAMPLES // self.hop + 1

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def cache_dir(self) -> Path:
        d = CACHE_ROOT / f"feat_{self.hash()}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.json").write_text(json.dumps(asdict(self), indent=2))
        return d


@dataclass
class TrainConfig:
    run_tag: str = "e2_baseline"
    tasks: list = field(default_factory=lambda: list(DIAGNOSTIC_PANEL))

    batch_size: int = 64          # was 8. Features are precomputed; 8 was throttling you.
    base_lr: float = 3e-4
    weight_decay: float = 1e-2
    num_epochs: int = 60
    patience: int = 10
    num_workers: int = 4          # now legal: this is a script, not a notebook
    grad_clip: float = 1.0
    accum_steps: int = 1          # effective batch = batch_size * accum_steps
    warmup_epochs: int = 3
    label_smoothing: float = 0.0

    dropout: float = 0.2
    attn_dropout: float = 0.1

    # encoder dims
    trunk_out_dim: int = 96
    elec_hidden_dim: int = 128
    coord_emb_dim: int = 32
    model_dim: int = 128
    task_emb_dim: int = 8

    # harmonizer
    num_virtual_sensors: int = 16
    num_sensor_heads: int = 4
    use_coords_in_keys: bool = True
    use_coords_in_values: bool = True   # was False. Coords are the only cross-subject anchor.
    use_sensor_self_attn: bool = True
    pool: str = "flatten"               # was "mean" -- "mean" destroys slot identity

    # conditioning
    use_subject_embedding: bool = False  # MUST be False for cross-subject. Non-negotiable.
    use_task_embedding: bool = True

    # augmentation
    aug_electrode_dropout: float = 0.2   # drop 20% of electrodes per sample
    aug_time_mask: int = 2               # SpecAugment-style time masks
    aug_freq_mask: int = 2
    aug_gauss_noise: float = 0.05

    # multi-task
    interleave: bool = True              # sample task per STEP, not per epoch
    loss_weighting: str = "uncertainty"  # "uncertainty" | "uniform"

    coordinate_system: str = "mni152"   # was "cortical". mni152 keeps depth + raises on missing.

    def to_dict(self):
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
