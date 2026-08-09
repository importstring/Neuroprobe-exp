# npx — Neuroprobe cross-subject pipeline

Your notebook, rewritten as a package. Verified: 10/10 self-checks pass and the full
gate → train → eval → submission-scaffold chain runs end to end on synthetic data
(`python scripts/99_synthetic_smoke.py`).

---

## What every part of the old notebook became

| Old notebook | New home | What changed |
|---|---|---|
| Cell 1: paths, `RUN_TAG`, `CHECKPOINT_PATH` | `npx/config.py` | Paths + a hashed `FeatureConfig`. Changing `n_fft` now creates a new cache directory, so you can never train on features built with a different spec. |
| Cell 2: imports, `TASK_PRESETS`, all the `UPPER_CASE` globals | `npx/config.py` | Globals became two dataclasses: `FeatureConfig` (invalidates the cache) and `TrainConfig` (doesn't). `TASK_PRESETS` collapsed to `ALL_TASKS` + `DIAGNOSTIC_PANEL`. |
| `BrainTreebankSubject` construction, `build_all_subjects_dict`, `ALL_SUBJECTS` | `npx/cache_build.py` | Subjects are loaded one at a time during the cache build and released. You were holding 10 subject objects in RAM for the whole session. |
| `build_laplacian_matrix`, `build_laplacian_matrix_cached`, `_LAPLACIAN_CACHE` | `npx/features.py::build_shaft_laplacian` | Rebuilt from electrode **labels** (stem ±1 on the same shaft), not kNN over coordinates. Your kNN version was connecting electrodes on different shafts in different lobes. The old version survives as `build_knn_laplacian` so you can ablate it. |
| `_spectrogram`, `estimate_train_spec_stats` (dead — it raised) | `npx/features.py::stft_features` + `estimate_global_stats` | Real STFT spec (512/128, cropped 0–150 Hz) and global per-(electrode, freq-bin) statistics that are actually computed, not a dead function behind `USE_GLOBAL_TRAIN_NORM=False`. |
| — | `npx/features.py::estimate_ea_matrix` | New. Euclidean Alignment per session: `X ← R^(−1/2) X`. Highest expected value single change in the pipeline. |
| `CrossSubjectLaplacianSpectrogramDataset` | `npx/cache_build.py` (compute) + `npx/cache_io.py::CachedTaskDataset` (read) | Split in two. Features are computed once into a float16 memmap; the Dataset now only reads bytes, normalizes, and augments. |
| `unpack_base_item`, `infer_subject_id`, `infer_electrode_coordinates`, `get_fallback_coords` | **deleted** | All four existed to guess at data you can just read once at build time. `infer_electrode_coordinates`'s silent fallback was a live corruption bug (see below). |
| `collate_cross_subject_batch` | `npx/cache_io.py::collate` | Same idea, plus padded coordinates are zeroed and the mask contract is asserted. |
| `ResidualConvBlock` | `npx/model.py` | BatchNorm → GroupNorm. BatchNorm over the flattened `B*E` axis makes normalization subject-specific, which is a domain-shift leak on a cross-subject benchmark. |
| `AdaptiveAvgPool2d((1,1))` | `npx/model.py::ElectrodeTrunk` | Pools time to 1, keeps 4 frequency bands. |
| `VirtualSensorHarmonizer` | `npx/model.py` | `use_coords_in_values=True`; orthogonal query init instead of `randn * 0.02`. |
| `LeaderboardSpectrogramEncoder` | `npx/model.py::Encoder` | `pooled = sensor_tokens.mean(1)` → `flatten(1)`. Subject embedding removed and now asserted absent. Optional `AnatomicalGNN`. |
| `LeaderboardMultiTaskModel` | `npx/model.py::MultiTaskModel` | Same shape; `weighted_loss` uncertainty term corrected to `exp(−logvar)·L + 0.5·logvar`. |
| The training loop cell | `npx/train.py` | Tasks interleaved per **step**, not per epoch. Cosine schedule + warmup. AMP. Per-task checkpoints. Every run appends to `logs/runs.csv`. Onset canary aborts a doomed run at epoch 10. |
| `evaluate_task`, `combined_score` | `npx/train.py::auroc_on` + `npx/evaluate.py` | Val is now the mean over all 10 evaluation sessions, not one. `combined_score` is gone — it was the reason one improving task could carry five degrading ones into your "best" checkpoint. |
| The "Final TEST evaluation" cell | `npx/evaluate.py::sweep` | Full 10 × 15 sweep, per-task checkpoint selection on val, and a submission scaffold in the exact JSON schema the repo's format tests expect. |
| Permutation test + batch-contract asserts (the good part) | `npx/selfcheck.py` | Kept and extended to 10 checks, including mask-invariance and no-subject-leakage. |
| — | `npx/baseline_linear.py` | New. The E0 gate. |

---

## Run order

```bash
pip install -r requirements.txt
python -m npx.selfcheck                      # 10 checks, ~5 s, no data needed
python scripts/99_synthetic_smoke.py         # whole chain on fake data, ~5 min

# real data
python scripts/01_build_cache.py --sessions train eval        # ~6 GB, hours
python scripts/00_gate_linear.py --tasks all                  # THE GATE
python scripts/02_train.py --run-tag e2_panel --tasks panel
python scripts/03_eval_all.py --run-tag e2_panel --tasks all --no-test
```

`--no-test` on `03_eval_all` reports val only and leaves the test half untouched. Use it for
everything except the final submission run.

---

## The five blocking bugs, restated

**C1 — `use_subject_embedding=True` on a cross-subject benchmark.** Test subjects 1, 3, 4, 7 and
10 never appear in training, so their embedding rows are untrained initialization noise, and
`.clamp(0, num_embeddings-1)` silently mapped them onto some other subject's learned vector.
During training it is also a free channel for memorizing the single training subject.
`selfcheck.py::test_no_subject_leakage` now fails if any subject-conditioned parameter exists.

**C2 — `infer_electrode_coordinates`'s silent fallback.** On any metadata-key mismatch it
returned another subject's coordinates, and your only guard was
`coords.shape[0] != x.shape[0]`. Subjects 1, 4, 5, 6, 8 and 10 all have exactly 120 Lite
electrodes, so the guard could never fire — you would have paired subject 10's signals with
subject 1's coordinate frame, on a benchmark whose entire difficulty is coordinate
harmonization, with every assert green. Deleted; coordinates now live in the session's own cache
directory and `allow_missing_coordinates=False`.

**C3 — `AdaptiveAvgPool2d((1,1))`.** Averaged the entire spectrogram to one scalar per channel.
The #1 entry is called "CNN Laplacian rereferencing spectrogram" — the frequency axis *is* the
signal.

**C4 — `pooled = sensor_tokens.mean(dim=1)`.** You built 16 virtual sensors with persistent
learned identities, then averaged them and discarded which one fired. A signal localized in two
occipital slots got diluted 8×. Plausibly why your visual tasks sat at chance.

**C5 — per-sample z-scoring.** Removes exactly the information the classifier needs: absolute
band power. On this leaderboard the *identical* architecture scored 0.522 with per-window-z and
0.547 with global-z. You were paying 0.025 for free.

Also worth knowing: `COORDINATE_SYSTEM="cortical"` is an atlas *projection* — depth electrodes
get flattened onto the surface, losing the depth axis, and neuroprobe returns NaN rather than
raising for electrodes it cannot project. `mni152` preserves true 3D position and raises. Default
is now `mni152`.

---

## Experiment ladder

Each rung is one `--run-tag`. Never change two rungs at once. `+0.01` on the 15-task mean is
noise-adjacent (SE ≈ 0.004); `+0.03` is solid; `+0.05` wins.

| Rung | Command | Gate to pass |
|---|---|---|
| **E0** | `00_gate_linear.py --tasks all` | overall = 0.539 ± 0.015. **If this fails, stop and debug. Do not tune.** |
| **E1** | `02_train.py --run-tag e1_panel --tasks panel` | onset ≥ 0.70, speech ≥ 0.68 on val |
| **E2** | `--run-tag e2_norm_ablate --norm per_sample` | must be *worse* than E1. If it isn't, your global stats are broken. |
| **E3** | `--run-tag e3_no_ea --align none` (rebuild cache first) | E1 − E3 = the value of Euclidean Alignment. Expect +0.01 to +0.03. |
| **E4** | `--run-tag e4_knn --reref knn` / `--reref none` | confirms the shaft Laplacian is earning its place |
| **E5** | `--run-tag e5_V32 --virtual-sensors 32`, then 64 | pick the best V on val, then freeze it |
| **E6** | `--run-tag e6_pool_attn --pool attn` | vs `flatten`; also run `--pool mean` once to measure what C4 cost you |
| **E7** | `--run-tag e7_gnn --gnn --gnn-k 6 --gnn-layers 2` | this is the GNN bet. Sweep k ∈ {4, 6, 10}, layers ∈ {1, 2, 3}. |
| **E8** | `--run-tag e8_all15 --tasks all --epochs 80` | first full-15 run; read the per-task table, not the mean |
| **E9** | per-task specialists: `--tasks pitch frame_brightness` initialized from E8's `best_mean.pt` via `--init-from` | Route B. This is where the +0.04 lives. |
| **E10** | ensemble: average sigmoid outputs of your best 3–5 checkpoints per task, selected on val | usually +0.005 to +0.015 for free |

**Decision tree at E7.** If the GNN gains ≥ +0.015 on the panel, sweep it hard and make it the
spine. If it lands within ±0.01, it is noise — drop it and spend the time on E9 instead. If it
*loses* more than 0.015, the graph is wrong before the architecture is: check that coordinates
are normalized with the shared fixed transform in `cache_io.normalize_coords` and not per
subject.

**Decision tree at E9.** If pitch or frame_brightness clears 0.55, that is the largest
unclaimed value on the board and everything else should stop. If both stay under 0.52 after
per-task specialization, accept that the visual and prosodic tasks are genuinely near-unreachable
from 1 s of iEEG in a 120-electrode Lite montage, and redirect to squeezing onset/speech/volume.

---

## Route A vs Route B

The top entry is 0.578, made of onset 0.780, speech 0.751, delta_volume 0.662, word_index 0.617,
volume 0.587 — and **ten tasks at or below 0.555**, most of them essentially at chance.

- **Route A** (push onset/speech higher): about +0.017 of headroom, contested by three funded labs.
- **Route B** (lift the ten dead tasks): 0.560 on each is worth **+0.040**; 0.600 is worth
  **+0.067**. That is larger than the entire linear-to-SOTA spread on this leaderboard, and as far
  as the submission history shows, nobody is working it.

Target 0.63–0.66 overall. That is a "large margin" win and it comes almost entirely from Route B.

---

## Standing rules

1. **Never read `test` during development.** Use `--no-test`. Test is opened once, for the
   submission run.
2. **Do not fit weights on `val`.** Selecting a checkpoint on val is legal — it is the labelled
   half neuroprobe hands you — but declare it in `metadata.json`.
3. **Every run goes through `02_train.py`** so `logs/runs.csv` stays complete. If a result is not
   in that file, it did not happen.
4. **`onset` is the canary.** Below 0.65 by epoch 10 means a pipeline bug, not a bad
   hyperparameter. Training aborts automatically.
5. **+0.01 or it didn't happen.**
6. **Ablate everything.** E2, E3, E4 and the `--pool mean` run exist so that when you write
   `PUBLICATION.bib` you can state which changes earned their keep. The submission requires a
   publication anyway.

---

## Compute

Every cross-subject test subject (1, 3, 4, 7, 10) has legally usable **unlabelled** data under
SUBMIT.md's pretraining allowance: `btbank1_0, btbank3_2, btbank4_2, btbank7_100/101/102,
btbank10_100/101`. That is the strongest unexploited lever on the board after Route B — masked
spectrogram-patch pretraining on `--sessions pretrain` gives the harmonizer a representation of
each test subject's montage before it ever sees a label. Build that cache in week 2 and let it run
while you work.

One thing to settle before committing to a schedule: **is your 16 GB system RAM on a CPU-only
machine, or GPU VRAM?** The synthetic smoke test above took ~100 s per epoch on 40 electrodes and
300 samples on CPU. Your real setup is 120 electrodes and ~3,500 samples per session — roughly
100× that work. CPU-only, a single full-15 run is weeks, and the ladder above does not fit in four
months. If there is no CUDA GPU, add Kaggle's free 30 GPU-hours/week (~480 hours over four
months): build the cache locally, upload it as a Kaggle dataset, and train there. The package is
already written so that only `npx/config.py`'s two paths need to change.
