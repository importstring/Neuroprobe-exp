"""
npx/train.py  --  interleaved multi-task training with per-task model selection.

Replaces the training loop cell of the old notebook.

Answering your question directly -- "am I doing the all-5-at-once thing right?":

No. Your loop was

    for epoch: for task in TASKS: for batch in train_loaders[task]: step()

That is not multi-task learning. That is sequential fine-tuning: the trunk gets pulled all the
way to task A, then all the way to task B, and by the time the epoch ends it has partially
forgotten A. The gradient for a shared trunk is only a multi-task gradient if the tasks are
mixed WITHIN the optimizer's window. Here a task is sampled per STEP, so consecutive updates
come from different tasks and the trunk actually has to find a shared representation.

And you had ONE checkpoint keyed on `combined_score`, the mean over tasks. So a run where
onset improved by 0.06 and the other five degraded by 0.01 each got saved as "best", and the
five degraded heads were what you would have submitted. Fixed: per-task best checkpoints, plus
the shared multi-task checkpoint, so at submission time you select per task on val.

On 5 vs 15: build the harness for 15 (the labels are already all in the cache, it costs nothing)
but ITERATE on the 4-task diagnostic panel. If you only iterate on the strong acoustic tasks you
will optimize yourself into the crowded lane where total remaining headroom is about +0.017.
The 10 tasks that every submission leaves at ~0.500 are worth +0.040 if you get them to 0.560
and +0.067 at 0.600 -- larger than the entire linear-to-SOTA spread on this board, and nobody
is working them.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .config import (
    ALL_TASKS, CKPT_ROOT, DEVICE, DIAGNOSTIC_PANEL, EVAL_SESSIONS, LOG_ROOT, SEED,
    TRAIN_SESSION, FeatureConfig, TrainConfig,
)
from .cache_io import CachedTaskDataset, SessionCache, collate, make_loader, val_test_ranges
from .model import MultiTaskModel, count_params


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Cycler:
    """Infinite shuffled iterator over a DataLoader."""

    def __init__(self, loader):
        self.loader, self.it = loader, iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


@torch.no_grad()
def auroc_on(model, loader, task, device=DEVICE):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        ps.append(model(b, task).float().cpu().numpy())
        ys.append(b["y"].cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def build_val_loaders(fcfg, tcfg, tasks, sessions=EVAL_SESSIONS):
    """Val = first half of each evaluation session. Test is never opened here."""
    out = {}
    for task in tasks:
        per_session = []
        for sid, tid in sessions:
            try:
                cache = SessionCache(fcfg, sid, tid)
                vr, _ = val_test_ranges(cache, task)
                ds = CachedTaskDataset(cache, task, sample_range=vr, train=False, tcfg=tcfg)
                per_session.append(((sid, tid), make_loader(ds, tcfg, shuffle=False)))
            except (FileNotFoundError, AssertionError) as e:
                print(f"  [val skip] btbank{sid}_{tid}/{task}: {e}")
        out[task] = per_session
    return out


def train(fcfg: FeatureConfig, tcfg: TrainConfig, use_gnn=False, gnn_k=6, gnn_layers=2,
          init_from: Path | None = None):
    seed_everything()
    run_dir = CKPT_ROOT / tcfg.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"feature": fcfg.hash(), "feature_spec": fcfg.__dict__, "train": tcfg.to_dict(),
         "use_gnn": use_gnn, "gnn_k": gnn_k, "gnn_layers": gnn_layers}, indent=2, default=str))

    tasks = tcfg.tasks
    sid, tid = TRAIN_SESSION
    train_cache = SessionCache(fcfg, sid, tid)
    train_loaders, cyclers, steps = {}, {}, 0
    for t in tasks:
        ds = CachedTaskDataset(train_cache, t, sample_range=None, train=True, tcfg=tcfg)
        train_loaders[t] = make_loader(ds, tcfg, shuffle=True)
        cyclers[t] = Cycler(train_loaders[t])
        steps += len(train_loaders[t])
        print(f"  train {t}: {len(ds)} samples, pos_rate={ds.labels.mean():.3f}")
    print(f"  steps/epoch = {steps}")

    val_loaders = build_val_loaders(fcfg, tcfg, tasks)

    model = MultiTaskModel(tcfg, tasks, use_gnn=use_gnn, gnn_k=gnn_k, gnn_layers=gnn_layers).to(DEVICE)
    if init_from is not None:
        sd = torch.load(init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        print(f"  init_from {init_from.name}: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"  params: {count_params(model):,}")

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.base_lr, weight_decay=tcfg.weight_decay)
    total_steps = steps * tcfg.num_epochs
    warm = steps * tcfg.warmup_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total_steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_per_task = {t: -np.inf for t in tasks}
    best_mean, best_mean_state, stale = -np.inf, None, 0
    rows, start_epoch = [], 1

    # ---- resume ----------------------------------------------------------------
    # A laptop running for weeks WILL be interrupted: Windows Update, a lid close, a
    # thermal shutdown. Every epoch writes last.pt so a reboot costs one epoch, not a run.
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_mean, best_per_task = ck["best_mean"], ck["best_per_task"]
        stale, rows = ck["stale"], ck["rows"]
        print(f"  RESUMED from epoch {ck['epoch']} (best_mean={best_mean:.4f})")

    for epoch in range(start_epoch, tcfg.num_epochs + 1):
        model.train()
        t0 = time.time()
        losses = {t: [] for t in tasks}
        order = [t for t in tasks for _ in range(max(1, steps // len(tasks)))]
        random.shuffle(order)
        acc = max(1, tcfg.accum_steps)
        opt.zero_grad(set_to_none=True)
        for k_step, task in enumerate(order):
            b = cyclers[task].next()
            b = {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(b, task)
                raw = crit(pred, b["y"])
                loss = model.weighted_loss(raw, task) / acc
            scaler.scale(loss).backward()
            # Gradient accumulation over `acc` micro-batches. Because a task is sampled per
            # micro-batch, an accumulated step mixes several tasks into ONE update -- which is
            # a stronger multi-task gradient than batch_size alone would give you, and it is
            # how you get an effective batch of 64 on a 6 GB laptop GPU.
            if (k_step + 1) % acc == 0 or k_step == len(order) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            losses[task].append(float(raw.detach()))

        # ---- validation: mean AUROC across the 10 evaluation sessions, per task ----
        per_task_auroc, detail = {}, {}
        for task in tasks:
            vals = []
            for (vsid, vtid), loader in val_loaders[task]:
                a = auroc_on(model, loader, task)
                detail[f"{task}|btbank{vsid}_{vtid}"] = a
                if not np.isnan(a):
                    vals.append(a)
            per_task_auroc[task] = float(np.mean(vals)) if vals else float("nan")

        mean_auroc = float(np.nanmean(list(per_task_auroc.values())))
        lr_now = opt.param_groups[0]["lr"]
        print(f"\nepoch {epoch:03d}  {time.time() - t0:.0f}s  lr={lr_now:.2e}  mean_val_auroc={mean_auroc:.4f}")
        for task in tasks:
            print(f"    {task:18s} loss={np.mean(losses[task]):.4f}  val_auroc={per_task_auroc[task]:.4f}"
                  f"  log_var={float(model.log_vars[task]):+.3f}")

        # per-task checkpoints -- this is the fix for the single-global-checkpoint bug
        for task in tasks:
            a = per_task_auroc[task]
            if not np.isnan(a) and a > best_per_task[task]:
                best_per_task[task] = a
                torch.save({"epoch": epoch, "task": task, "val_auroc": a,
                            "model_state_dict": model.state_dict(),
                            "config": tcfg.to_dict(), "feature_hash": fcfg.hash()},
                           run_dir / f"best_{task}.pt")

        if mean_auroc > best_mean:
            best_mean, stale = mean_auroc, 0
            best_mean_state = copy.deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "mean_val_auroc": best_mean,
                        "model_state_dict": best_mean_state, "per_task": per_task_auroc,
                        "config": tcfg.to_dict(), "feature_hash": fcfg.hash()},
                       run_dir / "best_mean.pt")
        else:
            stale += 1

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_mean": best_mean, "best_per_task": best_per_task,
                    "stale": stale, "rows": rows, "feature_hash": fcfg.hash(),
                    "config": tcfg.to_dict()}, run_dir / "last.pt")

        rows.append({"epoch": epoch, "mean_val_auroc": mean_auroc, "lr": lr_now,
                     **{f"auroc_{t}": per_task_auroc[t] for t in tasks},
                     **{f"loss_{t}": float(np.mean(losses[t])) for t in tasks}})
        with open(run_dir / "history.json", "w") as f:
            json.dump({"rows": rows, "last_detail": detail}, f, indent=2)

        # onset is the canary. If it is below 0.65 by epoch 10 something upstream is broken
        # and there is no point burning the rest of the budget.
        if "onset" in per_task_auroc and epoch >= 10 and per_task_auroc["onset"] < 0.65:
            print("  !! CANARY FAILED: onset < 0.65 at epoch 10. Aborting -- debug the pipeline, "
                  "do not tune hyperparameters.")
            break
        if stale >= tcfg.patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_mean_state is not None:
        model.load_state_dict(best_mean_state)

    _append_runs_csv(tcfg, fcfg, use_gnn, best_mean, best_per_task)
    print(f"\nbest mean val auroc = {best_mean:.4f}")
    print("best per task:", {k: round(v, 4) for k, v in best_per_task.items()})
    return model, best_per_task, best_mean


def _append_runs_csv(tcfg, fcfg, use_gnn, best_mean, best_per_task):
    """
    Every run lands in one CSV keyed by config hash. If a result is not in this file it did not
    happen. Four weeks of experiments with no run log is four weeks of anecdotes.
    """
    path = LOG_ROOT / "runs.csv"
    row = {"run_tag": tcfg.run_tag, "feature_hash": fcfg.hash(), "train_hash": tcfg.hash(),
           "use_gnn": use_gnn, "best_mean_val_auroc": round(best_mean, 5),
           "tasks": "|".join(tcfg.tasks),
           "per_task": json.dumps({k: round(v, 5) for k, v in best_per_task.items()}),
           "reref": fcfg.reref, "align": fcfg.align, "norm": fcfg.norm,
           "n_fft": fcfg.n_fft, "hop": fcfg.hop, "freq_max": fcfg.freq_max_hz,
           "pool": tcfg.pool, "V": tcfg.num_virtual_sensors, "lr": tcfg.base_lr,
           "bs": tcfg.batch_size, "wd": tcfg.weight_decay, "dropout": tcfg.dropout,
           "coords_in_values": tcfg.use_coords_in_values, "subj_emb": tcfg.use_subject_embedding,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--tasks", nargs="+", default=DIAGNOSTIC_PANEL,
                    help="'panel' | 'all' | explicit task names")
    ap.add_argument("--gnn", action="store_true")
    ap.add_argument("--gnn-k", type=int, default=6)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--virtual-sensors", type=int, default=16)
    ap.add_argument("--pool", default="flatten", choices=["flatten", "attn", "mean"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1,
                    help="Gradient accumulation. Effective batch = batch-size * accum.")
    ap.add_argument("--elec-dropout", type=float, default=0.2)
    ap.add_argument("--coords-in-values", type=int, default=1)
    ap.add_argument("--subject-embedding", type=int, default=0)
    ap.add_argument("--norm", default="global_elec_freq")
    ap.add_argument("--reref", default="shaft")
    ap.add_argument("--align", default="ea")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--freq-max", type=float, default=150.0)
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    tasks = ALL_TASKS if args.tasks == ["all"] else (
        DIAGNOSTIC_PANEL if args.tasks == ["panel"] else args.tasks)

    fcfg = FeatureConfig(reref=args.reref, align=args.align, n_fft=args.n_fft, hop=args.hop,
                         win_len=args.n_fft, freq_max_hz=args.freq_max, norm=args.norm)
    tcfg = TrainConfig(
        run_tag=args.run_tag, tasks=tasks, batch_size=args.batch_size, base_lr=args.lr,
        weight_decay=args.wd, num_epochs=args.epochs, dropout=args.dropout,
        num_virtual_sensors=args.virtual_sensors, pool=args.pool, num_workers=args.workers,
        accum_steps=args.accum, aug_electrode_dropout=args.elec_dropout,
        use_coords_in_values=bool(args.coords_in_values),
        use_subject_embedding=bool(args.subject_embedding),
    )
    train(fcfg, tcfg, use_gnn=args.gnn, gnn_k=args.gnn_k, gnn_layers=args.gnn_layers,
          init_from=Path(args.init_from) if args.init_from else None)


if __name__ == "__main__":
    main()
