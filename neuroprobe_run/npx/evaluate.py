"""
npx/evaluate.py  --  full 10-session x 15-task sweep + leaderboard-format submission writer.

Replaces the "Final TEST evaluation" cell, which only evaluated the 6 tasks in TASKS on a single
session and would not have produced a submittable artifact.

The output JSON schema below was read directly from the current #1 entry
(leaderboard/CNN_Laplacian_rereferencing_spectrogram_Geeling_Chau_21_04_2026/Cross-Subject/
population_onset.json), so it matches what the repo's automated format tests expect:

    {model_name, description, author, organization, organization_url, timestamp,
     evaluation_results: {"btbank<S>_<T>": {"population": {"one_second_after_onset":
        {time_bin_start, time_bin_end, folds: [{train_accuracy, train_roc_auc, val_accuracy,
         val_roc_auc, test_accuracy, test_roc_auc, fold_idx}]}}}}}

Per-task checkpoint selection: for each task, load whichever checkpoint had the best VAL AUROC
for that task (best_<task>.pt from training, or best_mean.pt). Selecting per task on val is
legitimate -- val is the labelled first half of the evaluation session that neuroprobe itself
hands you. Test is read exactly once, at the end, to fill in test_roc_auc. Never look at it
during development.

Run:
    python -m scripts.03_eval_all --run-tag e5_gnn --tasks all --write-submission
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from .cache_io import CachedTaskDataset, SessionCache, make_loader, val_test_ranges
from .config import (
    ALL_TASKS, CKPT_ROOT, DEVICE, EVAL_SESSIONS, LOG_ROOT, SUBMIT_ROOT, TRAIN_SESSION,
    FeatureConfig, TrainConfig,
)
from .model import MultiTaskModel


@torch.no_grad()
def predict(model, loader, task):
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}
        ps.append(torch.sigmoid(model(b, task)).float().cpu().numpy())
        ys.append(b["y"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def load_model_for_task(run_dir: Path, task: str, fcfg: FeatureConfig):
    for name in (f"best_{task}.pt", "best_mean.pt"):
        p = run_dir / name
        if p.exists():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if ck.get("feature_hash") not in (None, fcfg.hash()):
                raise RuntimeError(
                    f"{p.name} was trained on feature cache {ck['feature_hash']} but you are "
                    f"evaluating with {fcfg.hash()}. Refusing to mix feature specs.")
            cfgd = ck["config"]
            tcfg = TrainConfig(**{k: v for k, v in cfgd.items() if k in TrainConfig().__dict__})
            model = MultiTaskModel(tcfg, tcfg.tasks,
                                   use_gnn=json.loads((run_dir / "config.json").read_text()).get("use_gnn", False))
            model.load_state_dict(ck["model_state_dict"])
            return model.to(DEVICE), tcfg, name
    raise FileNotFoundError(f"no checkpoint for {task} in {run_dir}")


def sweep(run_tag: str, fcfg: FeatureConfig, tasks, write_submission=False, meta=None,
          read_test=True):
    run_dir = CKPT_ROOT / run_tag
    results = {}
    summary = {}

    for task in tasks:
        try:
            model, tcfg, src = load_model_for_task(run_dir, task, fcfg)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [skip] {task}: {e}")
            continue
        if task not in model.task_to_id:
            print(f"  [skip] {task}: head not trained in this run")
            continue

        # training-session metrics, reported for the submission file
        tr_cache = SessionCache(fcfg, *TRAIN_SESSION)
        tr_ds = CachedTaskDataset(tr_cache, task, None, train=False, tcfg=tcfg)
        ytr, ptr = predict(model, make_loader(tr_ds, tcfg, False), task)
        tr_m = metrics(ytr, ptr)

        per_session = {}
        for sid, tid in EVAL_SESSIONS:
            key = f"btbank{sid}_{tid}"
            try:
                cache = SessionCache(fcfg, sid, tid)
                vr, ter = val_test_ranges(cache, task)
            except (FileNotFoundError, AssertionError) as e:
                print(f"  [skip] {key}/{task}: {e}")
                continue
            yv, pv = predict(model, make_loader(
                CachedTaskDataset(cache, task, vr, train=False, tcfg=tcfg), tcfg, False), task)
            v_m = metrics(yv, pv)
            if read_test:
                yt, pt = predict(model, make_loader(
                    CachedTaskDataset(cache, task, ter, train=False, tcfg=tcfg), tcfg, False), task)
                t_m = metrics(yt, pt)
            else:
                t_m = {"accuracy": float("nan"), "roc_auc": float("nan")}
            per_session[key] = {
                "population": {"one_second_after_onset": {
                    "time_bin_start": 0.0, "time_bin_end": 1.0,
                    "folds": [{
                        "train_accuracy": tr_m["accuracy"], "train_roc_auc": tr_m["roc_auc"],
                        "val_accuracy": v_m["accuracy"], "val_roc_auc": v_m["roc_auc"],
                        "test_accuracy": t_m["accuracy"], "test_roc_auc": t_m["roc_auc"],
                        "fold_idx": 0,
                    }]}}}
        results[task] = per_session
        vals = [v["population"]["one_second_after_onset"]["folds"][0]["val_roc_auc"]
                for v in per_session.values()]
        tests = [v["population"]["one_second_after_onset"]["folds"][0]["test_roc_auc"]
                 for v in per_session.values()]
        summary[task] = {"checkpoint": src, "n_sessions": len(per_session),
                         "val_auroc": float(np.nanmean(vals)) if vals else float("nan"),
                         "test_auroc": float(np.nanmean(tests)) if tests else float("nan")}
        print(f"  {task:18s} val={summary[task]['val_auroc']:.4f}  "
              f"test={summary[task]['test_auroc']:.4f}  ({src})")

    overall_val = float(np.nanmean([s["val_auroc"] for s in summary.values()])) if summary else float("nan")
    overall_test = float(np.nanmean([s["test_auroc"] for s in summary.values()])) if summary else float("nan")
    print(f"\n=== {run_tag} ===")
    print(f"overall val  AUROC = {overall_val:.4f}   ({len(summary)}/15 tasks)")
    print(f"overall test AUROC = {overall_test:.4f}")
    print("reference: #1 = 0.578, #2 = 0.575, linear = 0.539")
    print("standard error on a 15-task/10-session mean is about 0.004, so treat anything under "
          "+0.01 as noise.")
    (LOG_ROOT / f"sweep_{run_tag}.json").write_text(json.dumps(
        {"run_tag": run_tag, "overall_val": overall_val, "overall_test": overall_test,
         "summary": summary}, indent=2))

    if write_submission:
        write_submission_dir(run_tag, results, meta or {})
    return summary, overall_val, overall_test


def write_submission_dir(run_tag: str, results: dict, meta: dict):
    """
    Emits leaderboard/<MODEL>_<First>_<Last>_<DD>_<MM>_<YYYY>/Cross-Subject/population_*.json
    plus metadata.json. You still have to write PUBLICATION.bib (mandatory -- a tech report or
    arXiv preprint counts) and sign ATTESTATION.txt by hand, then open the PR.
    """
    model_name = meta.get("model_name", run_tag)
    author = meta.get("author", "Simon Bergeron")
    day = time.strftime("%d_%m_%Y")
    slug = f"{model_name}_{author.replace(' ', '_')}_{day}"
    base = SUBMIT_ROOT / "leaderboard" / slug
    (base / "Cross-Subject").mkdir(parents=True, exist_ok=True)

    header = {
        "model_name": model_name,
        "description": meta.get("description", f"{model_name} cross-subject submission."),
        "author": author,
        "organization": meta.get("organization", ""),
        "organization_url": meta.get("organization_url", ""),
        "timestamp": int(time.time()),
    }
    for task, per_session in results.items():
        (base / "Cross-Subject" / f"population_{task}.json").write_text(json.dumps(
            {**header, "evaluation_results": per_session}, indent=2))
    (base / "metadata.json").write_text(json.dumps(header, indent=2))
    missing = sorted(set(ALL_TASKS) - set(results.keys()))
    (base / "README_BEFORE_PR.txt").write_text(
        "Still required before opening the PR:\n"
        "  1. PUBLICATION.bib  (MANDATORY -- an arXiv preprint or tech report is acceptable)\n"
        "  2. ATTESTATION.txt  (signed; declare that no forbidden session was used)\n"
        "  3. Run the repo's format tests locally.\n"
        f"  4. Missing task files ({len(missing)}): {missing}\n"
        "  5. If you used val for per-task checkpoint selection, say so explicitly in\n"
        "     metadata.json's description. It is legal but you want it on the record.\n")
    print(f"\nsubmission scaffold written to {base}")
    if missing:
        print(f"  WARNING: {len(missing)} of 15 task files missing: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--write-submission", action="store_true")
    ap.add_argument("--no-test", action="store_true",
                    help="Development mode: report val only and leave test untouched.")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--author", default="Simon Bergeron")
    ap.add_argument("--organization", default="")
    ap.add_argument("--reref", default="shaft")
    ap.add_argument("--align", default="ea")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    a = ap.parse_args()
    tasks = ALL_TASKS if a.tasks == ["all"] else a.tasks
    fcfg = FeatureConfig(reref=a.reref, align=a.align, n_fft=a.n_fft, hop=a.hop, win_len=a.n_fft)
    sweep(a.run_tag, fcfg, tasks, write_submission=a.write_submission,
          meta={"model_name": a.model_name or a.run_tag, "author": a.author,
                "organization": a.organization},
          read_test=not a.no_test)


if __name__ == "__main__":
    main()
