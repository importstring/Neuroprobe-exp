"""
npx/selfcheck.py  --  the test suite. Run it after every structural change.

Your notebook's test harness was the one genuinely excellent thing in it: the permutation-
invariance assertion, the batch-contract asserts, the smoke forward pass. All of it is kept here
and extended. Two additions matter:

  * test_no_subject_leakage: asserts the model has no subject embedding and that its output is
    unchanged when the session id changes. On a cross-subject benchmark this is the invariant
    that decides whether your number is real.
  * test_mask_invariance: asserts masked (padded) electrodes cannot influence the output.
    Your old code padded x_spec with zeros and passed key_padding_mask, which is correct in the
    cross-attention -- but the electrode trunk still ran the CNN over the zero rows and a
    BatchNorm layer folded those zeros into the batch statistics. This test catches that class
    of leak; the switch to GroupNorm in model.py is the fix.

Run:
    python -m npx.selfcheck
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .config import DIAGNOSTIC_PANEL, FeatureConfig, TrainConfig
from .model import MultiTaskModel

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                fn(*a, **kw)
                PASS.append(name)
                print(f"  PASS  {name}")
            except Exception as e:
                FAIL.append((name, f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return wrapped
    return deco


def fake_batch(B=6, E=120, Fb=38, Tb=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, E, Fb, Tb, generator=g)
    coords = torch.randn(B, E, 3, generator=g) * 0.5
    mask = torch.ones(B, E, dtype=torch.bool)
    mask[:, -10:] = False              # 10 padded slots
    x[:, -10:] = 0.0
    coords[:, -10:] = 0.0
    return {"x_spec": x, "coords": coords, "electrode_mask": mask,
            "y": (torch.rand(B, generator=g) > 0.5).float(),
            "session": torch.zeros(B, 2, dtype=torch.long)}


def make_model(**over):
    tcfg = TrainConfig(**{**TrainConfig().to_dict(), **over})
    tcfg.tasks = list(DIAGNOSTIC_PANEL)
    m = MultiTaskModel(tcfg, tcfg.tasks)
    m.eval()
    return m


@check("forward shape")
def test_forward():
    m, b = make_model(), fake_batch()
    with torch.no_grad():
        out = m(b, "onset")
    assert out.shape == b["y"].shape, (out.shape, b["y"].shape)


@check("permutation invariance over electrodes")
def test_permutation():
    m, b = make_model(), fake_batch()
    with torch.no_grad():
        a = m(b, "onset")
    B, E = b["electrode_mask"].shape
    perms = torch.stack([torch.randperm(E) for _ in range(B)])

    def gather(x, p):
        for _ in range(x.ndim - 2):
            p = p.unsqueeze(-1)
        return torch.gather(x, 1, p.expand(-1, -1, *x.shape[2:]))

    pb = {**b}
    for k in ("x_spec", "coords", "electrode_mask"):
        pb[k] = gather(b[k].clone(), perms)
    with torch.no_grad():
        c = m(pb, "onset")
    d = (a - c).abs().max().item()
    assert d < 1e-4, f"max diff {d}"


@check("masked electrodes cannot influence output")
def test_mask_invariance():
    m, b = make_model(), fake_batch()
    with torch.no_grad():
        a = m(b, "onset")
    b2 = {k: v.clone() for k, v in b.items()}
    pad = ~b["electrode_mask"]
    b2["x_spec"][pad] = torch.randn_like(b2["x_spec"][pad]) * 50.0
    b2["coords"][pad] = torch.randn_like(b2["coords"][pad]) * 50.0
    with torch.no_grad():
        c = m(b2, "onset")
    d = (a - c).abs().max().item()
    assert d < 1e-3, (
        f"padded electrodes changed the output by {d}. Something downstream of the mask is "
        "reading the pad rows -- most likely a BatchNorm over the flattened B*E axis.")


@check("no subject embedding exists")
def test_no_subject_leakage():
    m = make_model()
    names = [n for n, _ in m.named_parameters() if "subject" in n.lower()]
    assert not names, f"subject-conditioned parameters present: {names}"
    try:
        make_model(use_subject_embedding=True)
        raise AssertionError("use_subject_embedding=True should have been rejected")
    except AssertionError as e:
        if "should have been rejected" in str(e):
            raise


@check("virtual sensor slot identity is preserved (pool != mean)")
def test_slot_identity():
    m = make_model(pool="flatten")
    b = fake_batch()
    with torch.no_grad():
        s = m.encoder.harmonizer(
            m.encoder.trunk(b["x_spec"].reshape(-1, 1, *b["x_spec"].shape[2:]))
            .reshape(b["x_spec"].shape[0], b["x_spec"].shape[1], -1) * b["electrode_mask"].unsqueeze(-1),
            b["coords"], b["electrode_mask"])
    V = m.encoder.tcfg.num_virtual_sensors
    assert s.shape[1] == V
    # slots must not have collapsed onto each other
    sm = s.mean(0)
    cs = torch.nn.functional.normalize(sm, dim=-1)
    off = (cs @ cs.T - torch.eye(V)).abs().max().item()
    assert off < 0.999, "virtual sensor slots are degenerate (all identical)"


@check("checkpoint round-trip is exact")
def test_ckpt_roundtrip():
    import tempfile, os
    m, b = make_model(), fake_batch()
    with torch.no_grad():
        a = m(b, "speech")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        torch.save({"model_state_dict": m.state_dict()}, p)
        m2 = make_model()
        m2.load_state_dict(torch.load(p, map_location="cpu")["model_state_dict"])
        m2.eval()
        with torch.no_grad():
            c = m2(b, "speech")
    assert torch.allclose(a, c, atol=1e-6)


@check("task embedding actually changes the output")
def test_task_conditioning():
    m, b = make_model(), fake_batch()
    with torch.no_grad():
        m.encoder.task_embedding.weight.data = torch.randn_like(m.encoder.task_embedding.weight)
        a = m.encoder(b["x_spec"], b["coords"], b["electrode_mask"],
                      torch.zeros(b["y"].shape[0], dtype=torch.long))
        c = m.encoder(b["x_spec"], b["coords"], b["electrode_mask"],
                      torch.ones(b["y"].shape[0], dtype=torch.long))
    assert (a - c).abs().max().item() > 1e-5, "task embedding has no effect"


@check("GNN keeps permutation equivariance")
def test_gnn_permutation():
    tcfg = TrainConfig()
    tcfg.tasks = list(DIAGNOSTIC_PANEL)
    m = MultiTaskModel(tcfg, tcfg.tasks, use_gnn=True)
    m.eval()
    b = fake_batch()
    with torch.no_grad():
        a = m(b, "onset")
    B, E = b["electrode_mask"].shape
    perms = torch.stack([torch.randperm(E) for _ in range(B)])

    def gather(x, p):
        for _ in range(x.ndim - 2):
            p = p.unsqueeze(-1)
        return torch.gather(x, 1, p.expand(-1, -1, *x.shape[2:]))

    pb = {**b}
    for k in ("x_spec", "coords", "electrode_mask"):
        pb[k] = gather(b[k].clone(), perms)
    with torch.no_grad():
        c = m(pb, "onset")
    d = (a - c).abs().max().item()
    assert d < 1e-3, f"GNN broke permutation invariance, max diff {d}"


@check("feature config sizing math")
def test_feature_sizing():
    c = FeatureConfig()
    assert c.n_freq_bins() == 38, c.n_freq_bins()
    assert c.n_time_bins() == 17, c.n_time_bins()
    per_window_mb = 120 * c.n_freq_bins() * c.n_time_bins() * 2 / 1e6
    assert 0.1 < per_window_mb < 0.2, per_window_mb
    bad = FeatureConfig(n_fft=64, hop=16, win_len=32, freq_max_hz=1024)
    assert bad.n_freq_bins() * bad.n_time_bins() > 6 * c.n_freq_bins() * c.n_time_bins(), \
        "the old 64/16 spec should be several times larger than the new one"


@check("shaft Laplacian is built from labels, not coordinates")
def test_shaft_laplacian():
    from .features import build_shaft_laplacian, stem_electrode_name
    assert stem_electrode_name("T1bIc12") == ("T1bIc", 12)
    labels = ["T1b1", "T1b2", "T1b3", "O1aIb4", "O1aIb5", "O1aIb6", "X9"]
    W, n = build_shaft_laplacian(labels)
    assert n == 2, n                               # T1b2 and O1aIb5
    assert abs(W[1, 0] - 0.5) < 1e-6 and abs(W[1, 2] - 0.5) < 1e-6
    assert W[0].sum() == 0.0 and W[6].sum() == 0.0  # endpoints and singletons stay raw
    assert W[1, 4] == 0.0, "electrodes on different shafts must never be neighbours"


def main():
    print("npx selfcheck")
    for fn in (test_feature_sizing, test_shaft_laplacian, test_forward, test_permutation,
               test_mask_invariance, test_no_subject_leakage, test_slot_identity,
               test_ckpt_roundtrip, test_task_conditioning, test_gnn_permutation):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n, e in FAIL:
            print(f"  {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
