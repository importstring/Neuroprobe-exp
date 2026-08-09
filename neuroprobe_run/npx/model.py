"""
npx/model.py  --  encoder, harmonizer, optional anatomical GNN, multi-task heads.

Replaces `ResidualConvBlock`, `VirtualSensorHarmonizer`, `LeaderboardSpectrogramEncoder`,
`LeaderboardMultiTaskModel`.

Changes from your version, in descending order of how much they cost you:

1.  `nn.AdaptiveAvgPool2d((1, 1))` in the electrode trunk averaged the ENTIRE
    frequency-by-time spectrogram into a single scalar per channel. You built a spectrogram
    pipeline and then destroyed the spectrogram. The #1 entry is literally called
    "CNN Laplacian rereferencing spectrogram"; the frequency axis IS the signal. Here the
    trunk pools time to 1 but keeps 4 frequency bands, so band structure survives to the
    harmonizer.

2.  `pooled = sensor_tokens.mean(dim=1)` averaged the 16 virtual sensors. The whole point of
    learned persistent queries is that slot 3 always means the same thing across subjects.
    Averaging them throws away WHICH slot fired, so a signal localized in two occipital slots
    gets diluted 8x. That is a plausible reason your visual tasks sat at chance. Fixed: flatten
    (default) or masked attention pooling.

3.  `USE_SUBJECT_EMBEDDING=True` on a CROSS-SUBJECT benchmark. Test subjects 1, 3, 4, 7, 10 are
    never in training, so their embedding rows are pure initialization noise, and your
    `.clamp(0, num_embeddings-1)` silently mapped them to a random learned vector. Worse, during
    training the embedding is a free channel for the model to memorize the single training
    subject through. Removed. There is no legitimate cross-subject use of a subject ID.

4.  `use_coords_in_values=False`. Coordinates were allowed to decide WHERE to look but not to
    inform WHAT was reported. Anatomy is the only thing that means the same thing in two
    different skulls -- it belongs in the values too.

5.  Optional anatomical GNN (`--gnn`): inductive GraphSAGE-style message passing over a kNN
    graph in MNI space, before the harmonizer. Deliberately inductive (aggregation over
    neighbours, no fixed node ordering, no learned per-node parameters, no eigendecomposition)
    so it transfers to a montage it has never seen. The 0.515 GLIS-GNN entry on the leaderboard
    is weak evidence against GNNs, not strong: it is one attempt by one person who says he did
    not push it. Prohibitions that still hold: no spectral GCN, no functional/correlation graphs
    (they are session-specific and will not transfer), no Laplacian positional encodings
    (eigenvector signs and orderings are not comparable across subjects).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                          nn.GroupNorm(min(8, out_ch), out_ch))
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + r)


class ElectrodeTrunk(nn.Module):
    """
    [B*E, 1, F, T] -> [B*E, out_dim]

    GroupNorm, not BatchNorm. BatchNorm over B*E flattens 120 electrodes of one subject into
    the batch statistics, so the normalization itself becomes subject-specific -- which is a
    domain-shift leak on a cross-subject benchmark, and it changes behaviour between train and
    eval in a way that interacts badly with batch_size=8.
    """

    def __init__(self, out_dim=96, freq_keep=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.GroupNorm(8, 32), nn.ReLU(),
            ResidualConvBlock(32, 64, stride=2),
            ResidualConvBlock(64, 96, stride=2),
            nn.AdaptiveAvgPool2d((freq_keep, 1)),  # keep frequency structure, pool time
        )
        self.proj = nn.Sequential(
            nn.Linear(96 * freq_keep, out_dim), nn.LayerNorm(out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)


# ---------------------------------------------------------------------------
class AnatomicalGNN(nn.Module):
    """
    Inductive message passing over a kNN graph in normalized MNI space.
    Edges are built ON THE FLY from the batch's coordinates, so nothing about the montage is
    baked into the weights and an unseen electrode layout is handled without retraining.
    """

    def __init__(self, dim, k=6, layers=2, dropout=0.1, sigma=0.25):
        super().__init__()
        self.k, self.sigma = k, sigma
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "self": nn.Linear(dim, dim),
                "neigh": nn.Linear(dim, dim),
                "edge": nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)),
                "norm": nn.LayerNorm(dim),
            }) for _ in range(layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, h, coords, mask):
        """h [B,E,D], coords [B,E,3] normalized, mask [B,E] bool"""
        B, E, D = h.shape
        with torch.no_grad():
            d2 = torch.cdist(coords, coords)                     # [B,E,E]
            invalid = ~mask[:, None, :] | torch.eye(E, dtype=torch.bool, device=h.device)[None]
            d2 = d2.masked_fill(invalid, float("inf"))
            k = min(self.k, E - 1)
            nd, ni = torch.topk(d2, k, dim=-1, largest=False)     # [B,E,k]
            valid = torch.isfinite(nd)
            ni = ni.masked_fill(~valid, 0)
            nd = nd.masked_fill(~valid, 0.0)

        idx = ni.unsqueeze(-1).expand(-1, -1, -1, D)
        for blk in self.blocks:
            hn = torch.gather(h.unsqueeze(1).expand(-1, E, -1, -1), 2, idx)   # [B,E,k,D]
            rel = coords.unsqueeze(2).expand(-1, -1, k, -1) - torch.gather(
                coords.unsqueeze(1).expand(-1, E, -1, -1), 2,
                ni.unsqueeze(-1).expand(-1, -1, -1, 3))
            ef = torch.cat([rel, nd.unsqueeze(-1)], dim=-1)                   # [B,E,k,4]
            w = blk["edge"](ef).squeeze(-1)
            w = w.masked_fill(~valid, float("-inf"))
            w = torch.softmax(w, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
            agg = (w.unsqueeze(-1) * hn).sum(2)                               # [B,E,D]
            h = blk["norm"](h + self.drop(F.relu(blk["self"](h) + blk["neigh"](agg))))
            h = h * mask.unsqueeze(-1)
        return h


# ---------------------------------------------------------------------------
class VirtualSensorHarmonizer(nn.Module):
    def __init__(self, elec_hidden_dim, coord_dim, coord_emb_dim, model_dim,
                 num_virtual_sensors, num_heads, attn_dropout=0.1, dropout=0.1,
                 use_coords_in_keys=True, use_coords_in_values=True, use_sensor_self_attn=True):
        super().__init__()
        self.use_coords_in_keys = use_coords_in_keys
        self.use_coords_in_values = use_coords_in_values
        self.use_sensor_self_attn = use_sensor_self_attn

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, coord_emb_dim), nn.LayerNorm(coord_emb_dim), nn.ReLU(),
            nn.Linear(coord_emb_dim, coord_emb_dim),
        )
        kd = elec_hidden_dim + (coord_emb_dim if use_coords_in_keys else 0)
        vd = elec_hidden_dim + (coord_emb_dim if use_coords_in_values else 0)
        self.key_proj = nn.Sequential(nn.Linear(kd, model_dim), nn.LayerNorm(model_dim))
        self.val_proj = nn.Sequential(nn.Linear(vd, model_dim), nn.LayerNorm(model_dim))

        # Orthogonal init, unit scale. Your `torch.randn(V, D) * 0.02` was a real problem:
        # queries that small make the attention logits nearly identical for every slot, so all V
        # slots start as the same near-uniform average over electrodes and have to break symmetry
        # by themselves. Under LayerNorm + weight decay they often never do -- which is one more
        # reason averaging them looked harmless. Orthogonal rows start the slots distinguishable.
        q = torch.empty(num_virtual_sensors, model_dim)
        nn.init.orthogonal_(q)
        self.virtual_queries = nn.Parameter(q)
        self.cross_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=attn_dropout,
                                                batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(model_dim)
        self.cross_attn_dropout = nn.Dropout(dropout)

        if use_sensor_self_attn:
            self.sensor_self_attn = nn.MultiheadAttention(model_dim, num_heads,
                                                          dropout=attn_dropout, batch_first=True)
            self.sensor_attn_norm = nn.LayerNorm(model_dim)
            self.sensor_attn_dropout = nn.Dropout(dropout)
        self.sensor_ff_norm = nn.LayerNorm(model_dim)
        self.sensor_ff = nn.Sequential(nn.Linear(model_dim, model_dim * 4), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(model_dim * 4, model_dim))
        self.sensor_ff_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, elec_feat, coords, elec_mask, return_attn=False):
        assert elec_feat.shape[:2] == coords.shape[:2] == elec_mask.shape
        assert elec_mask.any(1).all(), "every sample needs >=1 real electrode"

        cf = self.coord_mlp(coords)
        ki = torch.cat([elec_feat, cf], -1) if self.use_coords_in_keys else elec_feat
        vi = torch.cat([elec_feat, cf], -1) if self.use_coords_in_values else elec_feat
        keys, values = self.key_proj(ki), self.val_proj(vi)

        B = elec_feat.size(0)
        q = self.virtual_queries.unsqueeze(0).expand(B, -1, -1)
        out, attn = self.cross_attn(
            query=self.cross_attn_norm(q), key=keys, value=values,
            key_padding_mask=~elec_mask.bool(), need_weights=return_attn,
            average_attn_weights=True,
        )
        s = q + self.cross_attn_dropout(out)
        if self.use_sensor_self_attn:
            sn = self.sensor_attn_norm(s)
            o2, _ = self.sensor_self_attn(sn, sn, sn, need_weights=False)
            s = s + self.sensor_attn_dropout(o2)
        s = s + self.sensor_ff_dropout(self.sensor_ff(self.sensor_ff_norm(s)))
        s = self.out_norm(s)
        return (s, attn) if return_attn else s


# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, tcfg, task_count, use_gnn=False, gnn_k=6, gnn_layers=2):
        super().__init__()
        self.tcfg = tcfg
        self.trunk = ElectrodeTrunk(out_dim=tcfg.elec_hidden_dim, dropout=tcfg.dropout)
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = AnatomicalGNN(tcfg.elec_hidden_dim, k=gnn_k, layers=gnn_layers,
                                     dropout=tcfg.dropout)
        self.harmonizer = VirtualSensorHarmonizer(
            elec_hidden_dim=tcfg.elec_hidden_dim, coord_dim=3,
            coord_emb_dim=tcfg.coord_emb_dim, model_dim=tcfg.model_dim,
            num_virtual_sensors=tcfg.num_virtual_sensors, num_heads=tcfg.num_sensor_heads,
            attn_dropout=tcfg.attn_dropout, dropout=tcfg.dropout,
            use_coords_in_keys=tcfg.use_coords_in_keys,
            use_coords_in_values=tcfg.use_coords_in_values,
            use_sensor_self_attn=tcfg.use_sensor_self_attn,
        )
        if tcfg.pool == "flatten":
            pooled_dim = tcfg.model_dim * tcfg.num_virtual_sensors
        elif tcfg.pool == "attn":
            pooled_dim = tcfg.model_dim
            self.pool_q = nn.Parameter(torch.randn(tcfg.model_dim) * 0.02)
        elif tcfg.pool == "mean":  # the old behaviour, kept for ablation
            pooled_dim = tcfg.model_dim
        else:
            raise ValueError(tcfg.pool)

        cond = tcfg.task_emb_dim if tcfg.use_task_embedding else 0
        self.task_embedding = nn.Embedding(task_count, tcfg.task_emb_dim) if tcfg.use_task_embedding else None
        assert not tcfg.use_subject_embedding, (
            "use_subject_embedding must be False for cross-subject. Test subjects never appear "
            "in training, so their embedding rows are untrained noise."
        )
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim + cond, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(tcfg.dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(tcfg.dropout),
        )
        self.out_dim = 128

    def forward(self, x_spec, coords, electrode_mask, task_idx):
        B, E, Fb, Tb = x_spec.shape
        h = self.trunk(x_spec.reshape(B * E, 1, Fb, Tb)).reshape(B, E, -1)
        h = h * electrode_mask.unsqueeze(-1)
        if self.use_gnn:
            h = self.gnn(h, coords, electrode_mask)
        s = self.harmonizer(h, coords, electrode_mask)          # [B,V,D]

        if self.tcfg.pool == "flatten":
            pooled = s.flatten(1)
        elif self.tcfg.pool == "attn":
            w = torch.softmax((s @ self.pool_q) / (s.size(-1) ** 0.5), dim=1)
            pooled = (w.unsqueeze(-1) * s).sum(1)
        else:
            pooled = s.mean(1)

        parts = [pooled]
        if self.task_embedding is not None:
            parts.append(self.task_embedding(task_idx))
        return self.proj(torch.cat(parts, -1))


class MultiTaskModel(nn.Module):
    def __init__(self, tcfg, tasks, use_gnn=False, gnn_k=6, gnn_layers=2):
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_id = {t: i for i, t in enumerate(self.tasks)}
        self.encoder = Encoder(tcfg, len(self.tasks), use_gnn=use_gnn, gnn_k=gnn_k,
                               gnn_layers=gnn_layers)
        d = self.encoder.out_dim
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(tcfg.dropout),
                             nn.Linear(64, 1)) for t in self.tasks
        })
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.tasks})
        self.loss_weighting = tcfg.loss_weighting

    def forward(self, batch, task_name):
        x = batch["x_spec"]
        ti = torch.full((x.shape[0],), self.task_to_id[task_name],
                        device=x.device, dtype=torch.long)
        z = self.encoder(x, batch["coords"], batch["electrode_mask"], ti)
        return self.heads[task_name](z).squeeze(-1)

    def weighted_loss(self, raw_loss, task_name):
        if self.loss_weighting == "uniform":
            return raw_loss
        lv = self.log_vars[task_name]
        return torch.exp(-lv) * raw_loss + 0.5 * lv


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
