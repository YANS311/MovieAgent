"""Chapter-3 SKB-HIF computation graph with an explicit M3L field contract.

Unavailable gender/age/occupation fields are absent. They are neither zero-filled
nor replaced with behavior features. The random-split M3L protocol supplies a
train-only, leave-one-target-out history rather than a temporal sequence.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def mlp(width_in, widths, dropout):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(width_in, width), nn.ReLU()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        width_in = width
    return nn.Sequential(*layers), width_in


class MultiHeadRankedQuadraticPair(nn.Module):
    def __init__(self, quadratic_dim=128, num_heads=4, row_rank=2):
        super().__init__()
        assert quadratic_dim % num_heads == 0
        self.quadratic_dim = quadratic_dim
        self.num_heads = num_heads
        self.row_rank = row_rank
        self.head_dim = quadratic_dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, row_rank, self.head_dim, self.head_dim))
        self.bias = nn.Parameter(torch.empty(num_heads, row_rank, self.head_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, e_a, e_b):
        batch = e_a.size(0)
        a = e_a.reshape(batch, self.num_heads, self.head_dim)
        b = e_b.reshape(batch, self.num_heads, self.head_dim)
        response = torch.einsum('bhj,hrtj->bhrt', b, self.weight) + self.bias.unsqueeze(0)
        response = F.relu(response)
        return (a.unsqueeze(2) * response).sum(dim=2).reshape(batch, self.quadratic_dim) / math.sqrt(self.row_rank)


class GroupedMultiHeadQuadraticInteraction(nn.Module):
    PAIRS = (('I1', 'up', 'item'), ('I2', 'beh', 'item'),
             ('I3', 'sem', 'item'), ('I4', 'beh', 'sem'))

    def __init__(self, group_dims, quadratic_dim=128, output_dim=256, num_heads=4, row_rank=2):
        super().__init__()
        self.group_projections = nn.ModuleDict({k: nn.Linear(v, quadratic_dim) for k, v in group_dims.items()})
        self.pair_maps = nn.ModuleDict({
            name: MultiHeadRankedQuadraticPair(quadratic_dim, num_heads, row_rank)
            for name, _, _ in self.PAIRS
        })
        self.output_projection = nn.Linear(len(self.PAIRS) * quadratic_dim, output_dim)

    def forward(self, up, item, beh, sem):
        groups = {'up': up, 'item': item, 'beh': beh, 'sem': sem}
        projected = {k: self.group_projections[k](v) for k, v in groups.items()}
        interactions = {
            name: self.pair_maps[name](projected[left], projected[right])
            for name, left, right in self.PAIRS
        }
        concat = torch.cat([interactions[name] for name, _, _ in self.PAIRS], dim=-1)
        return F.relu(self.output_projection(concat)), projected, interactions, concat


class BoundedRMSScaleBalance(nn.Module):
    def __init__(self, s_min=.5, s_max=2., scale_init=1., eps=1e-8):
        super().__init__()
        self.s_min, self.s_max, self.eps = s_min, s_max, eps
        p = (scale_init - s_min) / (s_max - s_min)
        raw = math.log(p / (1 - p))
        self.raw_s_quad = nn.Parameter(torch.tensor(raw))
        self.raw_s_implicit = nn.Parameter(torch.tensor(raw))

    def bounded(self, x):
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(x)

    def forward(self, z_quad, z_implicit):
        rq = torch.sqrt(z_quad.square().mean(-1, keepdim=True) + self.eps)
        rb = torch.sqrt(z_implicit.square().mean(-1, keepdim=True) + self.eps)
        sq, sb = self.bounded(self.raw_s_quad), self.bounded(self.raw_s_implicit)
        return sq * z_quad / rq, sb * z_implicit / rb, rq, rb, sq, sb


class HybridBehaviorFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Linear(3 * hidden, hidden)

    def forward(self, z_bh, z_quad, z_cross):
        concat = torch.cat((z_bh, z_quad, z_cross), dim=-1)
        return F.relu(self.projection(concat)), concat


class CanonicalSKBHIFFinal(nn.Module):
    """SKB-HIF with only unavailable demographic subfields disabled."""

    def __init__(self, n_users, n_items, text_bank, genre_item_ids, genre_item_mask,
                 n_genres=19, embed_dim=128, dropout=.1):
        super().__init__()
        self.n_users, self.n_items, self.embedding_dim = n_users, n_items, embed_dim
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.genre_embedding = nn.Embedding(n_genres, embed_dim)
        self.text_projection = nn.Linear(text_bank.shape[1], embed_dim)
        self.register_buffer('text_bank', text_bank.float(), persistent=False)
        self.register_buffer('genre_item_ids', genre_item_ids.long(), persistent=False)
        self.register_buffer('genre_item_mask', genre_item_mask.bool(), persistent=False)
        nn.init.normal_(self.user_embedding.weight, std=.05)
        nn.init.normal_(self.item_embedding.weight, std=.05)
        nn.init.normal_(self.genre_embedding.weight, std=.02)

        # Chapter 3 shared fields minus explicitly absent demographic subfields:
        # user, movie, pooled genre, pooled history, projected semantic vector.
        shared_width = 5 * embed_dim
        self.attention, attention_width = mlp(4 * embed_dim, (512, 256), dropout)
        self.attention_head = nn.Linear(attention_width, 1)
        self.mlp_sk, hidden = mlp(shared_width, (512, 256), dropout)
        self.mlp_behavior, _ = mlp(shared_width + embed_dim, (512, 256), dropout)
        self.grouped_quadratic = GroupedMultiHeadQuadraticInteraction(
            {'up': embed_dim, 'item': embed_dim, 'beh': 2 * embed_dim, 'sem': 2 * embed_dim},
            quadratic_dim=128, output_dim=hidden, num_heads=4, row_rank=2)
        self.cross_scale_balance = BoundedRMSScaleBalance(.5, 2., 1., 1e-8)
        self.hybrid_fusion = HybridBehaviorFusion(hidden)
        self.vector_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.dnn_predict = nn.Linear(hidden, 1, bias=False)
        # Chapter 3 linear columns are profile + rag; absent profiles leave rag only.
        self.semantic_linear = nn.Linear(embed_dim, 1, bias=True)
        self._eval_text_cache = None
        self._eval_genre_cache = None

    def train(self, mode=True):
        if mode:
            self._eval_text_cache = self._eval_genre_cache = None
        return super().train(mode)

    @torch.no_grad()
    def prepare_catalog(self):
        self._eval_text_cache = self.text_projection(self.text_bank)
        values = self.genre_embedding(self.genre_item_ids)
        mask = self.genre_item_mask.to(values.dtype).unsqueeze(-1)
        self._eval_genre_cache = (values * mask).sum(1) / mask.sum(1).clamp_min(1.)

    def _text(self, items):
        if not self.training and self._eval_text_cache is not None:
            return self._eval_text_cache[items]
        return self.text_projection(self.text_bank[items])

    def _genre(self, items):
        if not self.training and self._eval_genre_cache is not None:
            return self._eval_genre_cache[items]
        values = self.genre_embedding(self.genre_item_ids[items])
        mask = self.genre_item_mask[items].to(values.dtype).unsqueeze(-1)
        return (values * mask).sum(-2) / mask.sum(-2).clamp_min(1.)

    def _history(self, history_ids, history_mask):
        ids = history_ids.clamp(0, self.n_items - 1)
        values = self.item_embedding(ids)
        mask = history_mask.to(values.dtype)
        pooled = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.)
        return values, mask, pooled

    def _attention_pool(self, query, keys, mask):
        q = query.unsqueeze(-2).expand(*query.shape[:-1], keys.shape[-2], query.shape[-1])
        attention_input = torch.cat((q, keys, q - keys, q * keys), dim=-1)
        logits = self.attention_head(self.attention(attention_input)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        weights = F.softmax(logits, dim=-1) * mask.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        return (weights.unsqueeze(-1) * keys).sum(-2)

    def _flat_forward(self, user, item, genre, hist_pool, hist_attn, semantic, return_components):
        shared = torch.cat((user, item, genre, hist_pool, semantic), dim=-1)
        z_sk = self.mlp_sk(shared)
        z_bh = self.mlp_behavior(torch.cat((shared, hist_attn), dim=-1))
        z_quad, projected, interactions, quad_concat = self.grouped_quadratic(
            user, item, torch.cat((hist_pool, hist_attn), dim=-1), torch.cat((semantic, genre), dim=-1))
        zq, zb, rq, rb, sq, sb = self.cross_scale_balance(z_quad, z_bh)
        z_cross = zq * zb
        z_hif, hybrid_concat = self.hybrid_fusion(z_bh, z_quad, z_cross)
        gate = self.vector_gate(z_sk)
        fused = gate * z_hif + (1. - gate) * z_sk
        score = self.dnn_predict(fused).squeeze(-1) + self.semantic_linear(semantic).squeeze(-1)
        if not return_components:
            return score
        return score, {'z_sk': z_sk, 'z_bh': z_bh, 'z_quad': z_quad,
                       'z_cross': z_cross, 'z_hif': z_hif, 'gate': gate,
                       'projected': projected, 'interactions': interactions,
                       'quad_concat': quad_concat, 'hybrid_concat': hybrid_concat,
                       'rms_quad': rq, 'rms_bh': rb, 's_quad': sq, 's_bh': sb}

    def forward(self, users, items, history_ids, history_mask, return_components=False):
        device = self.user_embedding.weight.device
        users = torch.as_tensor(users, dtype=torch.long, device=device)
        items = torch.as_tensor(items, dtype=torch.long, device=device)
        vector = items.ndim == 1
        if vector:
            keys, mask, hist_pool = self._history(history_ids, history_mask)
            user, item = self.user_embedding(users), self.item_embedding(items)
            semantic, genre = self._text(items), self._genre(items)
            hist_attn = self._attention_pool(item, keys, history_mask)
            return self._flat_forward(user, item, genre, hist_pool, hist_attn, semantic, return_components)

        if items.ndim != 2 or items.shape[0] != len(users):
            raise ValueError('items must have [batch,candidates] shape')
        batch, candidates = items.shape
        keys0, mask0, pool0 = self._history(history_ids, history_mask)
        user = self.user_embedding(users)[:, None, :].expand(batch, candidates, -1)
        item, semantic, genre = self.item_embedding(items), self._text(items), self._genre(items)
        keys = keys0[:, None, :, :].expand(-1, candidates, -1, -1)
        mask = history_mask[:, None, :].expand(-1, candidates, -1)
        hist_pool = pool0[:, None, :].expand(-1, candidates, -1)
        hist_attn = self._attention_pool(item, keys, mask)
        flatten = lambda x: x.reshape(batch * candidates, x.shape[-1])
        result = self._flat_forward(flatten(user), flatten(item), flatten(genre),
                                    flatten(hist_pool), flatten(hist_attn), flatten(semantic),
                                    return_components)
        if not return_components:
            return result.reshape(batch, candidates)
        score, components = result
        return score.reshape(batch, candidates), {
            k: ({n: v.reshape(batch, candidates, -1) for n, v in x.items()}
                if isinstance(x, dict) else x.reshape(batch, candidates, *x.shape[1:]))
            for k, x in components.items()
        }
