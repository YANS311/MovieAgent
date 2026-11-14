"""PAMK-Net: Popularity-Aware Selective Modality-KG Network (and 5 Controlled Ablations/Extensions).

Variants supported:
- `pamk_full`: Primary architecture (M-Debias + M-Select + M-Freq).
- `scma_backup`: Backup architecture / Ablation 1 (w/o M-Freq: disables Popular-Niche frequency-decoupled streams).
- `pamk_wo_select`: Ablation 2 (w/o M-Select: uniform 1/3 source weights & unit channel mask).
- `pamk_wo_debias`: Ablation 3 (w/o M-Debias: uncentered CLIP features & unstandardized context channels).
- `pamk_graph_r3`: Route R3 Extension (PAMK + Top-16 collaborative-refined multimodal-KG item-item graph propagation).
- `equal_param_null_ctrl`: Equal-parameter capacity control receiving ONLY frozen Parent24 states (h_ui, v_i, semantic)
  with zero image, KG, or frequency information.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_engine import FrequencyAndEvidenceContext
from .firewall_and_assets import ASSET_PATHS, N_ITEMS, N_USERS

EPS = 1e-8
STREAM_SCALE = 0.20


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def build_refined_item_graph(
    text_deb: torch.Tensor,
    image_deb: torch.Tensor,
    kernels: torch.Tensor,
    topk: int = 16,
) -> torch.Tensor:
    """Builds Route R3 Top-K symmetric row-normalized item-item adjacency matrix (N_ITEMS, N_ITEMS)
    combining debiased modality cosine similarity, relation-averaged KG kernel similarity,
    and Train-only collaborative co-interaction cosine similarity (DIGEST/LGMRec-inspired).
    """
    device = text_deb.device
    with torch.no_grad():
        s_text = torch.matmul(text_deb, text_deb.T)
        s_img = torch.matmul(image_deb, image_deb.T)
        s_mm = 0.5 * (s_text + s_img)
        s_kg = kernels.mean(dim=0)

        csr = sp.load_npz(ASSET_PATHS["train_context_csr"]).tocsr().astype(np.float32)
        col_norms = np.sqrt(np.asarray(csr.power(2).sum(axis=0)).ravel() + 1e-8)
        csr_norm = csr @ sp.diags(1.0 / col_norms)
        s_cf_np = (csr_norm.T @ csr_norm).toarray().astype(np.float32)
        s_cf = torch.as_tensor(s_cf_np, device=device)

        s_comp = 0.40 * s_cf + 0.35 * s_kg + 0.25 * s_mm
        idx = torch.arange(N_ITEMS, device=device)
        s_comp[idx, idx] = -1e9

        vals, nbrs = torch.topk(s_comp, k=topk, dim=-1)
        vals = torch.clamp(vals, min=0.0)
        adj = torch.zeros((N_ITEMS, N_ITEMS), dtype=torch.float32, device=device)
        adj.scatter_(1, nbrs, vals)
        adj = 0.5 * (adj + adj.T)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return adj


class PAMKNet(nn.Module):
    """Unified module implementing PAMK-Net (`pamk_full`) and its 5 controlled variants."""

    VALID_VARIANTS = (
        "pamk_full",
        "scma_backup",
        "pamk_wo_popularity",
        "pamk_wo_select",
        "pamk_wo_debias",
        "pamk_wo_knowledge",
        "pamk_kg_shuffled",
        "pamk_kg_information_masked",
        "pamk_graph_r3",
        "equal_param_null_ctrl",
    )

    def __init__(
        self,
        parent: nn.Module,
        text_bank: torch.Tensor,
        image_bank: torch.Tensor,
        relation_bank: torch.Tensor,
        relation_mask: torch.Tensor,
        context_engine: FrequencyAndEvidenceContext,
        variant: str = "pamk_full",
        hidden_dim: int = 256,
        stream_dim: int = 64,
        gate_hidden_dim: int = 32,
        select_temperature: float = 0.5,
        graph_topk: int = 16,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant {variant}; expected one of {self.VALID_VARIANTS}")
        self.variant = variant
        self.parent = parent.eval().requires_grad_(False)
        self.context = context_engine
        self.hidden_dim = int(hidden_dim)
        self.stream_dim = int(stream_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.select_temperature = float(select_temperature)

        self.use_debias = variant != "pamk_wo_debias"
        self.use_select = variant != "pamk_wo_select"
        self.use_freq = variant not in ("scma_backup", "pamk_wo_popularity", "equal_param_null_ctrl")
        self.use_graph = variant == "pamk_graph_r3"
        self.is_null_ctrl = variant == "equal_param_null_ctrl"
        self.use_knowledge = variant != "pamk_wo_knowledge"
        self.mask_kg_info = variant == "pamk_kg_information_masked"
        self.is_kg_shuffled = variant == "pamk_kg_shuffled"

        # --- Module 1: M-Debias (Spherical Modality Centering & KG Relation Buffers) ---
        text_raw_l2 = l2_norm(text_bank.float())
        image_raw_l2 = l2_norm(image_bank.float())
        text_deb = l2_norm(text_raw_l2 - text_raw_l2.mean(dim=0, keepdim=True))
        image_deb = l2_norm(image_raw_l2 - image_raw_l2.mean(dim=0, keepdim=True))

        self.register_buffer("text_raw_l2", text_raw_l2, persistent=False)
        self.register_buffer("image_raw_l2", image_raw_l2, persistent=False)
        self.register_buffer("text_deb", text_deb, persistent=False)
        self.register_buffer("image_deb", image_deb, persistent=False)

        self.register_buffer("relation_bank", relation_bank.float(), persistent=False)
        self.register_buffer("relation_mask", relation_mask.bool(), persistent=False)
        rel_mask_f = relation_mask.float().unsqueeze(-1)
        simple_kg = (relation_bank.float() * rel_mask_f).sum(dim=1) / rel_mask_f.sum(dim=1).clamp_min(1.0)
        # Preserve global RMS of simple_kg while centering its anisotropy cone
        kg_centered = simple_kg - simple_kg.mean(dim=0, keepdim=True)
        kg_rms_ratio = simple_kg.norm(dim=-1, keepdim=True).mean() / kg_centered.norm(dim=-1, keepdim=True).mean().clamp_min(EPS)
        simple_kg_deb = kg_centered * kg_rms_ratio
        self.register_buffer("simple_kg_raw", simple_kg, persistent=False)
        self.register_buffer("simple_kg_deb", simple_kg_deb, persistent=False)

        if self.is_null_ctrl:
            # Equal-parameter capacity control using ONLY frozen Parent24 (h_ui, v_i, semantic) (640d)
            in_ctrl_dim = 2 * self.hidden_dim + int(parent.embedding_dim)
            self.ctrl_mlp = nn.Sequential(
                nn.Linear(in_ctrl_dim, 512, bias=True),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Linear(512, self.hidden_dim, bias=True),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )
            for m in self.ctrl_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            self.W_CTRL = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            nn.init.zeros_(self.W_CTRL.weight)
            w_probe = torch.empty_like(self.W_CTRL.weight)
            nn.init.xavier_uniform_(w_probe, gain=0.05)
            self.register_buffer("_w_ctrl_probe", w_probe, persistent=False)
            return

        # --- Collaborative Query & Modality/KG Projections ---
        self.query_projection = nn.Linear(3 * self.hidden_dim, self.hidden_dim, bias=True)
        self.text_projection = nn.Linear(text_bank.shape[1], self.hidden_dim, bias=True)
        self.image_projection = nn.Linear(image_bank.shape[1], self.hidden_dim, bias=True)
        if self.use_knowledge:
            self.kg_projection = nn.Linear(relation_bank.shape[2], self.hidden_dim, bias=True)
            for layer in (self.query_projection, self.text_projection, self.image_projection, self.kg_projection):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            self.kg_projection = None
            for layer in (self.query_projection, self.text_projection, self.image_projection):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # --- Optional Route R3: Collaborative-Refined Item-Item Graph ---
        if self.use_graph:
            adj_ref = build_refined_item_graph(text_deb, image_deb, self.context.kernels, topk=graph_topk)
            self.register_buffer("adj_ref", adj_ref, persistent=False)
            self.graph_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=True)
            nn.init.xavier_uniform_(self.graph_gate.weight, gain=0.2)
            nn.init.constant_(self.graph_gate.bias, -2.0)

        # --- Module 2: M-Select (History-Consistency Guided Selective Modality-KG Gate) ---
        gate_in_dim = 19 if self.use_knowledge else 5
        gate_out_dim = (3 + 14) if self.use_knowledge else (2 + 2)
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, self.gate_hidden_dim, bias=True),
            nn.LayerNorm(self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, gate_out_dim, bias=True),
        )
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

        self.W_MM = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.zeros_(self.W_MM.weight)
        self.W_M = nn.Linear(2, self.hidden_dim, bias=False)
        nn.init.zeros_(self.W_M.weight)
        if self.use_knowledge:
            self.W_KG = nn.Linear(relation_bank.shape[2], self.hidden_dim, bias=False)
            self.W_A = nn.Linear(4, self.hidden_dim, bias=False)
            self.W_C = nn.Linear(8, self.hidden_dim, bias=False)
            for layer in (self.W_KG, self.W_A, self.W_C):
                nn.init.zeros_(layer.weight)
        else:
            self.W_KG = None
            self.W_A = None
            self.W_C = None

        # --- Module 3: M-Freq (Popular-Niche Frequency-Decoupled Dual-Stream Modulator) ---
        hc_dim = 19 if self.use_knowledge else 5
        stream_in_dim = hc_dim + 6 if self.use_freq else hc_dim
        self.stream_proj = nn.Linear(self.hidden_dim, 16, bias=True)
        nn.init.xavier_uniform_(self.stream_proj.weight, gain=0.5)
        nn.init.zeros_(self.stream_proj.bias)

        self.pop_mlp = nn.Sequential(
            nn.Linear(stream_in_dim + 16, self.stream_dim, bias=True),
            nn.LayerNorm(self.stream_dim),
            nn.GELU(),
            nn.Linear(self.stream_dim, self.stream_dim, bias=True),
            nn.LayerNorm(self.stream_dim),
            nn.GELU(),
        )
        self.niche_mlp = nn.Sequential(
            nn.Linear(stream_in_dim + 16, self.stream_dim, bias=True),
            nn.LayerNorm(self.stream_dim),
            nn.GELU(),
            nn.Linear(self.stream_dim, self.stream_dim, bias=True),
            nn.LayerNorm(self.stream_dim),
            nn.GELU(),
        )
        for mlp in (self.pop_mlp, self.niche_mlp):
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

        self.W_POP = nn.Linear(self.stream_dim, self.hidden_dim, bias=False)
        self.W_NICHE = nn.Linear(self.stream_dim, self.hidden_dim, bias=False)
        nn.init.zeros_(self.W_POP.weight)
        nn.init.zeros_(self.W_NICHE.weight)

        for name, layer in [
            ("_probe_mm", self.W_MM),
            ("_probe_pop", self.W_POP),
            ("_probe_niche", self.W_NICHE),
        ]:
            probe = torch.empty_like(layer.weight)
            nn.init.xavier_uniform_(probe, gain=0.02)
            self.register_buffer(name, probe, persistent=False)

        self._eval_text_cache: Optional[torch.Tensor] = None
        self._eval_image_cache: Optional[torch.Tensor] = None
        self._eval_kg_cache: Optional[torch.Tensor] = None

    def train(self, mode: bool = True) -> "PAMKNet":
        super().train(mode)
        self.parent.eval()
        if mode:
            self.clear_catalog()
        return self

    def clear_catalog(self) -> None:
        self._eval_text_cache = None
        self._eval_image_cache = None
        self._eval_kg_cache = None

    def _active_banks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mask_kg_info:
            return (
                self.text_deb if self.use_debias else self.text_raw_l2,
                self.image_deb if self.use_debias else self.image_raw_l2,
                torch.zeros_like(self.simple_kg_deb if self.use_debias else self.simple_kg_raw),
            )
        if self.use_debias:
            return self.text_deb, self.image_deb, self.simple_kg_deb
        return self.text_raw_l2, self.image_raw_l2, self.simple_kg_raw

    @torch.no_grad()
    def prepare_catalog(self) -> None:
        self.parent.prepare_catalog()
        if self.is_null_ctrl or self.training:
            return
        t_bank, i_bank, k_bank = self._active_banks()
        t_proj = l2_norm(self.text_projection(t_bank))
        i_proj = l2_norm(self.image_projection(i_bank))
        k_proj = l2_norm(self.kg_projection(k_bank)) if self.use_knowledge else None
        if self.mask_kg_info and k_proj is not None:
            k_proj = torch.zeros_like(k_proj)
        if self.use_graph:
            t_nbr = torch.matmul(self.adj_ref, t_proj)
            i_nbr = torch.matmul(self.adj_ref, i_proj)
            g_t = torch.sigmoid(self.graph_gate(torch.cat([t_proj, t_nbr], dim=-1)))
            g_i = torch.sigmoid(self.graph_gate(torch.cat([i_proj, i_nbr], dim=-1)))
            t_proj = l2_norm(t_proj + g_t * t_nbr)
            i_proj = l2_norm(i_proj + g_i * i_nbr)
            if self.use_knowledge:
                k_nbr = torch.matmul(self.adj_ref, k_proj)
                g_k = torch.sigmoid(self.graph_gate(torch.cat([k_proj, k_nbr], dim=-1)))
                k_proj = l2_norm(k_proj + g_k * k_nbr)
        self._eval_text_cache = t_proj
        self._eval_image_cache = i_proj
        self._eval_kg_cache = k_proj

    @torch.no_grad()
    def _frozen_parent_flat(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact 1D flat parent forward matching SpecificationCompleteU6._frozen_parent_flat."""
        p = self.parent
        keys, _, hist_pool = p._history(history_ids, history_mask)
        user, item = p.user_embedding(users), p.item_embedding(items)
        semantic, genre = p._text(items), p._genre(items)
        hist_attn = p._attention_pool(item, keys, history_mask)
        shared = torch.cat((user, item, genre, hist_pool, semantic), dim=-1)
        z_sk = p.mlp_sk(shared)
        z_bh = p.mlp_behavior(torch.cat((shared, hist_attn), dim=-1))
        z_quad, _, _, _ = p.grouped_quadratic(
            user, item, torch.cat((hist_pool, hist_attn), dim=-1), torch.cat((semantic, genre), dim=-1)
        )
        zq, zb, _, _, _, _ = p.cross_scale_balance(z_quad, z_bh)
        z_hif = p.hybrid_fusion(z_bh, z_quad, zq * zb)[0]
        gate = p.vector_gate(z_sk)
        h_ui = gate * z_hif + (1.0 - gate) * z_sk
        score = p.dnn_predict(h_ui).squeeze(-1) + p.semantic_linear(semantic).squeeze(-1)
        return score, h_ui, z_hif, semantic

    @torch.no_grad()
    def _frozen_parent_grid(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact 2D grid parent forward over (B_u, C) without redundant history embedding lookups."""
        p = self.parent
        batch, candidates = items.shape
        keys0, _, pool0 = p._history(history_ids, history_mask)
        user = p.user_embedding(users)[:, None, :].expand(batch, candidates, -1)
        item = p.item_embedding(items)
        semantic = p._text(items)
        genre = p._genre(items)
        keys = keys0[:, None, :, :].expand(-1, candidates, -1, -1)
        mask = history_mask[:, None, :].expand(-1, candidates, -1)
        hist_pool = pool0[:, None, :].expand(-1, candidates, -1)
        hist_attn = p._attention_pool(item, keys, mask)

        flat = lambda x: x.reshape(batch * candidates, x.shape[-1])
        u_f, i_f, g_f, hp_f, ha_f, s_f = (
            flat(user),
            flat(item),
            flat(genre),
            flat(hist_pool),
            flat(hist_attn),
            flat(semantic),
        )
        shared = torch.cat((u_f, i_f, g_f, hp_f, s_f), dim=-1)
        z_sk = p.mlp_sk(shared)
        z_bh = p.mlp_behavior(torch.cat((shared, ha_f), dim=-1))
        z_quad, _, _, _ = p.grouped_quadratic(
            u_f, i_f, torch.cat((hp_f, ha_f), dim=-1), torch.cat((s_f, g_f), dim=-1)
        )
        zq, zb, _, _, _, _ = p.cross_scale_balance(z_quad, z_bh)
        z_hif = p.hybrid_fusion(z_bh, z_quad, zq * zb)[0]
        gate = p.vector_gate(z_sk)
        h_ui = gate * z_hif + (1.0 - gate) * z_sk
        score = p.dnn_predict(h_ui).squeeze(-1) + p.semantic_linear(s_f).squeeze(-1)
        return score, h_ui, z_hif, s_f

    def _project_item_modalities(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.training and self._eval_text_cache is not None:
            return (
                self._eval_text_cache[items],
                self._eval_image_cache[items],
                self._eval_kg_cache[items] if self.use_knowledge else None,
            )
        t_bank, i_bank, k_bank = self._active_banks()
        if self.use_graph:
            t_all = l2_norm(self.text_projection(t_bank))
            i_all = l2_norm(self.image_projection(i_bank))
            t_nbr = torch.matmul(self.adj_ref, t_all)
            i_nbr = torch.matmul(self.adj_ref, i_all)
            g_t = torch.sigmoid(self.graph_gate(torch.cat([t_all[items], t_nbr[items]], dim=-1)))
            g_i = torch.sigmoid(self.graph_gate(torch.cat([i_all[items], i_nbr[items]], dim=-1)))
            t_item = l2_norm(t_all[items] + g_t * t_nbr[items])
            i_item = l2_norm(i_all[items] + g_i * i_nbr[items])
            if self.use_knowledge:
                k_all = l2_norm(self.kg_projection(k_bank))
                k_nbr = torch.matmul(self.adj_ref, k_all)
                g_k = torch.sigmoid(self.graph_gate(torch.cat([k_all[items], k_nbr[items]], dim=-1)))
                k_item = l2_norm(k_all[items] + g_k * k_nbr[items])
            else:
                k_item = None
            return t_item, i_item, k_item

        t_item = l2_norm(self.text_projection(t_bank[items]))
        i_item = l2_norm(self.image_projection(i_bank[items]))
        k_item = l2_norm(self.kg_projection(k_bank[items])) if self.use_knowledge else None
        return t_item, i_item, k_item

    def _compute_delta_from_features(
        self,
        items_flat: torch.Tensor,
        h_ui: torch.Tensor,
        v_i: torch.Tensor,
        semantic: torch.Tensor,
        ctx_dict: Dict[str, torch.Tensor],
        return_telemetry: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Core PAMK-Net computation given flattened (N, D) Parent24 representations and context dict."""
        if self.is_null_ctrl:
            ctrl_feat = l2_norm(self.ctrl_mlp(torch.cat([h_ui, v_i, semantic], dim=-1))) * STREAM_SCALE
            delta_h = self.W_CTRL(ctrl_feat)
            if self.training:
                delta_h = delta_h + F.linear(ctrl_feat - ctrl_feat.detach(), self._w_ctrl_probe)
            telemetry = {"delta_h": delta_h} if return_telemetry else None
            return delta_h, telemetry

        if not self.use_knowledge:
            # 1. Collaborative query & 2-modality representations (pure text + image, zero KG)
            wq1, wq2, wq3 = self.query_projection.weight.chunk(3, dim=1)
            q_raw = (
                F.linear(h_ui, wq1, self.query_projection.bias)
                + F.linear(v_i, wq2)
                + F.linear(h_ui * v_i, wq3)
            )
            query = l2_norm(q_raw)
            text, image, _ = self._project_item_modalities(items_flat)

            a_text = (query * text).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            a_image = (query * image).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            cos_ti = (text * image).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

            d_text = (1.0 + a_text) / 2.0
            d_image = (1.0 + a_image) / 2.0

            aligned_text = l2_norm(text + d_text * query)
            aligned_image = l2_norm(image + d_image * query)

            m_raw = ctx_dict["m_mean"]
            freq_ctx = ctx_dict["freq_ctx"]
            p_t = ctx_dict["p_t"].unsqueeze(-1)
            nu_t = ctx_dict["nu_t"].unsqueeze(-1)

            if self.use_debias:
                m_gate = ctx_dict["m_std"]
                mean_sig_m = self.context.sigma_m.mean()
                m_eff = m_gate * mean_sig_m
            else:
                m_gate = m_raw
                m_eff = m_raw

            # 5-dim Gate input: a_text, a_image, cos_ti, m_gate (2 dims)
            g_in = torch.cat([a_text, a_image, cos_ti, m_gate], dim=-1)
            gate_out = self.gate_net(g_in)
            src_logits = gate_out[:, :2]
            chan_logits = gate_out[:, 2:]

            if self.use_select:
                inv_d = 1.0 / float(self.hidden_dim)
                var_t = (2.0 - 2.0 * a_text) * inv_d
                var_i = (2.0 - 2.0 * a_image) * inv_d
                prior_logits = torch.cat([-var_t, -var_i], dim=-1)
                pi_src = F.softmax((src_logits + prior_logits) / self.select_temperature, dim=-1)
                g_m = 2.0 * torch.sigmoid(chan_logits)
            else:
                pi_src = torch.full((h_ui.size(0), 2), 0.5, dtype=h_ui.dtype, device=h_ui.device)
                g_m = torch.ones((h_ui.size(0), 2), dtype=h_ui.dtype, device=h_ui.device)

            pi_t = pi_src[:, 0:1]
            pi_i = pi_src[:, 1:2]
            e_mm_sel = l2_norm(pi_t * aligned_text + pi_i * aligned_image)
            m_sel = g_m * m_eff

            delta_h = self.W_MM(e_mm_sel) + self.W_M(m_sel)

            # Module 3 (M-Freq): Popular-Niche Frequency-Decoupled Streams with 5-dim pure multimodal context
            q_low = self.stream_proj(e_mm_sel + query)
            hc_vec = torch.cat([a_text, a_image, cos_ti, m_gate * g_m], dim=-1)
            if self.use_freq:
                stream_in = torch.cat([hc_vec, freq_ctx, q_low], dim=-1)
                phi_pop = l2_norm(self.pop_mlp(stream_in)) * STREAM_SCALE
                phi_niche = l2_norm(self.niche_mlp(stream_in)) * STREAM_SCALE
                delta_freq = p_t * self.W_POP(phi_pop) + nu_t * self.W_NICHE(phi_niche)
                if self.training:
                    delta_freq = (
                        delta_freq
                        + p_t * F.linear(phi_pop - phi_pop.detach(), self._probe_pop)
                        + nu_t * F.linear(phi_niche - phi_niche.detach(), self._probe_niche)
                    )
            else:
                stream_in = torch.cat([hc_vec, q_low], dim=-1)
                phi_pop = l2_norm(self.pop_mlp(stream_in)) * STREAM_SCALE
                phi_niche = torch.zeros_like(phi_pop)
                delta_freq = self.W_POP(phi_pop)
                if self.training:
                    delta_freq = delta_freq + F.linear(phi_pop - phi_pop.detach(), self._probe_pop)

            if self.training:
                delta_h = delta_h + F.linear(e_mm_sel - e_mm_sel.detach(), self._probe_mm)

            total_delta = delta_h + delta_freq
            telemetry = None
            if return_telemetry:
                telemetry = {
                    "delta_h": total_delta,
                    "pi_src": pi_src,
                    "pi_text": pi_t.squeeze(-1),
                    "pi_vis": pi_i.squeeze(-1),
                    "pi_kg": torch.zeros_like(pi_t.squeeze(-1)),
                    "g_chan": g_m,
                    "p_t": p_t.squeeze(-1),
                    "nu_t": nu_t.squeeze(-1),
                }
            return total_delta, telemetry

        # 1. Collaborative query & item modality/KG representations
        wq1, wq2, wq3 = self.query_projection.weight.chunk(3, dim=1)
        q_raw = (
            F.linear(h_ui, wq1, self.query_projection.bias)
            + F.linear(v_i, wq2)
            + F.linear(h_ui * v_i, wq3)
        )
        query = l2_norm(q_raw)
        text, image, kg_emb = self._project_item_modalities(items_flat)

        a_text = (query * text).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        a_image = (query * image).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        a_kg = (query * kg_emb).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        cos_ti = (text * image).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        cos_ik = (image * kg_emb).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

        d_text = (1.0 + a_text) / 2.0
        d_image = (1.0 + a_image) / 2.0
        d_kg = (1.0 + a_kg) / 2.0

        aligned_text = l2_norm(text + d_text * query)
        aligned_image = l2_norm(image + d_image * query)
        aligned_kg = l2_norm(kg_emb + d_kg * query)

        # 2. Extract raw and debiased/standardized context channels
        a_raw = ctx_dict["a"]
        cv_raw = ctx_dict["cv"]
        m_raw = ctx_dict["m_mean"]
        freq_ctx = ctx_dict["freq_ctx"]
        p_t = ctx_dict["p_t"].unsqueeze(-1)
        nu_t = ctx_dict["nu_t"].unsqueeze(-1)

        if self.mask_kg_info:
            a_kg = torch.zeros_like(a_kg)
            cos_ik = torch.zeros_like(cos_ik)
            aligned_kg = torch.zeros_like(aligned_kg)
            a_raw = torch.zeros_like(a_raw)
            cv_raw = torch.zeros_like(cv_raw)

        if self.use_debias:
            a_gate = ctx_dict["a_std"]
            cv_gate = ctx_dict["cv_std"]
            m_gate = ctx_dict["m_std"]
            c_lim = self.context.zscore_clip
            # Mean-sigma-preserved variance equalization: equalizes actor/director/keyword/genre to identical variance
            # while preserving the exact zero point and the global Euclidean scale for Adam(lr=0.001)
            mean_sig_a = self.context.sigma_a.mean()
            mean_sig_c = self.context.sigma_c.mean()
            mean_sig_m = self.context.sigma_m.mean()
            a_eff = (a_raw / (self.context.sigma_a + 1e-6)).clamp(-c_lim, c_lim) * mean_sig_a
            cv_eff = (cv_raw / (self.context.sigma_c + 1e-6)).clamp(-c_lim, c_lim) * mean_sig_c
            m_eff = m_gate * mean_sig_m
        else:
            a_gate, cv_gate, m_gate = a_raw, cv_raw, m_raw
            a_eff, cv_eff, m_eff = a_raw, cv_raw, m_raw

        # 3. Module 2 (M-Select): 19-dim History-Consistency Guided Selective Gate
        g_in = torch.cat([a_text, a_image, a_kg, cos_ti, cos_ik, m_gate, a_gate, cv_gate], dim=-1)
        gate_out = self.gate_net(g_in)
        src_logits = gate_out[:, :3]
        chan_logits = gate_out[:, 3:]

        if self.use_select:
            inv_d = 1.0 / float(self.hidden_dim)
            var_t = (2.0 - 2.0 * a_text) * inv_d
            var_i = (2.0 - 2.0 * a_image) * inv_d
            var_k = (2.0 - 2.0 * a_kg) * inv_d
            prior_logits = torch.cat([-var_t, -var_i, -var_k], dim=-1)
            pi_src = F.softmax((src_logits + prior_logits) / self.select_temperature, dim=-1)
            g_chan = 2.0 * torch.sigmoid(chan_logits)
        else:
            pi_src = torch.full((h_ui.size(0), 3), 1.0 / 3.0, dtype=h_ui.dtype, device=h_ui.device)
            g_chan = torch.ones((h_ui.size(0), 14), dtype=h_ui.dtype, device=h_ui.device)

        pi_t = pi_src[:, 0:1]
        pi_i = pi_src[:, 1:2]
        pi_k = pi_src[:, 2:3]
        e_mm_sel = l2_norm(pi_t * aligned_text + pi_i * aligned_image + pi_k * aligned_kg)

        g_a = g_chan[:, 0:4]
        g_c = g_chan[:, 4:12]
        g_m = g_chan[:, 12:14]

        a_sel = g_a * a_eff
        cv_sel = g_c * cv_eff
        m_sel = g_m * m_eff

        _, _, kg_raw_bank = self._active_banks()
        delta_h = (
            self.W_MM(e_mm_sel)
            + self.W_KG(kg_raw_bank[items_flat])
            + self.W_A(a_sel)
            + self.W_C(cv_sel)
            + self.W_M(m_sel)
        )

        # 4. Module 3 (M-Freq): Popular-Niche Frequency-Decoupled Nonlinear Streams
        q_low = self.stream_proj(e_mm_sel + query)
        hc_vec = torch.cat([a_text, a_image, a_kg, cos_ti, cos_ik, m_gate * g_m, a_gate * g_a, cv_gate * g_c], dim=-1)
        if self.use_freq:
            stream_in = torch.cat([hc_vec, freq_ctx, q_low], dim=-1)
            phi_pop = l2_norm(self.pop_mlp(stream_in)) * STREAM_SCALE
            phi_niche = l2_norm(self.niche_mlp(stream_in)) * STREAM_SCALE
            delta_freq = p_t * self.W_POP(phi_pop) + nu_t * self.W_NICHE(phi_niche)
            if self.training:
                delta_freq = (
                    delta_freq
                    + p_t * F.linear(phi_pop - phi_pop.detach(), self._probe_pop)
                    + nu_t * F.linear(phi_niche - phi_niche.detach(), self._probe_niche)
                )
        else:
            stream_in = torch.cat([hc_vec, q_low], dim=-1)
            phi_pop = l2_norm(self.pop_mlp(stream_in)) * STREAM_SCALE
            phi_niche = torch.zeros_like(phi_pop)
            delta_freq = self.W_POP(phi_pop)
            if self.training:
                delta_freq = delta_freq + F.linear(phi_pop - phi_pop.detach(), self._probe_pop)

        if self.training:
            delta_h = delta_h + F.linear(e_mm_sel - e_mm_sel.detach(), self._probe_mm)

        total_delta = delta_h + delta_freq
        telemetry = None
        if return_telemetry:
            telemetry = {
                "delta_h": total_delta,
                "pi_src": pi_src,
                "pi_text": pi_t.squeeze(-1),
                "pi_vis": pi_i.squeeze(-1),
                "pi_kg": pi_k.squeeze(-1),
                "g_chan": g_chan,
                "a_sel": a_sel,
                "cv_sel": cv_sel,
                "p_t": p_t.squeeze(-1),
                "nu_t": nu_t.squeeze(-1),
            }
        return total_delta, telemetry

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
        return_telemetry: bool = False,
        precomputed_ctx: Optional[Dict[str, torch.Tensor]] = None,
        precomputed_parent: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        device = next(self.parameters()).device
        users = torch.as_tensor(users, dtype=torch.long, device=device)
        items = torch.as_tensor(items, dtype=torch.long, device=device)

        if items.ndim == 1:
            if precomputed_parent is not None:
                parent_score, h_ui, v_i, semantic = precomputed_parent
            else:
                with torch.no_grad():
                    parent_score, h_ui, v_i, semantic = self._frozen_parent_flat(users, items, history_ids, history_mask)
            if self.is_null_ctrl:
                ctx_dict = {}
            elif precomputed_ctx is not None:
                ctx_dict = precomputed_ctx
            else:
                ctx_dict = self.context.forward_pairs(users, items, standardize=True)
            delta, telemetry = self._compute_delta_from_features(
                items, h_ui, v_i, semantic, ctx_dict, return_telemetry=return_telemetry
            )
            score = parent_score + self.parent.dnn_predict(delta).squeeze(-1)
            if not return_telemetry:
                return score
            telemetry["parent_score"] = parent_score
            return score, telemetry

        batch, candidates = items.shape
        items_flat = items.reshape(-1)
        if precomputed_parent is not None:
            parent_score, h_ui, v_i, semantic = precomputed_parent
        else:
            with torch.no_grad():
                parent_score, h_ui, v_i, semantic = self._frozen_parent_grid(users, items, history_ids, history_mask)

        if self.is_null_ctrl:
            ctx_dict = {}
        elif precomputed_ctx is not None:
            ctx_dict = precomputed_ctx
        elif candidates == N_ITEMS:
            grid_ctx = self.context.forward_grid(users, standardize=True)
            ctx_dict = {k: v.reshape(batch * candidates, *v.shape[2:]) for k, v in grid_ctx.items()}
        else:
            u_flat = users[:, None].expand(batch, candidates).reshape(-1)
            ctx_dict = self.context.forward_pairs(u_flat, items_flat, standardize=True)

        delta, telemetry = self._compute_delta_from_features(
            items_flat, h_ui, v_i, semantic, ctx_dict, return_telemetry=return_telemetry
        )
        score = (parent_score + self.parent.dnn_predict(delta).squeeze(-1)).reshape(batch, candidates)
        if not return_telemetry:
            return score
        telemetry["parent_score"] = parent_score.reshape(batch, candidates)
        for k, v in list(telemetry.items()):
            if isinstance(v, torch.Tensor) and v.shape[0] == batch * candidates:
                telemetry[k] = v.reshape(batch, candidates, *v.shape[1:])
        return score, telemetry


KG_SHUFFLE_PERM_SEED = 20240901


def get_canonical_kg_permutation(seed: int = KG_SHUFFLE_PERM_SEED) -> np.ndarray:
    """Generates a deterministic, fixed-point-free permutation (derangement) of N_ITEMS (9,031)
    using only catalog size N_ITEMS with zero Validation/Test label or interaction dependency.
    """
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(N_ITEMS).astype(np.int64)
        if not np.any(perm == np.arange(N_ITEMS)):
            return perm


def build_pamk_model(
    variant: str = "pamk_full",
    device: torch.device = torch.device("cuda:0"),
    parent: Optional[nn.Module] = None,
    context_engine: Optional[FrequencyAndEvidenceContext] = None,
) -> PAMKNet:
    from .firewall_and_assets import load_config, load_frozen_parent24, seed_everything

    cfg = load_config()
    hp = cfg["hyperparameters"]
    seed_everything(cfg["seed"])

    if parent is None:
        parent = load_frozen_parent24(device)

    kg_perm = get_canonical_kg_permutation() if variant == "pamk_kg_shuffled" else None
    if context_engine is None or (variant == "pamk_kg_shuffled" and getattr(context_engine, "kg_perm", None) is None):
        context_engine = FrequencyAndEvidenceContext(
            device=device,
            shrinkage_tau0=hp["shrinkage_tau0"],
            zscore_clip=hp["zscore_clip"],
            kg_perm=kg_perm,
        )

    text = torch.as_tensor(
        np.asarray(np.load(ASSET_PATHS["text_minilm"], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    image = torch.as_tensor(
        np.asarray(np.load(ASSET_PATHS["image_clip_image"], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    kg_cache = torch.load(
        ASSET_PATHS["corrected_relation_evidence_cache"], map_location=device, weights_only=True
    )
    rel_bank = kg_cache["relation_bank"]
    rel_mask = kg_cache["relation_mask"]
    if kg_perm is not None:
        perm_t = torch.as_tensor(kg_perm, dtype=torch.long, device=device)
        rel_bank = rel_bank[perm_t].contiguous()
        rel_mask = rel_mask[perm_t].contiguous()

    seed_everything(cfg["seed"])
    model = PAMKNet(
        parent=parent,
        text_bank=text,
        image_bank=image,
        relation_bank=rel_bank,
        relation_mask=rel_mask,
        context_engine=context_engine,
        variant=variant,
        hidden_dim=hp["hidden_dim"],
        stream_dim=hp["stream_dim"],
        gate_hidden_dim=hp["gate_hidden_dim"],
        select_temperature=hp["select_temperature"],
        graph_topk=hp["graph_topk"],
    ).to(device)
    return model
