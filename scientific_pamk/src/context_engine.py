"""Target-Excluded Multimodal-KG & Popularity Context Engine (1D Pair & 2D Grid Modes)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .firewall_and_assets import ASSET_PATHS, N_ITEMS, N_USERS


class FrequencyAndEvidenceContext(nn.Module):
    """Computes target-excluded KG kernels (a), paired multimodal-KG co-evidence (cv),
    unpaired history-target modality means (m_mean), and popularity-niche context (freq_ctx).

    Supports:
    - `forward_pairs(users, targets)`: exact chunk-wise gather for 1D training pairs `(B,)`.
    - `forward_grid(users)`: fast exact GPU matrix-subtraction over all `N_ITEMS` for `(B_u, N_ITEMS)` evaluation.
    """

    def __init__(
        self,
        device: torch.device,
        shrinkage_tau0: float = 20.0,
        zscore_clip: float = 5.0,
        stats_seed: int = 322024,
        stats_sample_size: int = 32768,
        enable_grid_cache: bool = True,
        kg_perm: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.shrinkage_tau0 = float(shrinkage_tau0)
        self.zscore_clip = float(zscore_clip)
        self.enable_grid_cache = bool(enable_grid_cache)
        self.kg_perm = None if kg_perm is None else np.asarray(kg_perm, dtype=np.int64).copy()

        # 1. Load Train CSR context
        csr = sp.load_npz(ASSET_PATHS["train_context_csr"]).tocsr()
        assert csr.shape == (N_USERS, N_ITEMS), f"Unexpected CSR shape: {csr.shape}"
        self.register_buffer("indptr", torch.as_tensor(csr.indptr.astype(np.int64), device=device), persistent=False)
        self.register_buffer("indices", torch.as_tensor(csr.indices.astype(np.int64), device=device), persistent=False)

        # 2. Load KG kernels (4, N_ITEMS, N_ITEMS), background (N_ITEMS, 4), MM similarities (2, N_ITEMS, N_ITEMS)
        kernels_np = np.load(ASSET_PATHS["entity_kernels"]).astype(np.float32)
        bg_np = np.load(ASSET_PATHS["train_background"]).astype(np.float32)
        mm_sim_np = np.load(ASSET_PATHS["mm_similarities"]).astype(np.float32)

        if self.kg_perm is not None:
            assert self.kg_perm.shape == (N_ITEMS,), f"Expected kg_perm shape ({N_ITEMS},), got {self.kg_perm.shape}"
            assert np.array_equal(np.sort(self.kg_perm), np.arange(N_ITEMS)), "kg_perm must be a valid bijection"
            assert not np.any(self.kg_perm == np.arange(N_ITEMS)), "kg_perm must be a strict derangement (0 fixed points)"
            kernels_np = np.ascontiguousarray(kernels_np[:, self.kg_perm, :][:, :, self.kg_perm])
            bg_np = np.ascontiguousarray(bg_np[self.kg_perm])

        self.register_buffer("kernels", torch.as_tensor(kernels_np, device=device), persistent=False)
        self.register_buffer("bg", torch.as_tensor(bg_np, device=device), persistent=False)
        self.register_buffer("mm_sim", torch.as_tensor(mm_sim_np, device=device), persistent=False)

        # Precompute 8 paired product matrices (K_r * S_m) of shape (8, N_ITEMS, N_ITEMS) for fast 2D grid matmuls
        with torch.no_grad():
            km_prod = (self.kernels.unsqueeze(1) * self.mm_sim.unsqueeze(0)).reshape(8, N_ITEMS, N_ITEMS).contiguous()
        self.register_buffer("km_kernels", km_prod, persistent=False)

        # 3. Compute Train-only item popularity p_i and niche specificity nu_i = 1 - p_i
        train_items = np.load(ASSET_PATHS["train_item"], mmap_mode="r")
        item_counts = np.bincount(np.asarray(train_items, dtype=np.int64), minlength=N_ITEMS).astype(np.float64)
        c_max = float(item_counts.max())
        item_pop_np = (np.log1p(item_counts) / np.log1p(c_max)).astype(np.float32)
        item_niche_np = (1.0 - item_pop_np).astype(np.float32)
        self.item_counts_np = item_counts
        self.register_buffer("item_pop", torch.as_tensor(item_pop_np, device=device), persistent=False)
        self.register_buffer("item_niche", torch.as_tensor(item_niche_np, device=device), persistent=False)

        # Diagonals for exact target exclusion in 2D grid mode
        idx_all = torch.arange(N_ITEMS, device=device)
        self.register_buffer("kernels_diag", self.kernels[:, idx_all, idx_all].contiguous(), persistent=False)
        self.register_buffer("mm_diag", self.mm_sim[:, idx_all, idx_all].contiguous(), persistent=False)
        self.register_buffer("km_diag", self.km_kernels[:, idx_all, idx_all].contiguous(), persistent=False)

        # 4. Estimate channel Z-score statistics on deterministic Train sample (seed=322024)
        self.register_buffer("mu_a", torch.zeros(4, dtype=torch.float32, device=device))
        self.register_buffer("sigma_a", torch.ones(4, dtype=torch.float32, device=device))
        self.register_buffer("mu_c", torch.zeros(8, dtype=torch.float32, device=device))
        self.register_buffer("sigma_c", torch.ones(8, dtype=torch.float32, device=device))
        self.register_buffer("mu_m", torch.zeros(2, dtype=torch.float32, device=device))
        self.register_buffer("sigma_m", torch.ones(2, dtype=torch.float32, device=device))
        self._fit_zscore_buffers(seed=stats_seed, sample_size=stats_sample_size)

    @torch.no_grad()
    def _fit_zscore_buffers(self, seed: int, sample_size: int) -> None:
        train_u = np.load(ASSET_PATHS["train_user"], mmap_mode="r")
        train_i = np.load(ASSET_PATHS["train_item"], mmap_mode="r")
        n_total = int(train_u.shape[0])
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_total, size=min(sample_size, n_total), replace=False)
        u_t = torch.as_tensor(np.asarray(train_u[sel], dtype=np.int64), device=self.device)
        i_t = torch.as_tensor(np.asarray(train_i[sel], dtype=np.int64), device=self.device)
        out = self.forward_pairs(u_t, i_t, standardize=False)
        for key, mu_buf, sig_buf in [
            ("a", self.mu_a, self.sigma_a),
            ("cv", self.mu_c, self.sigma_c),
            ("m_mean", self.mu_m, self.sigma_m),
        ]:
            val = out[key]
            mu_buf.copy_(val.mean(dim=0))
            sig_buf.copy_(val.std(dim=0, unbiased=False).clamp_min(1e-6))

    def standardize_channels(
        self, a: torch.Tensor, cv: torch.Tensor, m_mean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.zscore_clip
        a_std = ((a - self.mu_a) / (self.sigma_a + 1e-6)).clamp(-c, c)
        cv_std = ((cv - self.mu_c) / (self.sigma_c + 1e-6)).clamp(-c, c)
        m_std = ((m_mean - self.mu_m) / (self.sigma_m + 1e-6)).clamp(-c, c)
        return a_std, cv_std, m_std

    @torch.no_grad()
    def forward_pairs(
        self, users: torch.Tensor, targets: torch.Tensor, chunk_size: int = 64, standardize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Exact 1D pair context computation over (users, targets) of shape (B,)."""
        b_size = int(users.numel())
        starts = self.indptr[users]
        ends = self.indptr[users + 1]
        lengths = ends - starts
        max_len = int(lengths.max().item()) if b_size > 0 else 0

        k_sum = torch.zeros((b_size, 4), dtype=torch.float32, device=self.device)
        m_sum = torch.zeros((b_size, 2), dtype=torch.float32, device=self.device)
        km_sum = torch.zeros((b_size, 4, 2), dtype=torch.float32, device=self.device)
        pop_sum = torch.zeros(b_size, dtype=torch.float32, device=self.device)
        pop_sq_sum = torch.zeros(b_size, dtype=torch.float32, device=self.device)
        valid_cnt = torch.zeros(b_size, dtype=torch.float32, device=self.device)

        for offset in range(0, max_len, chunk_size):
            pos = starts.unsqueeze(1) + torch.arange(
                offset, min(offset + chunk_size, max_len), device=self.device
            ).unsqueeze(0)
            valid = pos < ends.unsqueeze(1)
            hist = self.indices[pos.clamp(max=self.indices.numel() - 1)]
            valid = valid & (hist != targets.unsqueeze(1))
            hist_safe = hist.clamp(0, N_ITEMS - 1)
            mask = valid.to(torch.float32)

            k = self.kernels[:, targets.unsqueeze(1), hist_safe].permute(1, 2, 0) * mask.unsqueeze(-1)
            s = self.mm_sim[:, targets.unsqueeze(1), hist_safe].permute(1, 2, 0) * mask.unsqueeze(-1)
            p_h = self.item_pop[hist_safe] * mask

            k_sum += k.sum(dim=1)
            m_sum += s.sum(dim=1)
            km_sum += (k.unsqueeze(-1) * s.unsqueeze(-2)).sum(dim=1)
            pop_sum += p_h.sum(dim=1)
            pop_sq_sum += (p_h * p_h).sum(dim=1)
            valid_cnt += mask.sum(dim=1)

        denom = valid_cnt.clamp_min(1.0).unsqueeze(1)
        s_raw = k_sum / denom
        shrink = (valid_cnt / (valid_cnt + self.shrinkage_tau0)).unsqueeze(1)
        a = shrink * (s_raw - self.bg[targets])
        m_mean = m_sum / denom
        cv = (km_sum / denom.unsqueeze(-1) - s_raw.unsqueeze(-1) * m_mean.unsqueeze(-2)).reshape(b_size, 8)

        p_t = self.item_pop[targets]
        nu_t = self.item_niche[targets]
        mu_u = pop_sum / valid_cnt.clamp_min(1.0)
        var_u = (pop_sq_sum / valid_cnt.clamp_min(1.0) - mu_u * mu_u).clamp_min(0.0)
        sig_u = torch.sqrt(var_u + 1e-8)
        log_deg = torch.log1p(valid_cnt) / float(np.log1p(500.0))
        freq_ctx = torch.stack([p_t, nu_t, mu_u, sig_u, p_t - mu_u, log_deg], dim=-1)

        if standardize:
            a_std, cv_std, m_std = self.standardize_channels(a, cv, m_mean)
        else:
            a_std, cv_std, m_std = a, cv, m_mean

        return {
            "a": a,
            "cv": cv,
            "m_mean": m_mean,
            "a_std": a_std,
            "cv_std": cv_std,
            "m_std": m_std,
            "freq_ctx": freq_ctx,
            "p_t": p_t,
            "nu_t": nu_t,
            "valid_cnt": valid_cnt,
        }

    @torch.no_grad()
    def forward_grid(self, users: torch.Tensor, standardize: bool = True) -> Dict[str, torch.Tensor]:
        """Fast exact 2D grid context computation over all N_ITEMS for users of shape (B_u,).
        Uses pure GPU scatter_add_ and pre-packed symmetric matrix multiplications:
            sum_{j in H(u) \\ {i}} F(i, j) = (H_u @ F)_i - I(i in H(u)) * F(i, i)
        Returns tensors of shape (B_u, N_ITEMS, D).
        """
        bu = int(users.numel())
        starts = self.indptr[users]
        ends = self.indptr[users + 1]
        max_len = int((ends - starts).max().item()) if bu > 0 else 0
        h_mat = torch.zeros((bu, N_ITEMS), dtype=torch.float32, device=self.device)
        if max_len > 0:
            pos = starts.unsqueeze(1) + torch.arange(max_len, device=self.device).unsqueeze(0)
            valid = pos < ends.unsqueeze(1)
            hist = self.indices[pos.clamp(max=self.indices.numel() - 1)].clamp(0, N_ITEMS - 1)
            h_mat.scatter_add_(1, hist, valid.to(torch.float32))
            h_mat.clamp_(max=1.0)

        raw_deg = h_mat.sum(dim=1, keepdim=True)  # (B_u, 1)
        valid_cnt = (raw_deg - h_mat).clamp_min(0.0)  # (B_u, N_ITEMS)
        denom = valid_cnt.clamp_min(1.0)  # (B_u, N_ITEMS)

        # 1. Kernels (symmetric): (B_u, N_ITEMS, 4)
        k_sum = torch.stack(
            [torch.matmul(h_mat, self.kernels[r]) - h_mat * self.kernels_diag[r].unsqueeze(0) for r in range(4)],
            dim=-1,
        )

        # 2. MM similarities (symmetric): (B_u, N_ITEMS, 2)
        m_sum = torch.stack(
            [torch.matmul(h_mat, self.mm_sim[m]) - h_mat * self.mm_diag[m].unsqueeze(0) for m in range(2)],
            dim=-1,
        )

        # 3. Paired (K_r * S_m) (symmetric): (B_u, N_ITEMS, 4, 2)
        km_sum = torch.stack(
            [
                torch.matmul(h_mat, self.km_kernels[idx]) - h_mat * self.km_diag[idx].unsqueeze(0)
                for idx in range(8)
            ],
            dim=-1,
        ).view(bu, N_ITEMS, 4, 2)

        s_raw = k_sum / denom.unsqueeze(-1)
        shrink = (valid_cnt / (valid_cnt + self.shrinkage_tau0)).unsqueeze(-1)
        a = shrink * (s_raw - self.bg.unsqueeze(0))
        m_mean = m_sum / denom.unsqueeze(-1)
        cv = (km_sum / denom.unsqueeze(-1).unsqueeze(-1) - s_raw.unsqueeze(-1) * m_mean.unsqueeze(-2)).reshape(
            bu, N_ITEMS, 8
        )

        # 4. Frequency context (B_u, N_ITEMS, 6)
        p_all = self.item_pop.unsqueeze(0).expand(bu, N_ITEMS)
        nu_all = self.item_niche.unsqueeze(0).expand(bu, N_ITEMS)
        pop_tot = torch.matmul(h_mat, self.item_pop.unsqueeze(-1))
        pop_sq_tot = torch.matmul(h_mat, (self.item_pop * self.item_pop).unsqueeze(-1))
        pop_sum = pop_tot - h_mat * self.item_pop.unsqueeze(0)
        pop_sq_sum = pop_sq_tot - h_mat * (self.item_pop * self.item_pop).unsqueeze(0)

        mu_u = pop_sum / denom
        var_u = (pop_sq_sum / denom - mu_u * mu_u).clamp_min(0.0)
        sig_u = torch.sqrt(var_u + 1e-8)
        log_deg = torch.log1p(valid_cnt) / float(np.log1p(500.0))
        freq_ctx = torch.stack([p_all, nu_all, mu_u, sig_u, p_all - mu_u, log_deg], dim=-1)

        if standardize:
            a_std, cv_std, m_std = self.standardize_channels(a, cv, m_mean)
        else:
            a_std, cv_std, m_std = a, cv, m_mean

        return {
            "a": a,
            "cv": cv,
            "m_mean": m_mean,
            "a_std": a_std,
            "cv_std": cv_std,
            "m_std": m_std,
            "freq_ctx": freq_ctx,
            "p_t": p_all,
            "nu_t": nu_all,
            "valid_cnt": valid_cnt,
        }
