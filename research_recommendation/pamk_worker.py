"""Dedicated Python 3.9 inference worker for PAMK-Net. JSON-lines IPC, no service or training.

Multiple authorized Train research profiles; authorization is enforced by Django. Frozen Parent24 + PAMK-Net, no scientific source edits.
Lifetime capped at 20 minutes, including load and idle time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import uuid

WORKTREE=Path(__file__).resolve().parents[1]
CH4_ROOT=WORKTREE/'scientific_pamk'
ROOT=CH4_ROOT
FORMAL=CH4_ROOT
EVIDENCE=Path(os.environ['PAMK_EVIDENCE_DIR'])
from pamk_identity import CHECKPOINT as PAMK_CHECKPOINT, CHECKPOINT_SHA256 as PAMK_FULL_SHA
_cfg=json.loads(Path(os.environ['PAMK_ASSET_CONFIG']).read_text())
_ASSET_CONFIG_SHA=hashlib.sha256(Path(os.environ['PAMK_ASSET_CONFIG']).read_bytes()).hexdigest()
PARENT=Path(_cfg['assets']['parent24_checkpoint']['path'])
PARENT_SHA='b97feb824f427e6437f728459fbeb443af9b8bfdbba18ac33db8786007c33cd6'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def install_firewall(allowed):
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        name = p.name.lower()
        if name.startswith(('test_', 'x_label', 'kg_holdout_')) or name == 'interaction.inter':
            raise RuntimeError('DATA_BOUNDARY_REJECTED')
        protected = str(p).startswith(str(ROOT) + '/') or str(p).startswith(str(FORMAL) + '/') or str(p).startswith(str(CH4_ROOT) + '/')
        if protected:
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wa+')) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
                raise RuntimeError('RESEARCH_WRITE_REJECTED')
            if str(p) not in allowed and p.suffix not in ('.py', '.pyc', '.json'):
                raise RuntimeError('UNREGISTERED_RESEARCH_READ_REJECTED:' + str(p))
    sys.addaudithook(audit)


class PAMKEngine:
    def __init__(self):
        self.model = None
        self.load_count = 0
        self.inference_count = 0

    def load(self):
        if self.model is not None:
            return
        started = time.monotonic()
        if sha(PAMK_CHECKPOINT) != PAMK_FULL_SHA or sha(PARENT) != PARENT_SHA:
            raise RuntimeError('ONLY_ORIGINAL_PAMK_FULL_AND_PARENT24_ALLOWED')

        import numpy as np
        import torch
        from scipy.sparse import load_npz

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU_UNAVAILABLE')
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats()

        sys.path.insert(0,str(CH4_ROOT))
        from src.firewall_and_assets import load_frozen_parent24, ASSET_PATHS
        from src.context_engine import FrequencyAndEvidenceContext
        from src.pamk_model import build_pamk_model

        from src.firewall_and_assets import load_config
        expected = load_config()['frozen_sha256']
        used_keys = ('train_context_csr', 'eval_train_history_ids', 'corrected_relation_evidence_cache', 'entity_kernels', 'train_background', 'mm_similarities', 'text_minilm', 'image_clip_image', 'genre_item_ids', 'genre_item_mask')
        self.asset_sha256 = {}
        for key in used_keys:
            actual = sha(ASSET_PATHS[key])
            if actual != expected[key]:
                raise RuntimeError('FEATURE_ASSET_SHA_MISMATCH:' + key)
            self.asset_sha256[key] = actual
        sources = {'pamk_model.py':'8d5d3381d93e425314894737a5c8e838623ee76363066e464d52ffba9588a1e1', 'context_engine.py':'9406e58aaf69af5fd7bf29fcbb20aba5b7b34e83780f6faac11b677120d07faa'}
        for name, expected_sha in sources.items():
            if sha(CH4_ROOT / 'src' / name) != expected_sha:
                raise RuntimeError('MODEL_SOURCE_SHA_MISMATCH:' + name)
        if sha(CH4_ROOT/'canonical_skbhif_final.py') != _cfg['parent_source_sha256']: raise RuntimeError('PARENT_SOURCE_SHA_MISMATCH')
        parent = load_frozen_parent24(self.device)
        ctx_engine = FrequencyAndEvidenceContext(device=self.device)
        model = build_pamk_model('pamk_full', device=self.device, parent=parent, context_engine=ctx_engine)

        saved = torch.load(PAMK_CHECKPOINT, map_location=self.device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            model.prepare_catalog()

        self.train = load_npz(ASSET_PATHS['train_context_csr'])
        self.history = np.load(ASSET_PATHS['eval_train_history_ids'], mmap_mode='r')
        assert self.train.shape == (69878, 9031) and self.history.shape == (69878, 20)

        self.model = model
        self.load_count += 1
        torch.cuda.synchronize()
        self.load_seconds = time.monotonic() - started
        self.np, self.torch = np, torch

    def infer(self, request):
        k = request.get('top_k', 20)
        if type(k) is not int or not 1 <= k <= 100:
            raise ValueError('INVALID_TOP_K')
        smoke = request.get('smoke', False)
        if type(smoke) is not bool:
            raise ValueError('INVALID_SCOPE')

        self.load()
        if sha(Path(os.environ['PAMK_ASSET_CONFIG'])) != _ASSET_CONFIG_SHA:
            raise RuntimeError('ASSET_CONFIGURATION_CHANGED_RESTART_REQUIRED')
        if request.get('action') == 'inductive':
            from inductive_runtime import InductiveRuntime
            if not hasattr(self, 'inductive_runtime'):
                self.inductive_runtime = InductiveRuntime(self)
            return self.inductive_runtime.infer(request)
        np, torch = self.np, self.torch
        started = time.monotonic()
        raw_user = request['user']
        if isinstance(raw_user, bool) or not str(raw_user).isdecimal() or not 0 <= int(raw_user) < 69878:
            raise ValueError('INVALID_CANONICAL_USER')
        user = int(raw_user)
        seen = self.train.indices[self.train.indptr[user]:self.train.indptr[user + 1]]
        eligible = np.setdiff1d(np.arange(9031, dtype=np.int64), seen, assume_unique=True)
        assert len(seen) > 0
        if smoke:
            eligible = eligible[:8]

        hi = torch.as_tensor(self.history[user:user + 1].copy(), device=self.device, dtype=torch.long)
        hm = hi >= 0
        ut = torch.as_tensor([user], device=self.device, dtype=torch.long)
        it = torch.as_tensor(eligible[None, :], device=self.device, dtype=torch.long)

        gpu_begin, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gpu_begin.record()
        with torch.inference_mode():
            scores_tensor, telem = self.model(ut, it, hi, hm, return_telemetry=True)
        gpu_end.record()

        scores = scores_tensor.cpu().numpy().flatten()
        if not np.isfinite(scores).all():
            raise RuntimeError('NONFINITE_MODEL_SCORES')
        full_order = np.argsort(-scores, kind='stable')
        excluded = request.get('exclude_canonical_ids', [])
        if not isinstance(excluded, list) or any(type(x) is not int or not 0 <= x < 9031 for x in excluded):
            raise ValueError('INVALID_BUSINESS_EXCLUSIONS')
        excluded = set(excluded)
        original_ranks = {int(i): r + 1 for r, i in enumerate(full_order)}
        order = np.asarray([i for i in full_order if int(eligible[i]) not in excluded][:k], dtype=np.int64)
        torch.cuda.synchronize()

        top_indices = order
        top_eligible = eligible[top_indices]
        top_scores = scores[top_indices]

        # Extract per-item telemetry for returned recommendations
        pi_text = telem['pi_text'].cpu().numpy().flatten()[top_indices].tolist()
        pi_vis = telem['pi_vis'].cpu().numpy().flatten()[top_indices].tolist()
        pi_kg = telem['pi_kg'].cpu().numpy().flatten()[top_indices].tolist()
        p_t = telem['p_t'].cpu().numpy().flatten()[top_indices].tolist()
        nu_t = telem['nu_t'].cpu().numpy().flatten()[top_indices].tolist()
        p_score = telem['parent_score'].cpu().numpy().flatten()[top_indices].tolist()

        receipt_id = str(uuid.uuid4())
        items = []
        for rank_i, (cid, sc) in enumerate(zip(top_eligible, top_scores)):
            items.append({
                'canonical_movie_id': str(int(cid)),
                'score': float(sc),
                'original_model_rank': original_ranks[int(top_indices[rank_i])],
                'telemetry': {
                    'pi_text': float(pi_text[rank_i]),
                    'pi_vis': float(pi_vis[rank_i]),
                    'pi_kg': float(pi_kg[rank_i]),
                    'popularity': float(p_t[rank_i]),
                    'niche_specificity': float(nu_t[rank_i]),
                    'parent_score': float(p_score[rank_i]),
                }
            })

        self.inference_count += 1
        result = {
            'receipt_id': receipt_id,
            'model_name': 'PAMK-Net',
            'checkpoint_identity': PAMK_FULL_SHA,
            'asset_config_sha256': _ASSET_CONFIG_SHA,
            'parent_identity': PARENT_SHA,
            'user': str(user),
            'items': items,
            'inference_mode': 'REAL_ONLINE_INFERENCE',
            'catalogue_size': 9031,
            'train_seen_count': len(seen),
            'candidate_count': len(eligible),
            'canonical_zero_eligible': bool(0 in eligible),
            'all_scores_finite': bool(np.isfinite(scores).all()),
            'smoke': smoke,
            'score_digest': hashlib.sha256(eligible.tobytes() + scores.tobytes()).hexdigest(),
            'score_seconds': time.monotonic() - started,
            'gpu_forward_ms': gpu_begin.elapsed_time(gpu_end),
            'load_seconds': self.load_seconds,
            'load_count': self.load_count,
            'inference_count': self.inference_count,
            'business_excluded_count': len(excluded),
            'device': str(self.device),
            'worker_cpu_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'vram_peak_allocated_mib': torch.cuda.max_memory_allocated() / (1 << 20),
            'vram_peak_reserved_mib': torch.cuda.max_memory_reserved() / (1 << 20),
            'worker_pid': os.getpid(),
            'feature_sha256': self.asset_sha256,
            'NEW_TEST_ACCESS': 0,
        }

        path = EVIDENCE / ('RECEIPT_PAMK_' + receipt_id + '.json')
        try:
            path.write_text(json.dumps(result, indent=2) + '\n')
        except Exception:
            pass
        return result


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('WORKER_LIFETIME_LIMIT')))
    signal.alarm(1200)
    wire = sys.stdout
    sys.stdout = sys.stderr
    if os.environ.get('NEW_TEST_ACCESS', '0') != '0':
        raise RuntimeError('TEST_ACCESS_FORBIDDEN')
    def deny_test(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(args[0])).name.lower()
            if (name.startswith(('test_', 'x_label', 'kg_holdout_')) and Path(name).suffix in ('.npy','.npz','.csv','.pt','.pth','.pkl','.parquet')) or name in ('interaction.inter', 'validation_user.npy', 'validation_item.npy'):
                raise RuntimeError('TEST_OR_VALIDATION_LABEL_ACCESS_FORBIDDEN')
    sys.addaudithook(deny_test)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    engine = PAMKEngine()
    for line in sys.stdin:
        if len(line) > 65536:
            break
        try:
            request = json.loads(line)
            if request.get('action') == 'close':
                break
            result = {'ok': True, 'result': engine.infer(request)}
        except Exception as exc:
            result = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)[:200]}
            wire.write(json.dumps(result) + '\n')
            wire.flush()
            return 1
        wire.write(json.dumps(result, allow_nan=False) + '\n')
        wire.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
