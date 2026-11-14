"""Dedicated Python3.9 inference worker. JSON-lines IPC, no service or training.

One fixed Train research profile. Frozen Full factory, no scientific source edits.
Lifetime capped at 20 minutes, including load and idle time (below GPU budget).
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import resource
import sys
import time
import uuid

ROOT = Path('/home/daylight/下载/DjangoProject3_ch4_new/chapter4_m3l_author_aligned')
FORMAL = Path('/home/daylight/下载/DjangoProject3/DjangoProject3')
ORIGINAL = ROOT / '32_corrected_kg_final_candidate'
WORKTREE = Path(__file__).resolve().parents[1]
EVIDENCE = WORKTREE / 'docs/task58/evidence'
CHECKPOINT = ORIGINAL / '04_m3_paired_evidence/best.pt'
PARENT = ROOT / '20_canonical_skbhif_final/04_runs/seed2024/best.pt'
FULL_SHA = '73c7d95868005dae1a461d367a5afa0d3550c1a8379a9125764c4334dab2c7d6'
PARENT_SHA = 'b97feb824f427e6437f728459fbeb443af9b8bfdbba18ac33db8786007c33cd6'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def install_firewall(allowed):
    """No label data or unregistered research assets, including accidental loaders."""
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        name = p.name.lower()
        if name.startswith(('test_', 'x_label', 'kg_holdout_')) or name == 'interaction.inter':
            raise RuntimeError('DATA_BOUNDARY_REJECTED')
        protected = str(p).startswith(str(ROOT) + '/') or str(p).startswith(str(FORMAL) + '/')
        if protected:
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wa+')) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
                raise RuntimeError('RESEARCH_WRITE_REJECTED')
            if str(p) not in allowed and p.suffix not in ('.py', '.pyc'):
                raise RuntimeError('UNREGISTERED_RESEARCH_READ_REJECTED')
    sys.addaudithook(audit)


class Engine:
    def __init__(self):
        self.model = None
        self.load_count = 0

    def load(self):
        if self.model is not None:
            return
        started = time.monotonic()
        expected = json.loads((EVIDENCE / 'MODEL_INPUTS.json').read_text())
        if expected.get(str(CHECKPOINT)) != FULL_SHA or expected.get(str(PARENT)) != PARENT_SHA:
            raise RuntimeError('ONLY_ORIGINAL_FULL_AND_PARENT24_ALLOWED')
        install_firewall(set(expected))
        for path, identity in expected.items():
            if sha(path) != identity:
                raise RuntimeError('FROZEN_INPUT_IDENTITY_MISMATCH')
        config = json.loads((EVIDENCE / 'RUNTIME_CONFIG.json').read_text())
        if expected[str(CHECKPOINT)] != config['checkpoint_identity']:
            raise RuntimeError('CHECKPOINT_IDENTITY_MISMATCH')
        import numpy as np
        import torch
        from scipy.sparse import load_npz
        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU_UNAVAILABLE')
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats()
        sys.path.insert(0, str(ORIGINAL))
        import setup as frozen_setup
        from settings import CONFIG
        assert frozen_setup.PARENT == PARENT
        model = frozen_setup.build_model('full', self.device)
        # The residual includes parent parameters. Verify they remain Parent24.
        def state_digest(state):
            h = hashlib.sha256()
            for name, value in sorted(state.items()):
                h.update(name.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        parent_before = state_digest(model.parent.state_dict())
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        assert saved['variant'] == 'full' and saved['seed'] == 2024
        assert saved['config'] == CONFIG and CONFIG['learning_rate'] == .001
        assert saved['selection_metric'] == 'validation_recall@20' and saved['new_test_access'] == 0
        assert saved['science_freeze_sha'] == sha(frozen_setup.FREEZE)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        assert state_digest(model.parent.state_dict()) == parent_before
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            model.prepare_catalog()
        self.train = load_npz(ORIGINAL / '01_corrected_cache/TRAIN_CONTEXT_CSR.npz')
        self.history = np.load(ROOT / '20_canonical_skbhif_final/01_data/EVAL_TRAIN_HISTORY_IDS.npy', mmap_mode='r')
        assert self.train.shape == (69878, 9031) and self.history.shape == (69878, 20)
        self.config = config
        self.model = model
        self.load_count += 1
        torch.cuda.synchronize()
        self.load_seconds = time.monotonic() - started
        self.parent_digest = parent_before
        self.np, self.torch = np, torch

    def infer(self, request):
        # Validate BEFORE costly load. Only the fixed Train profile is admitted.
        cfg = json.loads((EVIDENCE / 'RUNTIME_CONFIG.json').read_text())
        if request.get('user') != cfg['profile_user']:
            raise ValueError('PROFILE_NOT_ADMITTED')
        k = request.get('top_k')
        if type(k) is not int or not 1 <= k <= 100:
            raise ValueError('INVALID_TOP_K')
        smoke = request.get('smoke', False)
        if type(smoke) is not bool:
            raise ValueError('INVALID_SCOPE')
        self.load()
        np, torch = self.np, self.torch
        started = time.monotonic()
        user = int(request['user'])
        seen = self.train.indices[self.train.indptr[user]:self.train.indptr[user + 1]]
        eligible = np.setdiff1d(np.arange(9031, dtype=np.int64), seen, assume_unique=True)
        assert len(seen) > 0
        if smoke:
            eligible = eligible[:8]
        hi = torch.as_tensor(self.history[user:user + 1].copy(), device=self.device, dtype=torch.long)
        hm = hi >= 0
        assert all(int(x) in seen for x in hi[hm].cpu().tolist())
        parts = []
        with torch.inference_mode():
            for start in range(0, len(eligible), 512):
                ids = eligible[start:start + 512]
                values = self.model(np.array([user]), ids[None, :], hi, hm).reshape(-1)
                if not torch.isfinite(values).all():
                    raise FloatingPointError('NONFINITE_MODEL_SCORE')
                parts.append(values.cpu().numpy())
        scores = np.concatenate(parts)
        order = np.argsort(-scores, kind='stable')[:k]
        torch.cuda.synchronize()
        receipt_id = str(uuid.uuid4())
        result = {
            'receipt_id': receipt_id, 'checkpoint_identity': self.config['checkpoint_identity'],
            'parent_identity': 'b97feb824f427e6437f728459fbeb443af9b8bfdbba18ac33db8786007c33cd6',
            'parent_state_digest': self.parent_digest, 'user': str(user),
            'items': [{'canonical_movie_id': str(int(eligible[i])), 'score': float(scores[i])} for i in order],
            'inference_mode': 'REAL_ONLINE_INFERENCE', 'catalogue_size': 9031,
            'train_seen_count': len(seen), 'candidate_count': len(eligible), 'smoke': smoke,
            'score_digest': hashlib.sha256(eligible.tobytes() + scores.tobytes()).hexdigest(),
            'score_seconds': time.monotonic() - started, 'load_seconds': self.load_seconds,
            'load_count': self.load_count, 'device': str(self.device),
            'worker_cpu_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'vram_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'vram_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'worker_pid': os.getpid(), 'NEW_TEST_ACCESS': 0}
        path = EVIDENCE / ('RECEIPT_' + receipt_id + '.json')
        path.write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('WORKER_LIFETIME_LIMIT')))
    signal.alarm(1200)
    wire = sys.stdout
    sys.stdout = sys.stderr  # Frozen imports cannot corrupt IPC.
    engine = Engine()
    for line in sys.stdin:
        if len(line) > 4096:
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
            # No retry after CUDA/identity/load failures; no recovery or fallback.
            return 1
        wire.write(json.dumps(result, allow_nan=False) + '\n')
        wire.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
