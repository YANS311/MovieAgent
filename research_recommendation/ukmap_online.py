"""Opt-in real Full backend. One process, serialized IPC, bounded lifetime.

The backend never imports torch in the Django process. No output-score cache:
each successful call executes checkpoint inference. The model process is reused.
"""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time

from .contracts import InferenceBatch, Prediction, RecommendationError, M3L

PYTHON = '/home/daylight/anaconda3/envs/DjangoProject3/bin/python'
WORKER = Path(__file__).with_name('ukmap_worker.py')
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/task58/evidence'


class UKMAPOnlineBackend:
    def __init__(self, spec, admitted_user, timeout=180):
        if spec.dataset != M3L or spec.name != 'UKMAP':
            raise RecommendationError('MODEL_DATASET_MISMATCH', 503)
        self.spec, self.admitted_user, self.timeout = spec, admitted_user, timeout
        self.lock = threading.Lock()
        self.process = None
        self.stderr = None
        self.started = None
        self.closed = False
        self.last_receipt = None
        atexit.register(self.close)

    @property
    def identity_key(self):
        # Reuse keys include both model and profile/catalogue/mapping versions.
        return (self.spec.checkpoint_identity, self.spec.train_manifest_identity,
                self.spec.catalogue_manifest_identity, self.spec.mapping_version,
                self.admitted_user)

    def _start(self):
        if self.closed:
            raise RecommendationError('MODEL_WORKER_UNAVAILABLE', 503)
        if self.process is not None:
            if self.process.poll() is not None:
                raise RecommendationError('MODEL_WORKER_UNAVAILABLE', 503)
            return
        checkpoint = Path('/home/daylight/下载/DjangoProject3_ch4_new/chapter4_m3l_author_aligned/32_corrected_kg_final_candidate/04_m3_paired_evidence/best.pt')
        if not checkpoint.is_file():
            raise RecommendationError('CHECKPOINT_UNAVAILABLE', 503)
        config = json.loads((EVIDENCE / 'RUNTIME_CONFIG.json').read_text())
        if (config['checkpoint_identity'] != self.spec.checkpoint_identity
                or config['profile_user'] != self.admitted_user
                or config['profile_manifest_sha256'] != self.spec.train_manifest_identity
                or config['catalogue_manifest_sha256'] != self.spec.catalogue_manifest_identity):
            raise RecommendationError('MODEL_CONFIGURATION_MISMATCH', 503)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        # Only a fixed local executable and trusted source, never request data.
        self.stderr = (EVIDENCE / 'WORKER_STDERR.txt').open('a')
        self.started = time.monotonic()
        self.process = subprocess.Popen([PYTHON, '-B', str(WORKER)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, env=env, cwd=ROOT,
            text=False, start_new_session=True)

    def call(self, user, top_k, smoke=False):
        if str(user) != self.admitted_user:
            raise RecommendationError('TRAIN_PROFILE_NOT_VERIFIED', 409)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise RecommendationError('INVALID_TOP_K')
        if not self.lock.acquire(timeout=2):
            raise RecommendationError('MODEL_BUSY', 503)
        try:
            self._start()
            if time.monotonic() - self.started > 1100:
                self._close_locked()
                raise RecommendationError('MODEL_SESSION_EXPIRED', 503)
            wire = json.dumps({'user': str(user), 'top_k': top_k, 'smoke': smoke}).encode() + b'\n'
            try:
                self.process.stdin.write(wire)
                self.process.stdin.flush()
                # Unbuffered read avoids hidden text buffers and has a hard deadline.
                deadline = time.monotonic() + self.timeout
                data = bytearray()
                while not data.endswith(b'\n'):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                        raise TimeoutError
                    block = os.read(self.process.stdout.fileno(), 65536)
                    if not block or len(data) + len(block) > 65536:
                        raise ValueError('INVALID_WORKER_WIRE')
                    data.extend(block)
                response = json.loads(data)
            except (OSError, ValueError, TimeoutError):
                self._close_locked()
                raise RecommendationError('MODEL_INFERENCE_UNAVAILABLE', 503) from None
            if not response.get('ok'):
                self._close_locked()
                raise RecommendationError('MODEL_INFERENCE_UNAVAILABLE', 503)
            result = response['result']
            if (result['checkpoint_identity'] != self.spec.checkpoint_identity
                    or result['user'] != self.admitted_user
                    or result['smoke'] is not smoke
                    or result['inference_mode'] != 'REAL_ONLINE_INFERENCE'):
                self._close_locked()
                raise RecommendationError('MODEL_PROVENANCE_MISMATCH', 502)
            self.last_receipt = result
            return result
        finally:
            self.lock.release()

    def infer(self, profile, top_k):
        if profile.dataset != M3L:
            raise RecommendationError('PROFILE_DATASET_MISMATCH', 409)
        r = self.call(profile.canonical_user_id, top_k)
        return InferenceBatch('UKMAP', M3L, self.spec.checkpoint_identity,
            profile.canonical_user_id,
            tuple(Prediction(x['canonical_movie_id'], x['score']) for x in r['items']),
            'REAL_ONLINE_INFERENCE',
            f"FULL_TRAIN_UNSEEN:{r['candidate_count']}/{r['catalogue_size']};filtered={r['train_seen_count']}",
            'Task58:RECEIPT_' + r['receipt_id'], None,
            self.spec.train_manifest_identity, self.spec.catalogue_manifest_identity)

    def _close_locked(self):
        self.closed = True
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write(b'{"action":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                # Only this backend's child is signalled. No scientific recovery.
                if self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            finally:
                for stream in (self.process.stdin, self.process.stdout, self.stderr):
                    if stream is not None:
                        stream.close()
        elif self.stderr is not None:
            self.stderr.close()

    def close(self):
        with self.lock:
            self._close_locked()
