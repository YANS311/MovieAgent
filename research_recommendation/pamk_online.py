"""Opt-in real PAMK-Net backend. One process, serialized IPC, bounded lifetime.

The backend never imports torch in the Django process.
Each successful call executes checkpoint inference in an isolated worker.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import select
import subprocess
import threading
import time

from .contracts import InferenceBatch, Prediction, RecommendationError, M3L

PYTHON = os.environ['PAMK_WORKER_PYTHON']
WORKER = Path(__file__).with_name('pamk_worker.py')
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('PAMK_EVIDENCE_DIR', str(ROOT.parent / 'chapter5_pamk_real_integration_20261112/evidence')))
from .pamk_identity import CHECKPOINT as PAMK_CHECKPOINT, CHECKPOINT_SHA256 as PAMK_SHA


class PAMKOnlineBackend:
    def __init__(self, spec, admitted_user='0', timeout=None):
        if spec.dataset != M3L or spec.name != 'PAMK-Net':
            raise RecommendationError('MODEL_DATASET_MISMATCH', 503)
        self.spec = spec
        self.admitted_user = None if admitted_user is None else str(admitted_user)
        self.timeout = float(os.environ.get('PAMK_WORKER_TIMEOUT','180')) if timeout is None else timeout
        self.lock = threading.Lock()
        self.process = None
        self.stderr = None
        self.started = None
        self.closed = False
        self.last_receipt = None
        atexit.register(self.close)

    @property
    def identity_key(self):
        return (
            self.spec.checkpoint_identity,
            self.spec.train_manifest_identity,
            self.spec.catalogue_manifest_identity,
            self.spec.mapping_version,
            self.admitted_user,
        )

    def _start(self):
        if self.closed:
            raise RecommendationError('MODEL_WORKER_UNAVAILABLE', 503)
        if self.process is not None:
            if self.process.poll() is not None:
                raise RecommendationError('MODEL_WORKER_UNAVAILABLE', 503)
            return
        if not PAMK_CHECKPOINT.is_file():
            raise RecommendationError('CHECKPOINT_UNAVAILABLE', 503)

        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES='0',
            PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',
        )
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.stderr = (EVIDENCE / 'PAMK_WORKER_STDERR.txt').open('a')
        self.started = time.monotonic()
        try:
            self.process = subprocess.Popen(
                [PYTHON, '-B', str(WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                env=env,
                cwd=ROOT,
                text=False,
                start_new_session=True,
            )
        except OSError:
            self._close_locked()
            raise RecommendationError('MODEL_WORKER_UNAVAILABLE', 503) from None

    def call(self, user, top_k, smoke=False, exclude_canonical_ids=None, history=None, context_identity=None, adapter_identity=None, encoding_method="learned"):
        if self.admitted_user is not None and str(user) != self.admitted_user:
            raise RecommendationError('TRAIN_PROFILE_NOT_VERIFIED', 409)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise RecommendationError('INVALID_TOP_K')
        if not self.lock.acquire(timeout=5):
            raise RecommendationError('MODEL_BUSY', 503)
        try:
            self._start()
            if time.monotonic() - self.started > 1100:
                self._close_locked()
                raise RecommendationError('MODEL_SESSION_EXPIRED', 503)
            request = {'user':str(user), 'top_k':top_k, 'smoke':smoke, 'exclude_canonical_ids':exclude_canonical_ids or []}
            if history is not None:
                request.update(action='inductive', history=history, context_identity=context_identity, adapter_identity=adapter_identity, encoding_method=encoding_method)
            wire = json.dumps(request).encode() + b'\n'
            try:
                self.process.stdin.write(wire)
                self.process.stdin.flush()
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
                logging.getLogger(__name__).error('PAMK worker rejected request: %s', response.get('error', 'unknown'))
                self._close_locked()
                raise RecommendationError('MODEL_INFERENCE_UNAVAILABLE', 503)
            result = response['result']
            if (
                result['checkpoint_identity'] != self.spec.checkpoint_identity
                or result['user'] != str(user)
                or result['smoke'] is not smoke
                or result['inference_mode'] != 'REAL_ONLINE_INFERENCE'
                or (history is not None and result.get('encoding_identity') != adapter_identity)
            ):
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
        return InferenceBatch(
            'PAMK-Net',
            M3L,
            self.spec.checkpoint_identity,
            profile.canonical_user_id,
            tuple(Prediction(x['canonical_movie_id'], x['score']) for x in r['items']),
            'REAL_ONLINE_INFERENCE',
            f"FULL_TRAIN_UNSEEN:{r['candidate_count']}/{r['catalogue_size']};filtered={r['train_seen_count']}",
            'Task84:RECEIPT_' + r['receipt_id'],
            None,
            self.spec.train_manifest_identity,
            self.spec.catalogue_manifest_identity,
        )

    def _close_locked(self):
        self.closed = True
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write(b'{"action":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
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
                        try:
                            stream.close()
                        except Exception:
                            pass
        elif self.stderr is not None:
            try:
                self.stderr.close()
            except Exception:
                pass

    def close(self):
        with self.lock:
            self._close_locked()
