"""CPU-only negative/contract tests. Do not start worker, CUDA, network or DB."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from research_recommendation.contracts import ModelSpec, UKMAPAdapter, M3L, ML1M, RecommendationError
from research_recommendation.ukmap_online import UKMAPOnlineBackend


class OnlineBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.spec = ModelSpec('UKMAP', M3L, 'a'*64, 'test-only-version', 'b'*64, 'c'*64)
        self.backend = UKMAPOnlineBackend(self.spec, '0', timeout=1)
        self.profile = SimpleNamespace(canonical_user_id='0', dataset=M3L)

    def tearDown(self):
        self.backend.close()

    def test_reject_other_profile_before_loading(self):
        with patch.object(self.backend, '_start') as start:
            with self.assertRaisesRegex(RecommendationError, 'TRAIN_PROFILE_NOT_VERIFIED'):
                self.backend.call('1', 10)
            start.assert_not_called()

    def test_invalid_topk_before_loading(self):
        for k in (0, 101, True, '10', 1.2):
            with self.subTest(k=k), self.assertRaises(RecommendationError):
                self.backend.call('0', k)
        self.assertIsNone(self.backend.process)

    def test_zero_user_is_valid_and_real_mode_explicit(self):
        receipt = {'checkpoint_identity': 'a'*64, 'user': '0', 'smoke': False,
                   'items': [{'canonical_movie_id': '0', 'score': 0.125}],
                   'candidate_count': 9000, 'catalogue_size': 9031, 'train_seen_count': 31,
                   'receipt_id': 'cpu-test-only'}
        with patch.object(self.backend, 'call', return_value=receipt):
            result = UKMAPAdapter(self.spec, self.backend).infer(self.profile, 10)
        self.assertEqual(result.inference_mode, 'REAL_ONLINE_INFERENCE')
        self.assertEqual(result.predictions[0].canonical_movie_id, '0')
        self.assertIn('filtered=31', result.candidate_scope)

    def test_wrong_dataset_before_loading(self):
        with self.assertRaisesRegex(RecommendationError, 'PROFILE_DATASET_MISMATCH'):
            self.backend.infer(SimpleNamespace(dataset=ML1M, canonical_user_id='0'), 10)

    def test_absent_checkpoint_is_503(self):
        with patch.object(Path, 'is_file', return_value=False), patch('subprocess.Popen') as popen:
            with self.assertRaisesRegex(RecommendationError, 'CHECKPOINT_UNAVAILABLE') as error:
                self.backend.call('0', 10)
            self.assertEqual(error.exception.status, 503)
            popen.assert_not_called()

    def test_no_automatic_restart_after_close(self):
        self.backend.close()
        with patch('subprocess.Popen') as popen:
            with self.assertRaisesRegex(RecommendationError, 'MODEL_WORKER_UNAVAILABLE'):
                self.backend.call('0', 10)
            popen.assert_not_called()

    def test_dead_child_is_not_restarted(self):
        child = Mock()
        child.poll.return_value = 1
        self.backend.process = child
        with patch('subprocess.Popen') as popen:
            with self.assertRaisesRegex(RecommendationError, 'MODEL_WORKER_UNAVAILABLE'):
                self.backend.call('0', 10)
            popen.assert_not_called()

    def test_busy_load_lock_returns_503(self):
        self.backend.lock = Mock()
        self.backend.lock.acquire.return_value = False
        with self.assertRaisesRegex(RecommendationError, 'MODEL_BUSY'):
            self.backend.call('0', 10)
        self.backend.lock = threading.Lock()

    def test_reuse_key_includes_profile_and_mapping_versions(self):
        different = UKMAPOnlineBackend(replace(self.spec, mapping_version='v2'), '0')
        try:
            self.assertNotEqual(self.backend.identity_key, different.identity_key)
            self.assertIn(self.spec.catalogue_manifest_identity, self.backend.identity_key)
            self.assertIn(self.spec.train_manifest_identity, self.backend.identity_key)
        finally:
            different.close()

    def test_no_torch_import_in_web_adapter(self):
        import ast
        import research_recommendation.ukmap_online as module
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [x.name for n in ast.walk(tree) if isinstance(n, ast.Import) for x in n.names]
        self.assertNotIn('torch', imports)
