"""Release-specific error contracts. GPU behavior is accepted separately over TCP."""
import unittest
from unittest.mock import patch
from django.test import SimpleTestCase
from django.db import OperationalError
from research_recommendation.contracts import ModelSpec,M3L,RecommendationError
from research_recommendation.pamk_online import PAMKOnlineBackend
from research_recommendation.product_middleware import PrivateProductResponseMiddleware
from research_recommendation.encoding_policy import policy
class ReleaseBoundaryTests(SimpleTestCase):
 def test_missing_worker_python_is_structured_503(self):
  b=PAMKOnlineBackend(ModelSpec('PAMK-Net',M3L,'a'*64,'release-test','b'*64,'c'*64),None)
  try:
   with patch('research_recommendation.pamk_online.PYTHON','/nonexistent/release-worker-python'):
    with self.assertRaises(RecommendationError) as caught:b._start()
    self.assertEqual(caught.exception.status,503)
  finally:b.close()
 def test_mean_policy_does_not_claim_encoder_checkpoint(self):
  m=policy('mean');l=policy('learned');self.assertIsNone(m['encoder_checkpoint_sha256']);self.assertNotEqual(m['encoding_identity'],l['encoding_identity']);self.assertEqual(m['model_name'],'PAMK-HistoryMean-v1')
 def test_invalid_policy_rejected(self):
  with self.assertRaises(ValueError):policy('random_user')
 def test_database_error_is_explicit_private_503(self):
  response=PrivateProductResponseMiddleware(lambda r:None).process_exception(None,OperationalError('synthetic-connection-error'))
  self.assertEqual(response.status_code,503);self.assertIn(b'DATABASE_UNAVAILABLE',response.content);self.assertIn('no-store',response['Cache-Control'])

 def test_feature_config_changes_encoding_identity(self):
  import os,json,tempfile
  from pathlib import Path
  original=policy('mean')['encoding_identity']
  cfg=json.loads(Path(os.environ['PAMK_ASSET_CONFIG']).read_text())
  cfg['assets']['image_clip_image']['sha256']='0'*64
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'changed-assets.json';p.write_text(json.dumps(cfg))
   with patch.dict(os.environ,{'PAMK_ASSET_CONFIG':str(p)}):changed=policy('mean')['encoding_identity']
  self.assertNotEqual(original,changed)
