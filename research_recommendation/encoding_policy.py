"""Encoding identity is distinct from checkpoint identity; no Django/torch import."""
import hashlib,json,os
from pathlib import Path
NAMES={'learned':('PAMK-Inductive-v1','PAMK_INDUCTIVE_PROFILE'),'mean':('PAMK-HistoryMean-v1','PAMK_HISTORY_MEAN_PROFILE')}
def policy(method):
 if method not in NAMES:raise ValueError('INVALID_ENCODING_METHOD')
 manifest=json.loads(Path(os.environ['PAMK_INDUCTIVE_MANIFEST']).read_text())
 if manifest['NEW_TEST_ACCESS']!=0:raise ValueError('TEST_ACCESS_FORBIDDEN')
 model,source=NAMES[method]
 payload={'version':model,'encoding_method':method,'base_sha256':manifest['base_sha256'],'model_source_sha256':manifest['encoder_source_sha256'],'runtime_source_sha256':hashlib.sha256(Path(__file__).with_name('inductive_runtime.py').read_bytes()).hexdigest(),'encoder_checkpoint_sha256':manifest['sha256'] if method=='learned' else None,'asset_config_sha256':hashlib.sha256(Path(os.environ['PAMK_ASSET_CONFIG']).read_bytes()).hexdigest()}
 return dict(payload,encoding_identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),model_name=model,source=source)
