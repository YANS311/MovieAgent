"""Controlled CH4 identity; paths supplied by read-only asset config."""
import os,json
from pathlib import Path
CHECKPOINT_SHA256='f098c45c993ff10d365a5dc8b2314ef816a465af867861b691f444a719c81088'
PARENT_SHA256='b97feb824f427e6437f728459fbeb443af9b8bfdbba18ac33db8786007c33cd6'
PROTOCOL='M3L_AUTHOR_ALIGNED_V1'
_cfg=json.loads(Path(os.environ['PAMK_ASSET_CONFIG']).read_text())
if _cfg['checkpoint']['sha256']!=CHECKPOINT_SHA256:raise RuntimeError('CONTROLLED_FULL_IDENTITY_REQUIRED')
CHECKPOINT=Path(_cfg['checkpoint']['path'])
