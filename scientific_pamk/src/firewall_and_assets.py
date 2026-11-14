"""Read-only inference asset loader; frozen model math is not edited."""
import os,json,random,sys
from pathlib import Path
import numpy as np
import torch
_cfg=json.loads(Path(os.environ['PAMK_ASSET_CONFIG']).read_text())
ASSET_PATHS={k:Path(v['path']) for k,v in _cfg['assets'].items()}
N_USERS=69878
N_ITEMS=9031
def load_config():return _cfg['model_config']
def seed_everything(seed=2024):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def load_frozen_parent24(device: torch.device) -> torch.nn.Module:
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from canonical_skbhif_final import CanonicalSKBHIFFinal

    text = torch.as_tensor(
        np.asarray(np.load(ASSET_PATHS["text_minilm"], mmap_mode="r"), dtype=np.float32),
        device=device,
    )
    genres = torch.as_tensor(np.load(ASSET_PATHS["genre_item_ids"]), device=device)
    genre_mask = torch.as_tensor(np.load(ASSET_PATHS["genre_item_mask"]), device=device)

    parent = CanonicalSKBHIFFinal(
        N_USERS, N_ITEMS, text, genres, genre_mask, n_genres=19, embed_dim=128, dropout=0.1
    ).to(device)
    ckpt = torch.load(ASSET_PATHS["parent24_checkpoint"], map_location=device, weights_only=False)
    assert ckpt["seed"] == 2024 and ckpt["selection_metric"] == "validation_recall@20"
    parent.load_state_dict(ckpt["model_state_dict"], strict=True)
    parent.eval()
    parent.requires_grad_(False)
    return parent
