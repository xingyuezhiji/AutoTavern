"""Local runtime for checkpoints produced by AutoTavern `train.py`.

Usage with the previously created arena connector directory:

  export TAVERN_BC_CHECKPOINT=/path/to/tavern_bc_policy.pt
  export TAVERN_POLICY_RUNTIME="tavern_bc_runtime:predict"
  ./run_local.sh practice-policy
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from action_limits import limit_action_mask


@dataclass
class ModelConfig:
    feature_dim: int = 4017
    card_dim: int = 447
    action_dim: int = 245
    card_vocab: int = 4096
    card_emb: int = 32
    hidden: int = 512
    dropout: float = 0.05


class TavernBCPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.card_emb = nn.Embedding(cfg.card_vocab, cfg.card_emb, padding_idx=0)
        self.feature_net = nn.Sequential(
            nn.LayerNorm(cfg.feature_dim),
            nn.Linear(cfg.feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.card_emb, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.action_dim),
        )

    def forward(self, features, cards):
        cards = cards.clamp(min=0, max=self.cfg.card_vocab - 1)
        emb = self.card_emb(cards)
        present = (cards != 0).float().unsqueeze(-1)
        pooled = (emb * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
        x = self.feature_net(features.float())
        return self.head(torch.cat([x, pooled], dim=-1))


_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = os.getenv("TAVERN_BC_CHECKPOINT", "outputs/tavern_bc_policy.pt")
    payload = torch.load(path, map_location=_DEVICE)
    cfg = ModelConfig(**payload["model_config"])
    model = TavernBCPolicy(cfg).to(_DEVICE)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _MODEL = model
    return model


def predict(observation, context):
    model = _load_model()
    features = torch.from_numpy(np.asarray(observation["features"], dtype=np.float32)).unsqueeze(0).to(_DEVICE)
    cards = torch.from_numpy(np.asarray(observation["cards"], dtype=np.int64)).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(features, cards)[0].detach().cpu().numpy()
    mask = limit_action_mask(observation["features"],
                             np.asarray(observation["action_mask"], dtype=bool))
    return np.where(mask, logits, -np.inf)
