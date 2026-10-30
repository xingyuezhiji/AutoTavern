"""Inference uses the training model class, not a similar replacement network."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import torch


class Policy:
    def __init__(self, checkpoint: Path):
        from train import ModelConfig, build_model_class
        torch.set_num_threads(1)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = ModelConfig(**payload["model_config"])
        self.catalog_sha256 = payload.get("catalog_sha256")
        if not self.catalog_sha256:
            raise ValueError("checkpoint has no training catalog fingerprint")
        if (self.config.feature_dim, self.config.card_dim, self.config.action_dim) != (4017, 447, 245):
            raise ValueError("checkpoint observation layout is not v2/245")
        self.model = build_model_class(torch.nn, torch)(self.config).cpu()
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def predict(self, observation: dict, mask: np.ndarray) -> tuple[int, list[dict]]:
        f = np.asarray(observation["features"], dtype=np.float32)
        c = np.asarray(observation["cards"])
        m = np.asarray(mask, dtype=bool)
        if f.shape != (4017,) or c.shape != (447,) or m.shape != (245,):
            raise ValueError("observation shape mismatch")
        if not np.isfinite(f).all() or not np.issubdtype(c.dtype, np.integer):
            raise ValueError("invalid feature/card values")
        if np.any(c < 0) or np.any(c >= self.config.card_vocab):
            raise ValueError("card vocabulary exceeds checkpoint embedding")
        if not m.any():
            raise ValueError("no executable legal action")
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(f.copy()).unsqueeze(0),
                                torch.from_numpy(c.astype(np.int64)).unsqueeze(0))[0].numpy()
        if not np.isfinite(logits[m]).all():
            raise ValueError("non-finite legal logits")
        order = np.flatnonzero(m)
        order = order[np.argsort(-logits[order], kind="stable")]
        scores = logits[order]
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        top = [dict(action=int(a), probability=float(p), logit=float(logits[a]))
               for a, p in zip(order[:5], probs[:5])]
        return int(order[0]), top
