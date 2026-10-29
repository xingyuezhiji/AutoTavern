"""Validate a local training checkpoint before GPU warm-start."""
import hashlib
from pathlib import Path


def load_resume(torch, path, *, ruleset, catalog, build, expected_sha256=None):
    source = Path(path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("resume checkpoint SHA256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("model_type") != "TavernBCPolicy":
        raise ValueError("unsupported resume model type")
    if payload.get("ruleset_id") != ruleset or payload.get("catalog_sha256") != catalog:
        raise ValueError("resume checkpoint ruleset/catalog mismatch")
    if payload.get("taverngym_build_info", {}).get("source_hash") != build["source_hash"]:
        raise ValueError("resume checkpoint simulator source mismatch")
    config = payload["model_config"]
    if [config[k] for k in ("feature_dim", "card_dim", "action_dim")] != [4017, 447, 245]:
        raise ValueError("resume checkpoint observation/action mismatch")
    print(f"resume checkpoint={source.name} sha256={digest}", flush=True)
    return payload


def load_policy_state(model, payload):
    """Only allow legacy migration into a newly zero-initialized identity head."""
    legacy = not payload["model_config"].get("action_identity", False)
    if legacy and model.cfg.action_identity:
        expected = {"identity_head.query.weight", "identity_head.query.bias"}
        actual = set(model.state_dict()) - set(payload["state_dict"])
        if actual != expected:
            raise ValueError("unexpected legacy migration parameter keys")
        state = dict(payload["state_dict"])
        for key in expected:
            initialized = model.state_dict()[key]
            if initialized.count_nonzero().item() != 0:
                raise ValueError("legacy identity migration must preserve original logits")
            state[key] = initialized
        model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(payload["state_dict"], strict=True)
