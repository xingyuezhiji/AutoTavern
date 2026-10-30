#!/usr/bin/env python3
"""Build a secret-free portable Windows pilot zip with an explicitly chosen model."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from verify_live_script import fixture, config
    files = [
        "train.py", "action_limits.py", "action_identity.py", "entity_policy.py", "LICENSE",
        "crates/taverngym/data/cards.json", "docs/LIVE_SCRIPT.md",
        "docs/live-script-verification.json", "tools/verify_live_script.py"]
    files.extend(str(p.relative_to(ROOT)) for p in (ROOT / "live_tavern").iterdir()
                 if p.suffix in (".py", ".cmd", ".json", ".txt")
                 and p.name != "config.local.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(files):
            bundle.write(ROOT / name, name)
        bundle.write(args.checkpoint, "models/tavern_bc_policy.pt")
        bundle.writestr("samples/Power.log", fixture() +
                        "D 12:00:01.0000000 GameState.SendOption() - selectedOption=0\n")
        bundle.writestr("samples/geometry.json", json.dumps(config(), indent=2))
        bundle.writestr("README.txt", "Windows: run live_tavern/setup-windows.cmd.\n"
                        "Instructions: docs/LIVE_SCRIPT.md.\n"
                        "This is an experimental pilot; live client not tested.\n")
        bundle.writestr("model-manifest.json", json.dumps({
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "live_client_tested": False}, indent=2))
    print(args.output)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
