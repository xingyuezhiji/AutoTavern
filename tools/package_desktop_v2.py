#!/usr/bin/env python3
"""Package an independent platform V2 bundle; exclude real logs and calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--platform", choices=("mac", "windows"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from verify_desktop_v2 import fixture_config, hero_fixture
    from verify_live_script import fixture
    report = json.loads((ROOT / "docs/desktop-v2-verification.json").read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if report["status"] != "passed" or report["model_sha256"] != digest:
        raise ValueError("package checkpoint must match the accepted report")
    files = ["LICENSE", "train.py", "action_identity.py", "action_limits.py", "entity_policy.py",
             "crates/taverngym/data/cards.json", "docs/DESKTOP_V2.md",
             "docs/desktop-v2-verification.json", "tools/verify_desktop_v2.py",
             "tools/verify_live_script.py", "tools/verify_mac_live_script.py"]
    # Shared modules only: no old launchers, private configuration or STOP markers.
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "live_tavern").iterdir()
              if p.suffix == ".py" or p.name == "compatibility.json"]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "desktop_tavern").iterdir()
              if p.suffix in (".py", ".txt")
              or p.name == f"config.{args.platform}.example.json"
              or (args.platform == "mac" and p.suffix in (".command", ".sh"))
              or (args.platform == "windows" and p.suffix == ".cmd")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            if name.endswith(".cmd"):
                archive.writestr(name, (ROOT / name).read_text().replace("\r\n", "\n").replace("\n", "\r\n"))
            else:
                archive.write(ROOT / name, name)
        archive.write(args.checkpoint, "models/tavern_bc_policy.pt")
        archive.writestr("samples/Power.log", hero_fixture() +
                         "D 12:00:01 GameState.SendChoices() - id=1 ChoiceType=MULLIGAN\n" +
                         fixture() + "D 12:00:02 GameState.SendOption() - selectedOption=0\n")
        archive.writestr("samples/geometry.json", json.dumps(fixture_config(), indent=2))
        archive.writestr("README.txt", f"AutoTavern Desktop V2 ({args.platform})\n"
                         f"Run desktop_tavern/setup-{args.platform}."
                         f"{'command' if args.platform == 'mac' else 'cmd'} first.\n"
                         "Read docs/DESKTOP_V2.md. Default mode is recommendations.\n"
                         "The bundled geometry/log are synthetic offline samples only.\n"
                         "No real game-client acceptance has been completed.\n")
        archive.writestr("model-manifest.json", json.dumps(
            dict(checkpoint_sha256=digest, parameters=report["parameters"],
                 platform=args.platform, reference_commit="997910c3962490513626948241fdbdb0bd39b4cd",
                 reference_code_included=False, live_client_tested=False), indent=2))
    print(json.dumps(dict(output=str(args.output),
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
