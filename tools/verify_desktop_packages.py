#!/usr/bin/env python3
"""Extract both generated bundles and replay without source checkout imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mac", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for platform in ("mac", "windows"):
        archive_path = getattr(args, platform)
        with tempfile.TemporaryDirectory(prefix=f"desktop-v2-{platform}-unpack-") as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                assert not any(".local." in n or "templates/" in n or n.endswith("/STOP")
                               or "__pycache__" in n for n in names)
                assert not any(n.endswith((".dll", ".so", ".dylib")) for n in names)
                assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
                archive.extractall(root)
                if platform == "mac":
                    assert not any(n.endswith(".cmd") for n in names)
                    for name in names:
                        if name.endswith((".sh", ".command")):
                            assert archive.getinfo(name).external_attr >> 16 & 0o111
                else:
                    assert not any(n.endswith((".command", ".sh")) for n in names)
                    assert b"\r\n" in archive.read("desktop_tavern/start-windows.cmd")
            manifest = json.loads((root / "model-manifest.json").read_text())
            checkpoint = root / "models/tavern_bc_policy.pt"
            assert manifest["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            replay = subprocess.run([sys.executable, "-m", "desktop_tavern", "--config",
                                     "samples/geometry.json", "replay", "samples/Power.log"],
                                    cwd=root, env=env, capture_output=True, text=True, check=True)
            rows = [json.loads(line) for line in replay.stdout.splitlines()]
            assert rows[-1]["decisions"] == 2
            assert any(r.get("kind") == "hero_select" for r in rows)
            assert any(r.get("action") == 8 for r in rows)
            acceptance = subprocess.run([sys.executable, "tools/verify_desktop_v2.py",
                                         "--checkpoint", "models/tavern_bc_policy.pt",
                                         "--report", "acceptance.json"], cwd=root, env=env,
                                        capture_output=True, text=True, check=True)
            checks = json.loads((root / "acceptance.json").read_text())
            # Stop/reset CLI needs no live game or configured account.
            for command in ("stop", "reset"):
                subprocess.run([sys.executable, "-m", "desktop_tavern", command], cwd=root,
                               env=env, capture_output=True, text=True, check=True)
            results.append(dict(platform=platform, archive=str(archive_path),
                                sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                                extracted_replay_decisions=rows[-1]["decisions"],
                                acceptance_checks=len(checks["checks"]),
                                model_sha256=manifest["checkpoint_sha256"],
                                launchers_checked="syntax/permissions/line-endings only",
                                source_checkout_imports=False, live_client_tested=False))
    args.report.write_text(json.dumps(dict(status="passed", bundles=results),
                                      ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status="passed", bundles=2, report=str(args.report))))


if __name__ == "__main__":
    main()
