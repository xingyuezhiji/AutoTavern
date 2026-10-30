from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


class SessionLock:
    """Lock an inode, not a stale PID. Keep the lock file after releasing it."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a+b")
        self.file.seek(0)
        if not self.file.read(1):
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("another desktop V2 session owns this installation") from None

    def close(self):
        if not self.file.closed:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Recorder:
    def __init__(self, root: Path):
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.root = root / f"{now}-{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True)

    def emit(self, record, state=None):
        row = dict(time=datetime.now(timezone.utc).isoformat(), **record)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if state and state.game_id:
            folder = self.root / f"game-{state.generation}"
            folder.mkdir(exist_ok=True)
            with (folder / "snapshots.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event=row, state=state.snapshot()),
                                        ensure_ascii=False) + "\n")
