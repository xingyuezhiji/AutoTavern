"""Explicitly calibrated local templates. No stock screenshots or OCR guesses."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageGrab

from live_tavern.power import NotReady

SCENES = ("lobby", "queue", "result", "heroes", "hero_confirm")


def capture(driver):
    rect = driver.rectangle()
    x, y, w, h = rect
    if hasattr(driver, "q"):
        q = driver.q
        image = q.CGWindowListCreateImage(q.CGRectMake(x, y, w, h),
                                        q.kCGWindowListOptionOnScreenOnly,
                                        q.kCGNullWindowID, q.kCGWindowImageDefault)
        if image is None:
            raise NotReady("cannot capture Hearthstone window")
        # Use a PNG representation to avoid assuming Quartz byte order.
        from AppKit import NSBitmapImageRep, NSPNGFileType
        import io
        rep = NSBitmapImageRep.alloc().initWithCGImage_(image)
        encoded = bytes(rep.representationUsingType_properties_(NSPNGFileType, {}))
        frame = Image.open(io.BytesIO(encoded)).convert("RGB")
    else:
        frame = ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).convert("RGB")
    if driver.rectangle() != rect:
        raise NotReady("window moved during screenshot")
    # Retina images are scaled back into the same logical points as the mouse.
    return frame.resize((w, h), Image.Resampling.LANCZOS), rect


def crop(frame, roi):
    if (len(roi) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in roi)
            or roi[0] >= roi[2] or roi[1] >= roi[3]):
        raise ValueError("scene roi must be [left,top,right,bottom] inside the client")
    w, h = frame.size
    bounds = tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(roi))
    if bounds[2] - bounds[0] < 8 or bounds[3] - bounds[1] < 8:
        raise ValueError("scene region is too small")
    return frame.crop(bounds)


@dataclass(frozen=True)
class Evidence:
    scene: str
    score: float
    timestamp: float
    rectangle: tuple
    digest: str


class Vision:
    def __init__(self, config, base: Path, clock=time.monotonic):
        self.config, self.base, self.clock = config, base, clock
        self.templates = {}
        for name, row in config.get("scenes", {}).items():
            if name not in SCENES:
                raise ValueError(f"unknown scene: {name}")
            threshold = row.get("max_error", .035)
            if not 0 < threshold <= .15:
                raise ValueError("scene max_error must be in (0,.15]")
            path = (base / row["template"]).resolve()
            if base.resolve() not in path.parents:
                raise ValueError("scene template must be inside the config directory")
            template = Image.open(path).convert("RGB")
            if min(template.size) < 8 or np.asarray(template).std() < 5:
                raise ValueError("scene template is too small or visually uniform")
            self.templates[name] = (row, template)

    def match(self, frame, rect):
        if list(rect[2:]) != self.config.get("client_size"):
            raise NotReady("screenshot size differs from calibration")
        hits = []
        for name, (row, template) in self.templates.items():
            candidate = crop(frame, row["roi"])
            if candidate.size != template.size:
                raise NotReady("scene template geometry differs from calibration")
            error = float(np.abs(np.asarray(candidate, dtype=np.float32)
                                 - np.asarray(template, dtype=np.float32)).mean() / 255)
            if error <= row.get("max_error", .035):
                hits.append((name, error))
        if len(hits) > 1:
            raise NotReady("multiple scene templates match; recalibrate distinct regions")
        if not hits:
            return None
        name, error = hits[0]
        return Evidence(name, error, self.clock(), rect,
                        hashlib.sha256(frame.tobytes()).hexdigest())

    def read(self, driver):
        frame, rect = capture(driver)
        return self.match(frame, rect)

    def require(self, evidence, driver):
        if self.clock() - evidence.timestamp > 1.5:
            raise NotReady("scene evidence is stale")
        fresh = self.read(driver)
        if not fresh or fresh.scene != evidence.scene or fresh.rectangle != evidence.rectangle:
            raise NotReady("scene changed before click")
        return fresh
