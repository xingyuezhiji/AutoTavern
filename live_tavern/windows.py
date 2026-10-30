"""Foreground-only Windows execution; dependencies are imported on Windows only."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import threading
import time

from .power import NotReady


class Windows:
    def __init__(self, config: dict, stop_file: Path):
        if os.name != "nt":
            raise RuntimeError("Windows official client required for UI execution")
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
        self.user32 = ctypes.windll.user32
        self.user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        self.user32.FindWindowW.restype = wintypes.HWND
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        import pyautogui
        self.mouse = pyautogui
        self.mouse.PAUSE = 0.02
        self.mouse.FAILSAFE = True
        self.config, self.stop_file = config, stop_file
        self.stop = threading.Event()
        threading.Thread(target=self._listen, daemon=True, name="tavern-stop").start()

    def _listen(self):
        while not self.stop.wait(0.025):
            if self.user32.GetAsyncKeyState(0x77) & 0x8000 or self.stop_file.exists():  # F8
                self.stop.set()

    def check_stop(self):
        if self.stop.is_set() or self.stop_file.exists():
            raise KeyboardInterrupt

    def rectangle(self, require_calibrated=True):
        self.check_stop()
        window = self.user32.FindWindowW(None, self.config["window_title"])
        if not window or self.user32.IsIconic(window):
            raise NotReady("Hearthstone window is missing or minimized")
        if self.user32.GetForegroundWindow() != window:
            raise NotReady("Hearthstone must be the foreground window")
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(window, ctypes.byref(rect)):
            raise NotReady("cannot read client rectangle")
        if not self.user32.ClientToScreen(window, ctypes.byref(origin)):
            raise NotReady("cannot map client rectangle to screen")
        size = [rect.right - rect.left, rect.bottom - rect.top]
        if min(size) <= 0:
            raise NotReady("empty client rectangle")
        if require_calibrated and (not self.config.get("calibrated") or
                                   size != self.config.get("client_size")):
            raise NotReady("client size differs from calibration")
        return origin.x, origin.y, size[0], size[1]

    def execute(self, plan, state, tail):
        self.rectangle()
        tail.poll()
        from .actions import validate_revision
        validate_revision(plan, state)
        if not state.options_ready or state.depth:
            raise NotReady("client options are no longer ready")
        # Every move rechecks foreground and client size. A window move during a drag
        # aborts, and finally always releases the held mouse button.
        rect = self.rectangle()
        def pixel(point):
            return rect[0] + round(point[0] * rect[2]), rect[1] + round(point[1] * rect[3])
        def move(point):
            x0, y0 = self.mouse.position()
            x1, y1 = pixel(point)
            for i in range(1, 13):
                self.check_stop()
                if self.rectangle() != rect:
                    raise NotReady("window moved during input")
                self.mouse.moveTo(round(x0 + (x1 - x0) * i / 12),
                                  round(y0 + (y1 - y0) * i / 12))
                if self.stop.wait(0.02):
                    raise KeyboardInterrupt
        try:
            for op in plan.operations:
                self.check_stop()
                if op["type"] == "click":
                    move(op["point"])
                    self.mouse.click(button=op["button"])
                elif op["type"] == "drag":
                    move(op["start"])
                    self.mouse.mouseDown()
                    move(op["end"])
                    self.mouse.mouseUp()
                else:
                    raise ValueError("unknown input operation")
        finally:
            # Releasing a held button must still work after the corner failsafe.
            # Disable that check solely for release, never for another action.
            failsafe = self.mouse.FAILSAFE
            try:
                self.mouse.FAILSAFE = False
                self.mouse.mouseUp()
            finally:
                self.mouse.FAILSAFE = failsafe

    def capture_point(self):
        """F6 records the pointer without bringing the terminal to the foreground."""
        while self.user32.GetAsyncKeyState(0x75) & 0x8000:
            self.check_stop()
            time.sleep(0.025)
        while not self.user32.GetAsyncKeyState(0x75) & 0x8000:
            self.check_stop()
            time.sleep(0.025)
        left, top, width, height = self.rectangle(False)
        x, y = self.mouse.position()
        point = [(x - left) / width, (y - top) / height]
        if not all(0 < v < 1 for v in point):
            raise ValueError("pointer is outside the client rectangle")
        return point, [width, height]
