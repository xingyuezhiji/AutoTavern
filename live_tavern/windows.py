"""Foreground-only Windows execution; dependencies are imported on Windows only."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import threading
import time

from .power import NotReady
from .input import Input


class Windows(Input):
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
                                   size != self.config.get("client_size") or
                                   self.config.get("coordinate_space") == "quartz_window_points"):
            raise NotReady("client size differs from calibration")
        return origin.x, origin.y, size[0], size[1]

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
