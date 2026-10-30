"""Quartz logical-point input. No screenshots, app activation or permission prompts."""
from __future__ import annotations

import sys
import threading
import time

from .input import Input
from .power import NotReady


def frameworks():
    if sys.platform != "darwin":
        raise RuntimeError("macOS client required")
    try:
        import Quartz
        import AppKit
        import ApplicationServices
    except ImportError as error:
        raise RuntimeError("run live_tavern/setup-mac.command to install PyObjC") from error
    return Quartz, AppKit, ApplicationServices


def permissions(q, ax):
    return dict(accessibility=bool(ax.AXIsProcessTrusted()),
                input_monitoring=bool(q.CGPreflightListenEventAccess()),
                screen_recording=bool(q.CGPreflightScreenCaptureAccess()))


def select_rectangle(windows, foreground_pid, config, require_calibrated=True):
    # CGWindowList is front-to-back. Never select a main window behind a dialog.
    front = next((w for w in windows if w.get("kCGWindowLayer") == 0), None)
    if front is None or front.get("kCGWindowOwnerPID") != foreground_pid:
        raise NotReady("Hearthstone must be the foreground window")
    if (front.get("kCGWindowOwnerName") != config.get("window_owner", "Hearthstone")
            or front.get("kCGWindowName") != config["window_title"]):
        raise NotReady("foreground window does not match configured Hearthstone owner/title")
    bounds = front.get("kCGWindowBounds", {})
    rect = tuple(round(float(bounds.get(k, 0))) for k in ("X", "Y", "Width", "Height"))
    if min(rect[2:]) <= 0:
        raise NotReady("empty window rectangle")
    if require_calibrated and (
            not config.get("calibrated") or list(rect[2:]) != config.get("client_size")
            or config.get("coordinate_space") != "quartz_window_points"):
        raise NotReady("Mac window geometry differs from calibration; recalibrate on this Mac")
    return rect


class QuartzMouse:
    FAILSAFE = True

    def __init__(self, q, owner):
        self.q, self.owner, self.held = q, owner, None

    def position(self):
        point = self.q.CGEventGetLocation(self.q.CGEventCreate(None))
        return point.x, point.y

    def guard(self):
        if self.FAILSAFE:
            self.owner.check_stop()
            rect = self.owner.rectangle()
            x, y = self.position()
            if rect[0] <= x <= rect[0] + 3 and rect[1] <= y <= rect[1] + 3:
                raise KeyboardInterrupt

    def post(self, kind, point, button):
        event = self.q.CGEventCreateMouseEvent(None, kind, point, button)
        if event is None:
            raise RuntimeError("cannot create Quartz mouse event")
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def moveTo(self, x, y):
        self.guard()
        q = self.q
        kind = (q.kCGEventRightMouseDragged if self.held == "right" else
                q.kCGEventLeftMouseDragged if self.held else q.kCGEventMouseMoved)
        button = q.kCGMouseButtonRight if self.held == "right" else q.kCGMouseButtonLeft
        self.post(kind, (x, y), button)

    def mouseDown(self, button="left"):
        self.guard()
        q = self.q
        self.held = button
        self.post(q.kCGEventRightMouseDown if button == "right" else q.kCGEventLeftMouseDown,
                  self.position(), q.kCGMouseButtonRight if button == "right" else q.kCGMouseButtonLeft)

    def mouseUp(self):
        # Release only an input this adapter holds; never fabricate a click on abort.
        if self.held is None:
            return
        q, button = self.q, self.held
        self.post(q.kCGEventRightMouseUp if button == "right" else q.kCGEventLeftMouseUp,
                  self.position(), q.kCGMouseButtonRight if button == "right" else q.kCGMouseButtonLeft)
        self.held = None

    def click(self, button="left"):
        self.mouseDown(button)
        time.sleep(.035)
        self.mouseUp()


class MacOS(Input):
    def __init__(self, config, stop_file):
        self.q, self.appkit, self.ax = frameworks()
        status = permissions(self.q, self.ax)
        missing = [key for key in ("accessibility", "input_monitoring", "screen_recording")
                   if not status[key]]
        if missing:
            raise RuntimeError("macOS permissions missing: " + ", ".join(missing)
                               + "; grant them to Terminal (or your launching IDE) in "
                               "System Settings > Privacy & Security, then restart it")
        self.config, self.stop_file = config, stop_file
        self.stop = threading.Event()
        self.mouse = QuartzMouse(self.q, self)
        threading.Thread(target=self._listen, daemon=True, name="tavern-stop").start()

    def key(self, code):
        return self.q.CGEventSourceKeyState(self.q.kCGEventSourceStateCombinedSessionState, code)

    def _listen(self):
        while not self.stop.wait(.025):
            if self.key(100) or self.stop_file.exists():  # F8, possibly Fn+F8
                self.stop.set()

    def check_stop(self):
        if self.stop.is_set() or self.stop_file.exists() or self.key(100):
            raise KeyboardInterrupt

    def rectangle(self, require_calibrated=True):
        self.check_stop()
        app = self.appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            raise NotReady("no foreground application")
        windows = self.q.CGWindowListCopyWindowInfo(
            self.q.kCGWindowListOptionOnScreenOnly | self.q.kCGWindowListExcludeDesktopElements,
            self.q.kCGNullWindowID) or []
        return select_rectangle(windows, int(app.processIdentifier()),
                                self.config, require_calibrated)

    def capture_point(self):
        while self.key(97):  # F6, possibly Fn+F6
            self.check_stop()
            time.sleep(.025)
        while not self.key(97):
            self.check_stop()
            time.sleep(.025)
        left, top, width, height = self.rectangle(False)
        x, y = self.mouse.position()
        point = [(x - left) / width, (y - top) / height]
        if not all(0 < value < 1 for value in point):
            raise ValueError("pointer is outside the Hearthstone window")
        return point, [width, height]


def diagnose(config):
    q, appkit, ax = frameworks()
    app = appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
    windows = q.CGWindowListCopyWindowInfo(
        q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements,
        q.kCGNullWindowID) or []
    result = dict(permissions=permissions(q, ax), coordinate_space="quartz_window_points",
                  matching_windows=[dict(title=w.get("kCGWindowName", ""),
                                         owner=w.get("kCGWindowOwnerName", ""),
                                         bounds=dict(w.get("kCGWindowBounds", {})))
                                    for w in windows
                                    if w.get("kCGWindowOwnerName") == config.get("window_owner", "Hearthstone")])
    try:
        result["foreground_rectangle"] = select_rectangle(
            windows, int(app.processIdentifier()) if app else -1, config, False)
    except NotReady as error:
        result["window_status"] = str(error)
    return result
