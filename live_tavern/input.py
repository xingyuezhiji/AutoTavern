"""One-action execution shared by platform adapters, in their native coordinates."""
from .power import NotReady


class Input:
    def execute(self, plan, state, tail):
        self.rectangle()
        tail.poll()
        from .actions import validate_revision
        validate_revision(plan, state)
        if not state.options_ready or state.depth:
            raise NotReady("client options are no longer ready")
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
                    if self.rectangle() != rect:
                        raise NotReady("window moved before click")
                    self.mouse.click(button=op["button"])
                elif op["type"] == "drag":
                    move(op["start"])
                    if self.rectangle() != rect:
                        raise NotReady("window moved before drag")
                    self.mouse.mouseDown()
                    move(op["end"])
                    self.mouse.mouseUp()
                else:
                    raise ValueError("unknown input operation")
        finally:
            failsafe = self.mouse.FAILSAFE
            try:
                self.mouse.FAILSAFE = False
                self.mouse.mouseUp()
            finally:
                self.mouse.FAILSAFE = failsafe
