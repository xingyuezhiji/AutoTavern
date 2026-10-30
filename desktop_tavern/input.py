"""Single-gesture native execution with guards immediately before mouse down."""
from __future__ import annotations

from live_tavern.actions import validate_revision
from live_tavern.power import NotReady


def execute(driver, plan, state, tail, guard):
    rect = driver.rectangle()

    def check():
        driver.check_stop()
        if tail.poll() or tail.pending:
            raise NotReady("log changed before input")
        validate_revision(plan, state)
        guard()
        if driver.rectangle() != rect:
            raise NotReady("window moved before input")

    def move(point):
        x0, y0 = driver.mouse.position()
        x1, y1 = rect[0] + point[0] * rect[2], rect[1] + point[1] * rect[3]
        for step in range(1, 13):
            driver.check_stop()
            if driver.rectangle() != rect:
                raise NotReady("window moved during input")
            driver.mouse.moveTo(round(x0 + (x1 - x0) * step / 12),
                                round(y0 + (y1 - y0) * step / 12))
            if driver.stop.wait(.02):
                raise KeyboardInterrupt

    if len(plan.operations) != 1:
        raise NotReady("desktop V2 executes one gesture at a time")
    try:
        check()
        op = plan.operations[0]
        if op["type"] == "click":
            move(op["point"])
            check()
            driver.mouse.click(button=op.get("button", "left"))
        elif op["type"] == "drag":
            move(op["start"])
            check()
            driver.mouse.mouseDown()
            move(op["end"])
            driver.mouse.mouseUp()
        else:
            raise ValueError("unknown gesture")
    finally:
        failsafe = driver.mouse.FAILSAFE
        try:
            driver.mouse.FAILSAFE = False
            driver.mouse.mouseUp()
        finally:
            driver.mouse.FAILSAFE = failsafe
