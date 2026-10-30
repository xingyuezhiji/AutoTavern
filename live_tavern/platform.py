"""Lazy imports keep offline replay independent of desktop dependencies."""
import sys


def backend(config, stop_file):
    if sys.platform == "darwin":
        from .macos import MacOS
        return MacOS(config, stop_file)
    if sys.platform == "win32":
        from .windows import Windows
        return Windows(config, stop_file)
    raise RuntimeError("UI execution supports macOS and Windows")
