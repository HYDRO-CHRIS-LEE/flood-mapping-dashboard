"""
Compatibility shim for gym.envs.classic_control.rendering.

The original env was written for gym<=0.21 which included a rendering module
based on pyglet. Modern gym/gymnasium removed it. This module provides:
  - A real rendering backend if pyglet is available and a display exists
  - Stub classes for headless mode (evaluation without rendering)
"""

import os

_HEADLESS = os.environ.get("SDL_VIDEODRIVER") == "dummy" or os.environ.get("DISPLAY") is None


class _GeomStub:
    """Minimal stub for rendering.Geom when running headless."""
    def __init__(self):
        self._color = (0, 0, 0, 1)
        self.attrs = []

    def render(self):
        pass

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        self._color = (r, g, b, 1)


class _TransformStub:
    """Minimal stub for rendering.Transform."""
    def __init__(self, translation=(0, 0), rotation=0, scale=(1, 1)):
        self.translation = translation
        self.rotation = rotation
        self.scale = scale

    def set_translation(self, x, y):
        self.translation = (x, y)

    def set_rotation(self, rot):
        self.rotation = rot

    def set_scale(self, sx, sy=None):
        self.scale = (sx, sy or sx)

    def enable(self):
        pass

    def disable(self):
        pass


class _ViewerStub:
    """Minimal stub for rendering.Viewer when running headless."""
    def __init__(self, width=420, height=580):
        self.width = width
        self.height = height
        self.window = None
        self.geoms = []

    def add_geom(self, geom):
        self.geoms.append(geom)

    def render(self, return_rgb_array=False):
        if return_rgb_array:
            import numpy as np
            return np.zeros((self.height, self.width, 3), dtype="uint8")
        return None

    def close(self):
        pass


if _HEADLESS:
    # Provide stubs — no display needed
    Geom = _GeomStub
    Viewer = _ViewerStub
    Transform = _TransformStub
else:
    # Try to load real rendering
    try:
        from gym.envs.classic_control.rendering import Geom, Viewer, Transform
    except (ImportError, ModuleNotFoundError):
        try:
            # pyglet-based rendering from older gym
            import pyglet
            from pyglet.gl import *

            class Geom:
                def __init__(self):
                    self._color = (0, 0, 0, 1)
                    self.attrs = []

                def render(self):
                    for attr in self.attrs:
                        attr.enable()
                    self.render1()
                    for attr in reversed(self.attrs):
                        attr.disable()

                def render1(self):
                    raise NotImplementedError

                def add_attr(self, attr):
                    self.attrs.append(attr)

                def set_color(self, r, g, b):
                    self._color = (r, g, b, 1)

            class Viewer:
                def __init__(self, width, height, display=None):
                    self.width = width
                    self.height = height
                    self.window = pyglet.window.Window(width=width, height=height)
                    self.geoms = []

                def add_geom(self, geom):
                    self.geoms.append(geom)

                def render(self, return_rgb_array=False):
                    self.window.clear()
                    self.window.switch_to()
                    self.window.dispatch_events()
                    for geom in self.geoms:
                        geom.render()
                    self.window.flip()
                    if return_rgb_array:
                        import numpy as np
                        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
                        image_data = buffer.get_image_data()
                        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
                        arr = arr.reshape(self.height, self.width, 4)[::-1, :, :3]
                        return arr

                def close(self):
                    self.window.close()
            Transform = _TransformStub
        except ImportError:
            Geom = _GeomStub
            Viewer = _ViewerStub
            Transform = _TransformStub
