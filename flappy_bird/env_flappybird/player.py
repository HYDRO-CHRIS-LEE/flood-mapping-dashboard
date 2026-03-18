import math
import os

from . import rendering_compat as rendering

if os.environ.get("SDL_VIDEODRIVER") != "dummy":
    try:
        import pyglet
        from pyglet.window import key
    except ImportError:
        pass

from . import resources
from . import sprite

class Player():

    def __init__(self, env, x, y, ay, max_v):

        self.pos_x = x
        self.pos_y = y
        self.vel_y = 0
        self.acc_y = ay
        self.max_v = max_v
        self.diameter = 36
        self.rotation = 0

        self.env = env
        self.init_viewer = False

        self.trans = rendering.Transform()

    def step(self, dt):

        self.pos_y += self.vel_y * dt

        if self.vel_y > 0 or self.vel_y > self.max_v:
            self.vel_y += self.acc_y * dt

        self.y = self.pos_y
        self.x = self.pos_x

        self.rotation = self.vel_y / 10
        if self.rotation < -90:
            self.rotation = -90

        if self.env.viewer != None:

            # init once
            if self.init_viewer == False:

                self.sprite = sprite.Sprite(resources.player_anim, 51, 36)
                self.sprite.img.scale = 3.0

                self.sprite.add_attr(self.trans)

                self.env.viewer.add_geom( self.sprite )
                
                self.init_viewer = True

            self.trans.set_translation(self.pos_x, self.pos_y)
            self.trans.set_rotation( math.radians(self.rotation) )

