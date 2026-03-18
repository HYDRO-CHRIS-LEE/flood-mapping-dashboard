import math
import os

from . import rendering_compat as rendering
from . import resources
from . import sprite

if os.environ.get("SDL_VIDEODRIVER") != "dummy":
    try:
        import pyglet
        from pyglet.window import key
        from pyglet.gl import *
    except ImportError:
        pass

# Default wall image dimensions (from spritemap: 26x160 at scale 3.0)
_DEFAULT_WALL_IMG_W = 26
_DEFAULT_WALL_IMG_H = 160


class Wall():

    def __init__(self, env, vel_x, gap_y, gap_size, win_width, win_height):

        if resources.wall1_image is not None:
            image_width = resources.wall1_image.width
            image_height = resources.wall1_image.height
        else:
            image_width = _DEFAULT_WALL_IMG_W
            image_height = _DEFAULT_WALL_IMG_H

        self.pos_x = win_width + image_width
        self.vel_x = vel_x

        self.gap_y = gap_y
        self.gap_size = gap_size

        y1 = gap_y + gap_size/2.0
        y2 = gap_y - gap_size/2.0

        self.wall1_y = win_height +(image_height*3)/2.0 - (win_height - y1)
        self.wall2_y = -(image_height*3)/2.0 + y2

        self.wall_width = 78
        self.wall_height = 480

        self.env = env
        self.init_viewer = False

        self.passed = False
        self.gone = False


    def __del__(self):

        if self.init_viewer and self.env.viewer:
            self.env.viewer.geoms.remove( self.wall1 )
            self.env.viewer.geoms.remove( self.wall2 )

            del self.wall1
            del self.wall2


    def step(self, dt):

        if self.pos_x < -self.wall_width:
            self.gone = True
        else:
            self.pos_x += self.vel_x * dt

        if self.env.viewer != None:            

            if self.init_viewer == False:
                # init viewer once

                self.wall1 = sprite.Sprite( resources.wall1_image, 78, 480 )
                self.wall2 = sprite.Sprite( resources.wall2_image, 78, 480 )

                self.wall1.img.scale = 3.0
                self.wall2.img.scale = 3.0

                self.wall1.x = self.pos_x
                self.wall2.x = self.pos_x

                self.wall1.y = self.wall1_y
                self.wall2.y = self.wall2_y

                self.trans_wall1 = rendering.Transform()
                self.trans_wall2 = rendering.Transform()

                self.wall1.add_attr(self.trans_wall1)
                self.wall2.add_attr(self.trans_wall2)

                self.trans_wall1.set_translation(self.wall1.x, self.wall1.y)
                self.trans_wall2.set_translation(self.wall2.x, self.wall2.y)

                self.env.viewer.add_geom( self.wall1 )
                self.env.viewer.add_geom( self.wall2 )

                self.init_viewer = True


            self.wall1.x = self.pos_x
            self.wall2.x = self.pos_x

            self.trans_wall1.set_translation(self.wall1.x, self.wall1.y)
            self.trans_wall2.set_translation(self.wall2.x, self.wall2.y)





