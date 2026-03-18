import random
import numpy as np
import os

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from . import rendering_compat as rendering

_HEADLESS = os.environ.get("SDL_VIDEODRIVER") == "dummy"

pyglet = None
key = None
if not _HEADLESS:
    try:
        import pyglet
        from pyglet.gl import *
        import pyglet.window.key as key
    except (ImportError, Exception):
        pyglet = None
        key = None

from . import resources
from . import player
from . import wall
from . import sprite
from . import util


class FlappyBirdEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, cw=420, ch=580):

        self.screen_width = cw
        self.screen_height = ch

        self.viewer = None     

        self.play_sound = False

        self.state = None

        self.walls = []
        self.wall_speed = -130
        self.gap_size = 170
        self.wall_interval = 2.

        self.reward_dead = -10
        self.reward_time = 10
        self.to_be_rewared_score = 0
        self.jump_vel = 240
        self.max_vel = -400
        self.acc_y = -500

        self.score = 0
        self.travel = 0

        self.best_score_so_far = 0
        self.minimum_score_for_saving = 30

        self.gy1_last = 1
        self.gy2_last = 0

        self.action_space = gym.spaces.discrete.Discrete(2)
        self.observation_space = gym.spaces.Box( low = np.array([-1, 0, 0, 0]), high = np.array([1, 1, 1, 1]) )

        self.num_actions = self.action_space.n  # 태스크의 행동 가짓수(2)를 받아옴
        self.num_states = self.observation_space.shape[0]  # 태스크의 상태 변수 수(5)를 받아옴        
        
        self.done = False

        self.background = sprite.Sprite(resources.background_image, 420, 580)
        self.background.img.scale = 3.0

        self.score_label = sprite.ShadowedText('0', x= self.screen_width//2, y=self.screen_height*4//5)
        self.score_label.visible = True

        self.player = player.Player(self, x=180, y=320, ay=self.acc_y, max_v=self.max_vel)

        self.render_counter = 0

        self.keys = None


    def step(self, action, gap_size, dt = 0.05):

        self.reward = 0

        self.player.step(dt)
        for wall in self.walls:
            wall.step(dt)

        if self.player.pos_y > self.screen_height:
            self.player.pos_y = self.screen_height

        if self.check_collision():
            pass            

        if self.travel >= self.wall_interval:
            self.create_a_wall( gap_size )
            self.travel = 0

        self.travel += dt    

        for wall in self.walls:

            if wall.passed == False and wall.pos_x + wall.wall_width/2 < self.player.pos_x - self.player.diameter/2:

                self.score += 1

                # if self.play_sound:
                #     resources.score_sound.play()

                self.score_label.set_text( str(self.score) )
                self.score_label.visible = True
                wall.passed = True

                self.to_be_rewared_score = 1

            if wall.gone:
                self.walls.remove(wall)
                del wall


        if action == 1:
            self.jump()

        w = None
        for wall in self.walls:
            if wall.passed == False:
                w = wall
                break

        py = self.player.pos_y / self.screen_height
        vy = self.player.vel_y / self.screen_height
        if w:
            # dx = (w.pos_x - self.player.pos_x + self.player.diameter) / self.screen_width
            # dx = (w.pos_x - self.player.pos_x + self.player.diameter) / self.screen_width
            dx = ( (wall.pos_x + wall.wall_width/2) - (self.player.pos_x - self.player.diameter/2) ) /self.screen_width
            gy1 = (w.wall1_y - w.wall_height/2) / self.screen_height
            gy2 = (w.wall2_y + w.wall_height/2) / self.screen_height
            self.gy1_last = gy1
            self.gy2_last = gy2
        else:
            dx = 1
            gy1 = self.gy1_last
            gy2 = self.gy2_last

        dy1 = gy1 - py - self.player.diameter / self.screen_height
        dy2 = -(gy2 - py + self.player.diameter / self.screen_height)

        # self.state = ( py, vy, dx, gy1, gy2 )
        self.state = ( vy, dx, dy1, dy2 )

        # print('state py {:.5f}, vy {:.5f}, dx {:.5f}, gy1 {:.5f}, gy2 {:.5f}'.format(py, vy, dx, gy1, gy2))
        # print('state vy {:.5f}, dx {:.5f}, dy1 {:.5f}, dy2 {:.5f}'.format(vy, dx, dy1, dy2))



        if self.hit:
            
            # if py > gy1:
            #     miss_h = py - gy1
            # elif py < gy2:
            #     miss_h = gy2 - py
            # else:
            #     miss_h = 0

            # self.reward = self.reward_dead * ( 1 + miss_h )
            self.reward = self.reward_dead
        else:
            # self.reward = self.to_be_rewared_score
            # self.to_be_rewared_score = 0

            self.reward = self.reward_time * dt

        done = self.hit

        # if self.print_counter % 1 == 0 or self.reward != 0:
        #     print('py:{:10.5f} vy:{:10.5f} dx:{:10.5f} gy1:{:10.5f} gy2:{:10.5f} reward: {} done {}'.format(py, vy, dx, gy1, gy2, self.reward, self.collided) )
        
        return np.array(self.state), self.reward, done, {}


    def reset(self, gap_size = 200, is_random_gap = False):

        self.is_random_gap = is_random_gap

        self.player.pos_y = 400
        self.player.vel_y = 0
        self.travel = 1

        while len(self.walls) > 0:
            w = self.walls.pop()
            del w

        self.score = 0
        self.score_label.set_text( str(self.score) )
        self.score_label.visible = True

        py = self.player.pos_y / self.screen_height
        vy = self.player.vel_y / self.screen_height
        dx = 1
        gy1 = 0.4 + gap_size / self.screen_height
        gy2 = 0.4
        self.gy1_last = gy1
        self.gy2_last = gy2


        dy1 = gy1 - py - self.player.diameter / self.screen_height
        dy2 = -(gy2 - py + self.player.diameter / self.screen_height)

        # self.state = ( py, vy, dx, gy1, gy2 )
        self.state = ( vy, dx, dy1, dy2 )

        self.reward = 0

        return np.array(self.state)


    def render(self, render_skip, mode='human', close=False):

        slow_down = False
        if self.keys and key and self.keys[key.LEFT]:
            slow_down = True

        if slow_down == False and render_skip > 0:
            self.render_counter += 1
            if self.render_counter % render_skip == 0:
                self.render_counter = 0
            else:
                return

        if self.viewer is None:

            if pyglet is not None:
                from pyglet.gl import glEnable, GL_TEXTURE_2D, glTexParameteri, GL_TEXTURE_MAG_FILTER, GL_NEAREST, GL_TEXTURE_MIN_FILTER
                glEnable(GL_TEXTURE_2D)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            
            self.keys = key.KeyStateHandler()
            self.viewer.window.push_handlers(self.keys)

            self.viewer.add_geom( self.background )
            self.viewer.add_geom( self.score_label )


            # # static objects
            # #             
            # self.background = sprite.Sprite(resources.background_image, 420, 580)
            # resources.center_image(self.background.img)
            # self.background.img.scale = 3.0

            # self.score_label = sprite.ShadowedText('0', x= self.viewer.width//2, y=self.viewer.height*4//5)
            # self.score_label.visible = False

            # self.viewer.add_geom( self.background )
            # self.viewer.add_geom( self.score_label )

            # # dynamic objects
            # #
            # self.player = player.Player(self.viewer, x=180, y=320)
        
        return self.viewer.render(return_rgb_array = (mode=='rgb_array') )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def create_a_wall(self, gap_size):

        if self.is_random_gap:
            w = wall.Wall( self, vel_x = self.wall_speed,
                                    gap_y = random.randint(200, self.screen_height-200), 
                                    gap_size = random.randint(120, 170),
                                    win_width = self.screen_width,
                                    win_height = self.screen_height )        
        else:
            w = wall.Wall( self, vel_x = self.wall_speed,
                                    gap_y = random.randint(200, self.screen_height-200), 
                                    gap_size = gap_size,
                                    win_width = self.screen_width,
                                    win_height = self.screen_height )        
                                
        self.walls.append(w)            

    def check_collision(self):
        self.hit = False

        for wall in self.walls:
            if util.check_overlap( self.player.pos_x, self.player.pos_y, self.player.diameter, wall.pos_x, wall.wall1_y, wall.wall_width, wall.wall_height ) or \
               util.check_overlap( self.player.pos_x, self.player.pos_y, self.player.diameter, wall.pos_x, wall.wall2_y, wall.wall_width, wall.wall_height ):
                self.hit = True

        if self.player.pos_y < self.player.diameter:
            self.hit = True

        if self.player.pos_y > self.screen_height - self.player.diameter:
            self.hit = True


        if self.hit:
            self.ready = False

        return self.hit


    def jump(self):

        self.player.vel_y = self.jump_vel

        # if self.play_sound:
        #     resources.flap_sound.play()
