import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

DEG2RAD = 57.29577951308232    # ratio of degrees to radians


def anglediff(a, b):
    """
    Computes minimal difference between two angles.  Always in [-180, 180)
    :param a: angle, in degrees
    :param b: angle, in degrees
    :return: difference in degrees between a and b
    """

    angle = a - b
    angle = (angle + 180) % 360 - 180
    return angle


def track_angle_dist(x1, y1, x2, y2, h):
    """
    Computes track angle and distance from agent 1's perspective.
    Track angle is the difference between agent 1's heading and the direction from agent 1 to agent 2
    If agent 1 is headed directly at agent 2, then the track angle is zero.
    :param x1: agent 1's x coordinate
    :param y1: agent 1's y coordinate
    :param x2: agent 2's x coordinate
    :param y2: agent 2's y coordinate
    :param h: agent 1's heading in degrees
    :return: track angle (degrees) & distance from agent 1 to agent 2
    """
    deltax = x2 - x1
    deltay = y2 - y1
    theta = np.arctan2(-deltax, deltay) * DEG2RAD
    d = np.sqrt((deltax) ** 2 + (deltay) ** 2)
    angle = anglediff(theta, h)

    return angle, d


class SpacewarEnv(gym.Env):

    # rendering parameters
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, p1thrust=0.02, p2thrust=0.02,
                 p1fuel=250, p2fuel=250,
                 p1turn=2, p2turn=2, drag=0.,
                 cone_angle=5, cone_min=2, cone_max=100,
                 max_life=25, kill_bonus=25, max_length=1000,
                 map_size=(600, 600), wrap=True,
                 gravity=1000., bh_rad=10,
                 debug=False, twoplayers=False):
        """
        :param p1thrust: player 1's max thrust (linear acceleration) (float)
        :param p2thrust: player 2's max thrust (linear acceleration) (float)
        :param p1fuel: player 1's starting fuel (int)
        :param p2fuel: player 2's starting fuel (int)
        :param p1turn: player 1's turn rate (float)
        :param p2turn: player 2's turn rate (float)
        :param drag: drag (deceleration, proportional to velocity) (float)
        :param cone_angle: upper bound on track angle magnitude for the kill zone (float)
        :param cone_min: lower bound on distance for the kill zone (float)
        :param cone_max: upper bound on distance for the kill zone (float)
        :param max_life: number of hits required to kill opponent (int)
        :param kill_bonus: extra reward for death of opponent (int)
        :param max_length: max number of timesteps (int)
        :param map_size: distance from center to edges of the map (xdist, ydist) (int, int)
        :param wrap: flag denoting whether an agent crashes or reappears at the other side when it hits the edge (bool)
        :param debug: debug flag
        """

        self.thrust1 = p1thrust
        self.thrust2 = p2thrust
        self.p1fuel = p1fuel
        self.p2fuel = p2fuel
        self.turn1 = p1turn
        self.turn2 = p2turn
        self.drag = drag
        self.cone_angle = cone_angle # degrees
        self.cone_min = cone_min
        self.cone_max = cone_max
        self.map_size = map_size
        self.max_life = max_life
        self.kill_bonus = kill_bonus
        self.max_speed = (100, 100)
        self.max_length = max_length
        self.wrap = wrap
        self.gravity = gravity
        self.bh_rad = bh_rad
        self.debug = debug
        self.twoplayers = twoplayers

        # set observation space
        high = np.array([self.map_size[0], # p1 x
                         self.map_size[1], # p1 y
                         self.max_speed[0],  # p1 vx
                         self.max_speed[0],  # p1 vy
                         180,  # p1 heading
                         self.p1fuel,  # p1 fuel
                         self.max_life, # p1 life
                         self.map_size[0], # p2 x
                         self.map_size[1], # p2 y
                         self.max_speed[0], # p2 vx
                         self.max_speed[0], # p2 vy
                         180, # p2 heading
                         self.p2fuel, # p2 fuel
                         self.max_life,  # p2 life
                         ],
                        dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # set action space
        if self.twoplayers:
            self.action_space = spaces.MultiDiscrete([6, 6])
        else:
            self.action_space = spaces.Discrete(6)  # left, right, thrust left, thrust right, do nothing, thrust forward

        # self.action_space = spaces.Discrete(6)  # left, right, thrust left, thrust right, do nothing, thrust forward
        self.actions = [
            [True, False, False],  # left
            [False, True, False],  # right
            [True, False, True],   # thrust left
            [False, True, True],   # thrust right
            [False, False, False], # do nothing
            [False, False, True]   # thrust forward
        ]

        self.seed()
        self.viewer = None
        self.state = None

        self.p1score = 0
        self.p2score = 0
        self.nsteps = 0

    def seed(self, seed=None):
        """
        Set random seed
        :param seed: the seed
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def testcone(self, x1, y1, x2, y2, h):
        """
        Test if agent 2 is in agent 1's kill zone.
        Kill zone is defined as track angle<cone_angle AND cone_min<=distance<=cone_max
        :param x1: agent 1's x coordinate
        :param y1: agent 1's y coordinate
        :param x2: agent 2's x coordinate
        :param y2: agent 2's y coordinate
        :param h: agent 1's heading in degrees
        :return: Boolean indicating whether agent 2 is in agent 1's kill zone
        """
        angle, d = track_angle_dist(x1, y1, x2, y2, h)
        return np.abs(angle) < self.cone_angle and d >= self.cone_min and d <= self.cone_max

    def a_grav(self, x, y):

        angle, r = track_angle_dist(0., 0., x, y, 0)

        a = self.gravity / r**2

        ax = a*np.sin(angle/DEG2RAD)
        ay = -a*np.cos(angle/DEG2RAD)

        crash = r <= (self.bh_rad+6)

        return ax, ay, crash


    def step(self, action):
        """
        :param action: a valid action in self.action_space
        :return: observation, reward, done, info
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        p1x, p1y, p1vx, p1vy, p1h, p1f, p1lf, p2x, p2y, p2vx, p2vy, p2h, p2f, p2lf = self.state

        if self.twoplayers:
            p1_action = self.actions[action[0]]
            p2_action = self.actions[action[1]]
        else:
            p1_action = self.actions[action]
            p2_action = (False, False, np.random.rand()<0.75)

            # p2thr = np.random.rand()<0.75
            # p2right = False
            # p2left = False
        p1left, p1right, p1thr = p1_action
        p2left, p2right, p2thr = p2_action


        p1x += p1vx
        p1y += p1vy
        p2x += p2vx
        p2y += p2vy

        if self.wrap:
            if p1x > self.map_size[0]:
                p1x -= self.map_size[0]*2
            if p1x <= -self.map_size[0]:
                p1x += self.map_size[0]*2
            if p1y > self.map_size[1]:
                p1y -= self.map_size[1]*2
            if p1y <= -self.map_size[1]:
                p1y += self.map_size[1]*2

            if p2x > self.map_size[0]:
                p2x -= self.map_size[0]*2
            if p2x <= -self.map_size[0]:
                p2x += self.map_size[0]*2
            if p2y > self.map_size[1]:
                p2y -= self.map_size[1]*2
            if p2y <= -self.map_size[1]:
                p2y += self.map_size[1]*2

        p1vx -= self.drag * p1vx
        p1vy -= self.drag * p1vy
        p2vx -= self.drag * p2vx
        p2vy -= self.drag * p2vy

        ax1, ay1, crash1 = self.a_grav(p1x, p1y)
        ax2, ay2, crash2 = self.a_grav(p2x, p2y)
        p1vx += ax1
        p1vy += ay1
        p2vx += ax2
        p2vy += ay2

        self.p1thrusting = p1thr and p1f>0
        if self.p1thrusting:
            p1f -= 1
        self.p2thrusting = p2thr and p2f > 0
        if self.p2thrusting:
            p2f -= 1

        if self.p1thrusting:
            p1vx += self.thrust1 * -np.sin(p1h / DEG2RAD)
            p1vy += self.thrust1 * np.cos(p1h / DEG2RAD)
        if self.p2thrusting:
            p2vx += self.thrust2 * -np.sin(p2h / DEG2RAD)
            p2vy += self.thrust2 * np.cos(p2h / DEG2RAD)

        if p1right:
            p1h += self.turn1
        if p1left:
            p1h -= self.turn1
        if p2right:
            p2h += self.turn2
        if p2left:
            p2h -= self.turn2

        if p1h < -180:
            p1h += 360
        if p1h > 180:
            p1h -= 360
        if p2h < -180:
            p2h += 360
        if p2h > 180:
            p2h -= 360

        # reward = 0.0

        self.nsteps += 1

        done = self.nsteps>=self.max_length

        prev_score = self.compute_score(p1lf, p2lf)

        if self.testcone(p1x, p1y, p2x, p2y, p1h):
            # reward += 1
            # print('Pos Reward!')
            p2lf -= 1
            if p2lf <= 0:
                p2lf = 0
                # p1sc = self.max_score + self.kill_bonus
                done = True
        if self.testcone(p2x, p2y, p1x, p1y, p2h):
            # reward -= 1
            # print('Neg Reward!')
            p1lf -= 1
            if p1lf <= 0:
                p1lf = 0
                # p2sc = self.max_score + self.kill_bonus
                done = True




        if np.abs(p1x) > self.map_size[0] or np.abs(p1y) > self.map_size[1] or crash1:
            # reward = self.crash_rew
            p1lf = 0
            done = True
        if np.abs(p2x) > self.map_size[0] or np.abs(p2y) > self.map_size[1] or crash2:
            p2lf = 0
            done = True

        new_score = self.compute_score(p1lf, p2lf)
        reward = new_score-prev_score

        self.state = (p1x, p1y, p1vx, p1vy, p1h, p1f, p1lf, p2x, p2y, p2vx, p2vy, p2h, p2f, p2lf)

        return np.array(self.state), reward, done, {}

    def compute_score(self, p1lf, p2lf):
        score = p1lf - p2lf
        if p2lf == 0:
            score += self.kill_bonus
        if p1lf == 0:
            score -= self.kill_bonus

        return score

    def reset(self):
        """
        Resets the environment
        :return: reset state
        """
        self.state = []

        self.p1score = 0
        self.p2score = 0
        self.nsteps = 0
        # (p1x, p1y, p1vx, p1vy, p1h, p1f, p1lf, p2x, p2y, p2vx, p2vy, p2h, p2f, p2lf)

        bounds = np.array([self.map_size[0],
                  self.map_size[1],
                  0, 0, 180, 0, 0,
                  self.map_size[0],
                  self.map_size[1],
                  0, 0, 180, 0 , 0])

        self.state = self.np_random.uniform(low=-bounds, high=bounds, size=bounds.shape)

        self.state[5] = self.p1fuel
        self.state[12] = self.p2fuel

        self.state[6] = self.max_life
        self.state[13] = self.max_life

        self.p1thrusting = False
        self.p2thrusting = False

        return np.array(self.state)

    def render(self, mode='human'):
        """
        Render the current state
        :param mode: either "human" or "rgb_array"
        :return: either rgb_array or render window status
        """
        screen_width = self.map_size[0]*2
        screen_height = self.map_size[1]*2
        center = self.map_size

        ulx = 5
        uly = screen_height - 5
        urx = screen_width - 5
        ury = screen_height - 5
        bar_width = screen_width*0.2
        bar_height = 10

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            blackhole = rendering.make_circle(radius=self.bh_rad, res=30, filled=True)
            self.bhtrans = rendering.Transform()
            blackhole.add_attr(self.bhtrans)

            f = (0, 10)
            br = (6, -10)
            bl = (-6, -10)

            p1 = rendering.FilledPolygon([f, br, bl])
            p2 = rendering.FilledPolygon([f, br, bl])

            self.p1trans = rendering.Transform()
            self.p2trans = rendering.Transform()

            p1.add_attr(self.p1trans)
            p1.set_color(.2, .8, .2)

            p2.add_attr(self.p2trans)
            p2.set_color(.8, .2, .2)

            p1_life_bar = rendering.FilledPolygon([[ulx, uly], [ulx+bar_width, uly], [ulx+bar_width, uly-bar_height], [ulx, uly-bar_height]])
            p1_life_bar_outline = rendering.PolyLine([[ulx, uly], [ulx+bar_width, uly], [ulx+bar_width, uly-bar_height], [ulx, uly-bar_height]], True)
            p2_life_bar = rendering.FilledPolygon([[urx-bar_width, ury], [urx, ury], [urx, ury-bar_height], [urx-bar_width, ury-bar_height]])
            p2_life_bar_outline = rendering.PolyLine([[urx-bar_width, ury], [urx, ury], [urx, ury-bar_height], [urx-bar_width, ury-bar_height]], True)
            # p1_fuel_bar = rendering.FilledPolygon([[ulx, uly], [ulx+bar_width, uly], [ulx+bar_width, uly-bar_height], [ulx, uly-bar_height]])
            # p1_fuel_bar_outline = rendering.PolyLine([[ulx, uly], [ulx+bar_width, uly], [ulx+bar_width, uly-bar_height], [ulx, uly-bar_height]], True)
            # p2_fuel_bar = rendering.FilledPolygon([[urx-bar_width, ury], [urx, ury], [urx, ury-bar_height], [urx-bar_width, ury-bar_height]])
            # p2_fuel_bar_outline = rendering.PolyLine([[urx-bar_width, ury], [urx, ury], [urx, ury-bar_height], [urx-bar_width, ury-bar_height]], True)




            # self.max_life
            p1_fuel_bar = rendering.FilledPolygon([[ulx, uly-2*bar_height], [ulx + bar_width, uly-2*bar_height], [ulx + bar_width, uly - 3*bar_height], [ulx, uly - 3*bar_height]])
            p1_fuel_bar_outline = rendering.PolyLine([[ulx, uly-2*bar_height], [ulx + bar_width, uly-2*bar_height], [ulx + bar_width, uly - 3*bar_height], [ulx, uly - 3*bar_height]], True)
            p2_fuel_bar = rendering.FilledPolygon([[urx - bar_width, ury-2*bar_height], [urx, ury-2*bar_height], [urx, ury - 3*bar_height], [urx - bar_width, ury - 3*bar_height]])
            p2_fuel_bar_outline = rendering.PolyLine([[urx - bar_width, ury-2*bar_height], [urx, ury-2*bar_height], [urx, ury - 3*bar_height], [urx - bar_width, ury - 3*bar_height]], True)

            p1_life_bar.set_color(.2, .8, .2)
            p2_life_bar.set_color(.8, .2, .2)
            self.p1_life_verts = p1_life_bar.v
            self.p2_life_verts = p2_life_bar.v

            p1_fuel_bar.set_color(.4,.4,.4)
            p2_fuel_bar.set_color(.4,.4,.4)
            self.p1_fuel_verts = p1_fuel_bar.v
            self.p2_fuel_verts = p2_fuel_bar.v


            self.viewer.add_geom(p1)
            self.viewer.add_geom(p2)
            self.viewer.add_geom(blackhole)
            self.viewer.add_geom(p1_fuel_bar)
            self.viewer.add_geom(p2_fuel_bar)
            self.viewer.add_geom(p1_fuel_bar_outline)
            self.viewer.add_geom(p2_fuel_bar_outline)
            self.viewer.add_geom(p1_life_bar)
            self.viewer.add_geom(p2_life_bar)
            self.viewer.add_geom(p1_life_bar_outline)
            self.viewer.add_geom(p2_life_bar_outline)



        if self.state is None:
            return None

        # print(self.p1_fuel_bar.v)

        # p1_fuel_verts = self.p1_fuel_bar.v
        # p2_fuel_verts = self.p2_fuel_bar.v

        p1x, p1y, p1vx, p1vy, p1h, p1f, p1lf, p2x, p2y, p2vx, p2vy, p2h, p2f, p2lf = self.state

        self.p1_fuel_verts[1][0] = bar_width*p1f/self.p1fuel + self.p1_fuel_verts[0][0]
        self.p1_fuel_verts[2][0] = bar_width*p1f/self.p1fuel + self.p1_fuel_verts[0][0]
        self.p2_fuel_verts[0][0] = self.p2_fuel_verts[1][0] - bar_width*p2f/self.p2fuel
        self.p2_fuel_verts[3][0] = self.p2_fuel_verts[1][0] - bar_width*p2f/self.p2fuel

        self.p1_life_verts[1][0] = bar_width * p1lf / self.max_life + self.p1_life_verts[0][0]
        self.p1_life_verts[2][0] = bar_width * p1lf / self.max_life + self.p1_life_verts[0][0]
        self.p2_life_verts[0][0] = self.p2_life_verts[1][0] - bar_width * p2lf / self.max_life
        self.p2_life_verts[3][0] = self.p2_life_verts[1][0] - bar_width * p2lf / self.max_life


        self.p1trans.set_translation(p1x + center[0], p1y + center[1])
        self.p1trans.set_rotation(p1h/(57.29577951308232))
        self.p2trans.set_translation(p2x + center[0], p2y + center[1])
        self.p2trans.set_rotation(p2h/(57.29577951308232))
        self.bhtrans.set_translation(center[0], center[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        close the environment
        :return:
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
