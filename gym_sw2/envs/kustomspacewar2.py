import math

import numpy as np
from gym import spaces

import spacewar2 as spacewar2

# TODO: Consider converting into a vectorized env (stable baselines can run multiple at once. Speeding up training and making it more stable)

DEG2RAD = 57.29577951308232
def anglediff(a, b):
    """
    Computes minimal difference between two angles.  Always in [-pi, pi)
    :param a: angle, in rad
    :param b: angle, in rad
    :return: difference in rad between a and b
    """

    angle = a - b
    if angle < -1*math.pi:
        angle+=2*math.pi
    if angle > math.pi:
        angle-=2*math.pi
    return angle


def kust_track_angle_dist(x1, y1, x2, y2, h):
    deltax = x2 - x1
    deltay = y2 - y1

    if deltax > math.pi:#these four if statements account for the shorter distance because of wrap
        deltax -= 2*math.pi
    if deltax < -1*math.pi:
        deltax += 2*math.pi
    if deltay > math.pi:
        deltay -= 2*math.pi
    if deltay < -1*math.pi:
        deltay += 2*math.pi

    #print('x dif: ',deltax,'\ny dif: ',deltay)
    theeta = np.arctan2(deltay,deltax)
    #print('angle between them is ',theeta*180/math.pi)
    dist = (deltax**2+deltay**2)**.5#2pi-dist or dist, whatever is smaller bc of wraparound
    angle = anglediff(theeta,h*math.pi/180+math.pi/2)#the heading value is dumb and says 0 deg points up. We switch that to pointing right by adding pi/2
    #print('modified heading: ',h+90)
    #print('heading adjust req = ',angle*180/math.pi)
    return angle,dist


class KustomSpacewarEnv(spacewar2.SpacewarEnv):
    def __init__(self, *args, **kwargs):
        super(KustomSpacewarEnv, self).__init__(*args, **kwargs)
        self.episodes = 0
        self.sp_policy = None
        self.p2_policy = None
        self.raw_shape_vec = np.array([ self.max_speed[0],  # p1 vx
                                        self.max_speed[0],  # p1 vy
                                        180,  # p1 heading
                                        self.p1fuel,  # p1 fuel
                                        self.max_life,  # p1 life
                                        self.max_speed[0],  # p2 vx
                                        self.max_speed[0],  # p2 vy
                                        180,  # p2 heading
                                        self.p2fuel,  # p2 fuel
                                        self.max_life,  # p2 life
                                        1,1,1,1,1,1,1,1, #all the sines and cosines that replace the agent locations
                                        self.gravity/(self.bh_rad+6)**2,self.gravity/(self.bh_rad+6)**2,self.gravity/(self.bh_rad+6)**2,self.gravity/(self.bh_rad+6)**2,#max for all accelerations, given a planet size of 6
                                        180,  # rel-angle
                                        (math.pi ** 2 + math.pi ** 2) ** (1 / 2)  # rel-dist, adjusted for wraparound
                                        ], dtype=np.float32)

        space = np.ones(len(self.raw_shape_vec))
        self.observation_space = spaces.Box(-space, space, dtype=np.float32)

    def step(self, action):
        if self.twoplayers:
            if self.p2_policy is None:
                raise Error("Need to initialize other player first by giving env a self play policy object and resetting it. Try using the env and model builder")
            p2_state = self.convert_observation(self.reverse_players(self.state))
            p2_action = self.p2_policy.predict(p2_state)[0]
            combined_action = np.stack([action[0], p2_action[0]], axis=0) # two outputs from auto built networks, ignore one...
            state, reward, done, _ = super(KustomSpacewarEnv, self).step(combined_action)
        else:
            state, reward, done, _ = super(KustomSpacewarEnv, self).step(action)
        return self.convert_observation(state), reward, done, {}

    def reset(self):
        self.episodes += 1
        if self.twoplayers:
            if self.sp_policy is not None:
                self.p2_policy = self.sp_policy.sample_policies(1)[0]
        new_state = super(KustomSpacewarEnv, self).reset()
        return self.convert_observation(new_state)

    def reverse_players(self, state):
        reversed_state = np.zeros(len(state))
        reversed_state[:7] += state[7:]
        reversed_state[7:] += state[:7]
        return reversed_state

    def convert_observation(self, obs):
        p1x, p1y, p1vx, p1vy, p1h, p1f, p1lf, p2x, p2y, p2vx, p2vy, p2h, p2f, p2lf = obs[0:14]

        # I add gravity to the state for likely better predictions
        ax1, ay1, _ = self.a_grav(p1x, p1y)
        ax2, ay2, _ = self.a_grav(p2x, p2y)
        # gravity should also serve as a mediocre distance to center variable

        # converting the x,y coordinates into a statespace of -pi to pi
        p1x *= math.pi / self.map_size[0]
        p1y *= math.pi / self.map_size[1]
        p2x *= math.pi / self.map_size[0]
        p2y *= math.pi / self.map_size[1]
        # instead of using rectangular coords, we convert to strange circular coords to make the wraparound make more sense to the AI
        # unfortunately, using sin and cos warps the perceived change in distance as the agent moves
        p1xsin = math.sin(p1x)  # gamma is p1x
        p1xcos = math.cos(p1x)  # gamma is p1x
        p1ysin = math.sin(p1y)  # theta is p1y
        p1ycos = math.cos(p1y)  # theta is p1y
        p2xsin = math.sin(p2x)  # gamma is p2x
        p2xcos = math.cos(p2x)  # gamma is p2x
        p2ysin = math.sin(p2y)  # theta is p2y
        p2ycos = math.cos(p2y)  # theta is p2y

        # could also add health left to each player. Will skip for now
        obs_and_compute = np.append(obs[2:7], obs[9:])
        obs_and_compute = np.append(obs_and_compute, [p1xsin, p1xcos, p1ysin, p1ycos, p2xsin, p2xcos, p2ysin, p2ycos, ax1,ay1,ax2,ay2])
        angle, dist = kust_track_angle_dist(p1x, p1y, p2x, p2y, p1h)
        obs_and_compute = np.append(obs_and_compute, angle)
        obs_and_compute = np.append(obs_and_compute, dist)
        obs_and_compute = (obs_and_compute - (self.raw_shape_vec/2)) / self.raw_shape_vec
        return obs_and_compute
