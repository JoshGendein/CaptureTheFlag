import gym
import numpy as np
from gym.envs.classic_control import rendering

class Env(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, get_world, get_render, horizon=250, screen_size=512):
        self.get_world = get_world
        self.get_render = get_render

        self.screen_size = screen_size

        self.world = None
        self.viewer = None

    def step(self, action):
        pass

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_size, self.screen_size)
            
            self.get_render(self.viewer)
            self.viewer.render()
        else:
            raise ValueError("Unsupported mode.")

    def reset(self):
        self.world = self.get_world()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
