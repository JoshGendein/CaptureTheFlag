import numpy as np
from env import Env

class Base(Env):
    def __init__(self, grid_size=30, horizon=250):

        super().__init__(get_world=self.get_world, get_render=self.get_render, horizon=horizon)

        self.grid_size = grid_size
        self.horizon = horizon

        self.placement_grid = np.zeros((self.grid_size, self.grid_size))
        self.modules = []
        self.map = []

    def add_module(self, module):
        self.modules.append(module)

    def get_observation(self):
        obs = {}

        for module in self.modules:
            obs.update(module.observation_step(self))

        return obs

    # Instantiate environement.
    def get_world(self):

        for module in self.modules:
            module.build_world_step(self)

        self.map = np.stack(self.map)
        return self.placement_grid

    # Asks each module to render it's components.
    def get_render(self, viewer):
        block_size = viewer.width // self.grid_size
        for module in self.modules:
            module.build_render(viewer, block_size)

        return viewer