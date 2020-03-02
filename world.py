import numpy as np

class World():
    def __init__(self, grid_size=32, horizon=250):

        self.grid_size = grid_size
        self.horizon = horizon

        self.placement_grid = np.zeros((self.grid_size, self.grid_size))
        self.modules = []

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

        return self.placement_grid

    # Asks each module to render it's components.
    def get_render(self, viewer):
        block_size = viewer.width // self.grid_size
        for module in self.modules:
            module.build_render(viewer, block_size)

        return viewer

    def set_action(self, action):
        for module in self.modules:
            module.take_action(self, action)