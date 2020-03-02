import numpy as np
from modules.module import EnvModule
from modules.util import placement_fn
from gym.envs.classic_control import rendering

class Flag(EnvModule):

    def __init__(self, flag_size=2):
        self.flag_size = flag_size

    def build_world_step(self, world):
        self.pos = placement_fn(world.grid_size, world.placement_grid, obj_size=(self.flag_size, self.flag_size))

        for i in range(self.flag_size):
            for j in range(self.flag_size):
                world.placement_grid[self.pos[0] + i][self.pos[1]+ j] = 1
        
        return True

    def build_render(self, viewer, block_size):
        l = self.pos[0] * block_size
        r = l + (block_size * self.flag_size)
        b = self.pos[1] * block_size
        t = b + (block_size * self.flag_size)

        current_block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        current_block.set_color(0, 255, 0)
        viewer.add_geom(current_block)

        return True


