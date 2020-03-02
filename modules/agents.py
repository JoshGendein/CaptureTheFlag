import numpy as np
from modules.module import EnvModule
from modules.util import placement_fn
from gym.envs.classic_control import rendering


class Agent:

    def __init__(self, pos, color):
        self.pos = pos
        self.color = color
    
    def move(self, vec):
        self.pos[0] += vec[0]
        self.pos[1] += vec[1]


class Agents(EnvModule):
    
    def __init__(self, n_agents, grid_size, colors=None):
        self.n_agents = n_agents
        self.colors = colors
        self.agents = []

    def build_world_step(self, world):
        for i in range(self.n_agents):
            pos = placement_fn(world.grid_size, world.placement_grid, obj_size=(1, 1))

            color = (self.colors[i]
                          if isinstance(self.colors[0], (list, tuple, np.ndarray))
                          else self.colors)

            agent = Agent(pos, color)
            self.agents.append(agent)
            world.placement_grid[pos[0]][pos[1]] = 1
    
    def build_render(self, viewer, block_size):
        for agent in self.agents:
            l = agent.pos[0] * block_size
            r = l + block_size
            b = agent.pos[1] * block_size
            t = b + block_size

            current_block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            current_block.set_color(agent.color[0], agent.color[1], agent.color[2])
            viewer.add_geom(current_block)
    
    def take_action(self, world, action):
        '''
            Three steps to take action:
            1. Set previous position in grid to 0.
            2. Update your position based on given action.
            3. Set new position in grid to 1.

            args:
                world (World): The world to update grid.
                action (List): List of actions for agents to take.

        '''
        for index, agent in enumerate(self.agents):
            world.placement_grid[agent.pos[0]][agent.pos[1]] = 0
            agent.move(action[index])
            world.placement_grid[agent.pos[0]][agent.pos[1]] = 1

