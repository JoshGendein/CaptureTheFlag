import numpy as np
import time
from modules.walls import Walls
from modules.agents import Agents
from modules.flag import Flag
from world import World
from env import Env


grid_size = 32

world = World(grid_size=grid_size)
world.add_module(Walls(grid_size=world.grid_size))
world.add_module(Agents(n_agents=1, grid_size=world.grid_size, colors=[(0,0,255)]))
world.add_module(Flag(flag_size=2))

env = Env(world=world)
env.reset()

for _ in range(env.horizon): 
    env.render()
    env.step(env.action_space.sample()) # Take a random action
    time.sleep(1 / env.horizon)

# env.close()