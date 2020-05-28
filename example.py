import numpy as np
import time
from environment.world import World
from environment.env import Env

np.set_printoptions(threshold=np.inf)

grid_size = 32


env = Env(grid_size=grid_size, n_agents=1, flag_size=1)
initial = env.reset()

start = time.time()

for _ in range(env.horizon): 
    env.render()
    obs, reward, done, _ = env.step([env.action_space.sample()]) # Take a random action
    if done:
        print("Got to the flag!")
        env.close()
        break
    # time.sleep(1 / env.horizon)

env.close()

end = time.time()

print(end-start)