import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering

from gym_ctf.envs.modules.walls import Walls
from gym_ctf.envs.modules.agents import Agents
from gym_ctf.envs.modules.flag import Flag
from gym_ctf.envs.modules.util import placement_fn


class CTFEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rbg_array'],
    }

    def __init__(self, horizon=250, screen_size=512, grid_size=32, n_agents=1, flag_size=1):
        self.vec_actions = {
            0 : (0, -1),
            1: (0, 1),
            2: (0, 0),
            3: (-1, 0),
            4: (1, 0)
        }

        self.horizon = horizon
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.placement_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.viewer = None
        self.action_space = spaces.Discrete(5)

        #TODO: idk if this is correct.
        self.observation_space = spaces.Box(low=0, high=255, shape=(grid_size, grid_size), dtype=np.uint8)


        # Modules in the environment
        self.walls = Walls(self.grid_size)
        self.agents = Agents(n_agents=n_agents, grid_size=self.grid_size, colors=[(0,0,255)])
        self.flag = Flag(flag_size=flag_size)

        self.modules = [self.walls, self.agents, self.flag]


        
    def step(self, action):
        '''
            Take actions in the environment.
            Args:
                action (List): A list of actions. Each action corresponds to each own agent.
        '''
        self.horizon -= 1
        actions = [self.vec_actions[action]]
        # for agent_action in action:
        #     assert self.action_space.contains(agent_action)

        #     vec_action = self.vec_actions[agent_action]
        #     actions.append(vec_action)

        self.set_action(actions)
        obs = self.get_observation()
        done = False
        
        if self.horizon == 0:
            done = True

        for pos in obs['agent_pos']:
            if(pos in obs['flag_pos']):
                done = True

        

        reward = 1 if actions[0] == (0, 1) else -1
        return obs['agent_obs'][0], reward, done, None

    def render(self, mode='human'):
        if mode == 'human' or mode == 'rgb_array':
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_size, self.screen_size)
            
                self.get_render(self.viewer)
            if mode == 'human':
                self.viewer.render()
            elif mode == 'rgb_array':
                return self.viewer.render(return_rgb_array=True)
        else:
            raise ValueError("Unsupported mode.")

    def reset(self):
        self.horizon = 250
        self.close()
        self.get_world()
        state = self.get_observation()
        return state['agent_obs'][0]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def set_action(self, action):        
        for index, agent in enumerate(self.agents.agents):
            curr_action = action[index]
            next_position = (agent.pos[0] + curr_action[0], agent.pos[1] + curr_action[1])

            if(next_position in self.flag.pos or self.placement_grid[next_position[0]][next_position[1]] == 0):
                self.placement_grid[agent.pos[0]][agent.pos[1]] = 0
                agent.move(curr_action)
                self.placement_grid[agent.pos[0]][agent.pos[1]] = 128

    def get_observation(self):
        '''
            Each agents has its own observation. 
            The observation is a placement grid but only obstacles that are in that agents line of sight are filled in.
            Line of sight is defined as a box around the agent of radius grid_size // 8
        '''
        obs = {}

        for module in self.modules:
            obs.update(module.observation_step(self))

        agent_obs = []

        for agent_pos in obs['agent_pos']:
            curr = np.zeros((self.grid_size, self.grid_size))
            self.get_agent_obs(agent_pos, curr)
            agent_obs.append(curr)

        obs['agent_obs'] = agent_obs

        return obs

    # Instantiate environement.
    def get_world(self):

        self.placement_grid = np.zeros((self.grid_size, self.grid_size))
        
        for module in self.modules:
            module.build_world_step(self)

        return self.placement_grid

    # Asks each module to render it's components.
    def get_render(self, viewer):
        block_size = viewer.width // self.grid_size
        for module in self.modules:
            module.build_render(viewer, block_size)

        return viewer

    def get_agent_obs(self, pos, grid):
        radius = self.grid_size // 8
        for x in range((pos[0] - radius), (pos[0] + radius + 1)):
            for y in range((pos[1] - radius), (pos[1] + radius + 1)):
                if x >= 0 and x < self.grid_size and y >= 0 and y < self.grid_size:
                    grid[x][y] = self.placement_grid[x][y]
    

