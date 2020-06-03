"""
Below are two ways to run the environment
Comment out the way you don't plan to use before running.
"""

################################################################################
"""
Run the environment using the standard gym library.
"""

import gym
import gym_ctf

env = gym.make("gym_ctf:ctf-v0")
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

env.close()

################################################################################
"""
Another way to run the environment is with the use of TF agents.
"""

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

py_env = suite_gym.load("gym_ctf:ctf-v0")

env = tf_py_environment.TFPyEnvironment(py_env)

# This creates a randomly initialized policy that the agent will follow.
# Similar to just taking random actions in the environment.
policy = random_tf_policy.RandomTFPolicy(
    env.time_step_spec(), 
    env.action_spec()
)

time_step = env.reset()

while not time_step.is_last():
    action_step = policy.action(time_step)
    time_step = env.step(action_step.action)
    py_env.render('human') # Default for this render is rgb_array.

py_env.close()

################################################################################
