from gym.envs.registration import register

register(
    id='ctf-v0',
    entry_point='gym_ctf.envs:CTFEnv',
)