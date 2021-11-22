from gym.envs.registration import register

register(
    id='GraphEnv-v1',
    entry_point='gym_environments.envs:Env1',
)