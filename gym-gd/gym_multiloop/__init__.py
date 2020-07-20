from gym.envs.registration import register

register(
    id='MultiLoop-v0',
    entry_point='gym_multiloop.envs:MultiLoopEnv',
)
