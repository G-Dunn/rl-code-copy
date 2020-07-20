from gym.envs.registration import register

####################################################### Square #################

register(
    id='GridWorld-v0',
    entry_point='gym_gridworld.envs:GridWorld'
)

