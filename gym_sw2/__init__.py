from gym.envs.registration import register

register(
    id='sw2-v0',
    entry_point='gym_sw2.envs:KustomSpacewarEnv',
)
#register(
#    id='sw2-extrahard-v0',
#    entry_point='gym_sw2.envs:SpacewarExtraHardEnv',
#)