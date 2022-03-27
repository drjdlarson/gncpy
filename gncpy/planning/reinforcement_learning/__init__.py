"""Module for reinforcement learning related functionality."""
from gym.envs.registration import register

register(
    id='SimpleUAV2d-v0',
    entry_point='gncpy.planning.reinforcement_learning.environments:SimpleUAV2d',
    kwargs=dict(config_file='simple_uav_2d_config.yaml')
)
