"""Module for reinforcement learning related functionality.

Exposes the following environments:

    - SimpleUAV2d
    -SimpleUAVHazards2d
"""
from gym.envs.registration import register

register(
    id='SimpleUAV2d-v0',
    entry_point='gncpy.planning.reinforcement_learning.environments:SimpleUAV2d',
    kwargs=dict(config_file='simple_uav_2d_config.yaml')
)

register(
    id='SimpleUAVHazards2d-v0',
    entry_point='gncpy.planning.reinforcement_learning.environments:SimpleUAVHazards2d',
    kwargs=dict(config_file='simple_uav_hazards_2d_config.yaml')
)
