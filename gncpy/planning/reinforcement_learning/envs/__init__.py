r"""Defines all RL environments.

Once imported, the following environments are exposed:

    * SimpleUAV2d
    * SimpleUAVHazards2d

"""
from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec  # noqa


# Hook to load plugins from entry points
_load_env_plugins()


# simple2d
# ----------------------------------------

register(
    id="SimpleUAV2d-v0",
    entry_point="gncpy.planning.reinforcement_learning.envs.simple2d.simpleUAV2d:SimpleUAV2d",
)

register(
    id='SimpleUAVHazards2d-v0',
    entry_point='gncpy.planning.reinforcement_learning.envs.simple2d.simpleUAV2d:SimpleUAVHazards2d',
)
