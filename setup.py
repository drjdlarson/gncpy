"""Setup script for gncpy."""
import itertools
from setuptools import setup

# environment specific requirements
extras = {
    "games": ["pygame>=2.1.2"],
    "reinforcement-learning": ["gym>=0.25", "opencv-python"],
}  # NOQA

extras["reinforcement-learning"].extend(
    [r for r in extras["games"] if r not in extras["reinforcement-learning"]]
)
extras["all"] = list(
    itertools.chain.from_iterable(map(lambda group: extras[group], extras.keys()))
)

setup(extras_require=extras,)
