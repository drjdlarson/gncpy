"""Reward functions for RL planning methods."""
import numpy as np
from abc import abstractmethod

from gncpy.planning.reinforcement_learning.enums import EventType


class Reward:
    """Abstract base class for all reward classes."""
    def __init__(self):
        pass

    @abstractmethod
    def calc_reward(self, *args):
        """Calculates the reward for a given scenario.

        This must be implemented by the child class.

        Parameters
        ----------
        *args : tuple
            additional arguments needed by the function.

        Raises
        ------
        NotImplementedError
            if the child class does not implement this function.
        """
        raise NotImplementedError


class BasicReward(Reward):
    """Implements a basic reward structure.

    This handles targets, hazards, death, velocity, and time factors
    when calculating the reward.

    Attributes
    ----------
    hazard_multiplier : float
        Scale factor for the hazard contribution. The default is 5.
    death_scale : float
        Scale factor for the exponential decay of the death penalty.
        The default is 10.
    death_decay : float
        Exponential decay factor for the death penalty. Should be specified
        as a positive number. The default is 0.
    death_penalty : float
        Minimum  value to penalize death. Should be specified as a
        positive number. The default is 10.
    time_penalty : float
        Amount to penalize each timestep. Should be specified as a
        positive number. . The default is 1.
    missed_multiplier : float
        Scale factor for missed targets. The default is 5.
    target_multiplier : float
        Scale factor for reached targets. The default is 10.
    wall_penalty : float
        Amout to penalize collisions with walls. Should be specified as a
        positive number. The default is 5.
    vel_penalty : float
        Amount to penalize the velocity. The default is 1.
    min_vel_per : float
        Minimum velocity as a percentage of the largest magnitude. Only
        values less than this are penalized. The default is 0.01.
    """

    def __init__(self, hazard_multiplier=5, death_scale=10, death_decay=0,
                 death_penalty=10, time_penalty=1, missed_multiplier=5,
                 target_multiplier=10, wall_penalty=5, vel_penalty=1,
                 min_vel_per=0.01):
        """Initializes an object.

        Parameters
        ----------
        hazard_multiplier : float, optional
            Scale factor for the hazard contribution. The default is 5.
        death_scale : float, optional
            Scale factor for the exponential decay of the death penalty.
            The default is 10.
        death_decay : float, optional
            Exponential decay factor for the death penalty. Should be specified
            as a positive number. The default is 0.
        death_penalty : float, optional
            Minimum  value to penalize death. Should be specified as a
            positive number. The default is 10.
        time_penalty : float, optional
            Amount to penalize each timestep. Should be specified as a
            positive number. . The default is 1.
        missed_multiplier : float, optional
            Scale factor for missed targets. The default is 5.
        target_multiplier : float, optional
            Scale factor for reached targets. The default is 10.
        wall_penalty : float, optional
            Amout to penalize collisions with walls. Should be specified as a
            positive number. The default is 5.
        vel_penalty : float, optional
            Amount to penalize the velocity. The default is 1.
        min_vel_per : float, optional
            Minimum velocity as a percentage of the largest magnitude. Only
            values less than this are penalized. The default is 0.01.
        """
        self.hazard_multiplier = hazard_multiplier

        self.death_scale = death_scale
        self.death_penalty = death_penalty
        self.death_decay = death_decay

        self.time_penalty = time_penalty
        self.missed_multiplier = missed_multiplier
        self.target_multiplier = target_multiplier

        self.wall_penalty = wall_penalty
        self.vel_penalty = vel_penalty
        self.min_vel_per = min_vel_per

    def _match_function(self, test_cap, req_cap):
        if len(req_cap) > 0:
            return sum([1 for c in test_cap if c in req_cap]) / len(req_cap)
        else:
            return 1

    def calc_reward(self, t, player_lst, target_lst,
                    all_capabilities, game_over):
        """Calculate the reward for a timestep.

        Parameters
        ----------
        t : float
            timestep.
        player_lst : list
            Each element is a player entity.
        target_lst : list
            each element is a target entity.
        all_capabilities : list
            All possible capabilities of targets and players.
        game_over : bool
            Flag indicating if the this is the last timetep of the game.

        Returns
        -------
        reward : float
            reward for the timestep.
        info : dict
            extra info useful for debugging.
        """
        reward = 0

        # accumulate rewards from all players
        r_vel = 0
        r_haz_cumul = 0
        r_tar_cumul = 0.
        r_death_cumul = 0
        r_wall_cumul = 0
        r_vel_cumul = 0
        for player in player_lst:
            r_hazard = 0
            r_target = 0
            r_death = 0
            r_wall = 0

            max_vel = np.linalg.norm(player.c_dynamics.state_high[player.c_dynamics.vel_inds])
            min_vel = np.linalg.norm(player.c_dynamics.state_low[player.c_dynamics.vel_inds])
            vel = np.linalg.norm(player.c_dynamics.state[player.c_dynamics.vel_inds])

            vel_per = vel / np.max((max_vel, min_vel))
            if vel_per < self.min_vel_per:
                r_vel += -self.vel_penalty

            for e_type, info in player.c_events.events:
                if e_type == EventType.HAZARD:
                    r_hazard += -(self.hazard_multiplier * (info['prob'] * 100)
                                  * (t - info['t_ent']))

                elif e_type == EventType.DEATH:
                    time_decay = self.death_scale * np.exp(-self.death_decay * t)
                    r_death = -(time_decay * self._match_function(player.c_capabilities.capabilities,
                                                                  all_capabilities)
                                + self.death_penalty)
                    r_hazard = 0
                    r_target = 0
                    r_wall = 0
                    r_vel = 0
                    break

                elif e_type == EventType.TARGET:
                    target = info['target']
                    match_per = self._match_function(player.c_capabilities.capabilities,
                                                     target.c_capabilities.capabilities)
                    r_target = (self.target_multiplier * target.c_priority.priority
                                * match_per)

                elif e_type == EventType.WALL:
                    r_wall += -self.wall_penalty

            r_haz_cumul += r_hazard
            r_tar_cumul += r_target
            r_death_cumul += r_death
            r_wall_cumul += r_wall
            r_vel_cumul += r_vel
            reward += r_hazard + r_target + r_death + r_wall

        # add fixed terms to reward
        r_missed = 0
        if game_over:
            for target in target_lst:
                if target.active:
                    r_missed += -target.c_priority.priority
            r_missed *= self.missed_multiplier

        reward += -self.time_penalty + r_missed + r_vel

        info = {'hazard': r_haz_cumul, 'target': r_tar_cumul,
                'death': r_death_cumul, 'wall': r_wall_cumul, 'missed': r_missed,
                'velocity': r_vel_cumul}

        return reward, info
