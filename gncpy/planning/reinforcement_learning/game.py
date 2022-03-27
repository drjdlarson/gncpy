"""Implements basic games for RL environments."""
from abc import ABC, abstractmethod
import numpy as np

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

import enum
import yaml


import gncpy.dynamics as gdyn
import serums.models as smodels


@enum.unique
class EventType(enum.Enum):
    """Define the different types of events in the game."""

    HAZARD = enum.auto()
    DEATH = enum.auto()
    TARGET = enum.auto()
    WALL = enum.auto()


# %% Entities
class EntityManager:
    def __init__(self):
        self._entities = []
        self._entities_to_add = []
        self._entity_map = {}
        self._total_entities = 0

    def _remove_dead_entities(self, vec):
        e_to_rm = []
        for ii, e in enumerate(vec):
            if not e.active:
                e_to_rm.append(ii)

        for ii in e_to_rm[::-1]:
            del vec[ii]

    def update(self):
        for e in self._entities_to_add:
            self._entities.append(e)
            if e.tag not in self._entity_map:
                self._entity_map[e.tag] = []
            self._entity_map[e.tag].append(e)
        self._entities_to_add = []

        self._remove_dead_entities(self._entities)

        for tag, ev in self._entity_map.items():
            self._remove_dead_entities(ev)

    def add_entity(self, tag):
        self._total_entities += 1
        e = Entity(self._total_entities, tag)
        self._entities_to_add.append(e)

        return e

    def get_entities(self, tag=None):
        if tag is None:
            return self._entities
        else:
            if tag in self._entity_map:
                return self._entity_map[tag]
            else:
                return []


class Entity:
    def __init__(self, e_id, tag):
        self._active = True
        self._id = e_id
        self._tag = tag

        self.c_transform = None
        self.c_dynamics = None
        self.c_shape = None
        self.c_collision = None
        self.c_birth = None
        self.c_events = None
        self.c_capabilities = None
        self.c_priority = None

    @property
    def active(self):
        return self._active

    @property
    def tag(self):
        return self._tag

    @property
    def id(self):
        return self._id

    def destroy(self):
        self._active = False


# %% Components
def _ctrl_mod(t, *args):
    return np.vstack((np.zeros((2, 2)), 5 * np.eye(2)))

class CDynamics:
    def __init__(self, dt):
        self.dynObj = gdyn.DoubleIntegrator(control_model=_ctrl_mod)

        n_states = len(gdyn.DoubleIntegrator.state_names)
        self.last_state = np.nan * np.ones((n_states, 1))
        self.state = np.nan * np.ones((n_states, 1))

        self.pos_inds = [0, 1]
        self.vel_inds = [2, 3]

        self.state_args = (dt, )

        self.max_vel = 10 * np.ones((2, 1))
        self.min_vel = -10 * np.ones((2, 1))


class CShape:
    def __init__(self, shape, w, h, color, zorder):
        if shape.lower() == 'rect':
            self.shape = pygame.Rect((0, 0), (w, h))

        self.color = color
        self.zorder = zorder


class CTransform:
    def __init__(self):
        self.pos = np.nan * np.ones((2, 1))
        self.last_pos = np.nan * np.ones((2, 1))
        self.vel = np.nan * np.ones((2, 1))


class CCollision:
    def __init__(self, w, h):
        self.aabb = pygame.Rect((0, 0), (w, h))


class CBirth:
    def __init__(self, b_type, loc, scale, params):
        self._model = smodels.Gaussian(mean=loc,
                                       covariance=scale**2)

    @property
    def loc(self):
        # TODO fix this hack by updating serums
        return self._model.sample().reshape((2, 1))


class CEvents:
    def __init__(self):
        self.events = []


class CHazard:
    def __init__(self, prob_of_death):
        self.prob_of_death = prob_of_death


class CCapabilities:
    def __init__(self, capabilities):
        self.capabilities = capabilities


class CPriority:
    def __init__(self, priority):
        self.priority = priority


# %% Systems
# %%% Base Classes
class Game(ABC):
    """Base class for defining games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities.
    """

    def __init__(self, render_mode, render_fps=60):
        """Initalize an object.

        The child class should call the :meth:`.parse_config_file` function
        after declaring all its attributes.
        """
        super().__init__()

        self._entity_manager = EntityManager()
        self._current_frame = 0
        self._render_mode = render_mode
        self.render_fps = render_fps

        self.game_over = False
        self.score = 0

    @abstractmethod
    def parse_config_file(self, config_file):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    def reset(self):
        self._entity_manager = EntityManager()
        self._current_frame = 0
        self.game_over = False
        self.score = 0

    @abstractmethod
    def s_movement(self, action):
        """Move entities according to their dynamics.

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        pass

    @abstractmethod
    def s_collision(self):
        """Check for collisions between entities.

        Returns
        -------
        None.
        """
        pass

    @abstractmethod
    def s_game_over(self):
        pass

    @abstractmethod
    def s_score(self):
        pass


class Game2d(Game):
    """Base class for defining 2d games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities.
    """

    def __init__(self, render_mode, render_fps=60):
        super().__init__(render_mode, render_fps=render_fps)

        pygame.init()

        self._window = None
        self._clock = pygame.time.Clock()
        self._img = None

        self.dt = None
        self.max_time = None
        self.origin_pos = None
        self.dist_per_pix = None

    def reset(self):
        super().reset()
        self._img = np.zeros((*self.get_image_size(), 3))

    def parse_config_file(self, config_file, verbose=True):
        with open(config_file, 'r') as fin:
            conf = yaml.safe_load(fin)

        if 'window' in conf.keys():
            self.setup_window(conf['window'])
        else:
            raise RuntimeError('Must specify window parameters in config')

        if 'physics' in conf.keys():
            self.setup_physics(conf['physics'])
        else:
            raise RuntimeError('Must specify physics parameters in config')

        for key, params in conf.items():
            if key in ('window', 'physics'):
                continue

            elif key == 'player':
                self.setup_player(params)

            elif key == 'obstacles':
                self.setup_obstacles(params)

            elif key == 'targets':
                self.setup_targets(params)

            elif key == 'hazards':
                self.setup_hazards(params)

            elif verbose:
                print('Unrecognized key ({}) in config file'.format(key))

    def _pixels_to_dist(self, pt, ind, translate):
        if translate:
            res = pt * self.dist_per_pix[ind] + self.origin_pos[ind]
        else:
            res = pt * self.dist_per_pix[ind]

        if isinstance(res, np.ndarray):
            if res.size > 1:
                return res
            else:
                return res.item()
        else:
            return res

    def _dist_to_pixels(self, pt, ind, translate):
        if translate:
            res = (pt - self.origin_pos[ind]) / self.dist_per_pix[ind]
        else:
            res = pt / self.dist_per_pix[ind]

        if isinstance(res, np.ndarray):
            if res.size > 1:
                return res.astype(int)
            else:
                return res.astype(int).item()
        else:
            return int(res)

    def setup_window(self, params):
        extra = {}
        if self._render_mode !='human':
            extra['flags'] = pygame.HIDDEN

        self._window = pygame.display.set_mode((int(params['width']),
                                                int(params['height'])),
                                               **extra)

    @abstractmethod
    def setup_player(self, params):
        pass

    @abstractmethod
    def setup_obstacles(self, params):
        pass

    @abstractmethod
    def setup_targets(self, params):
        pass

    @abstractmethod
    def setup_hazards(self, params):
        pass

    def setup_physics(self, params):
        self.dt = float(params['dt'])
        self.max_time = float(params['max_time'])
        self.origin_pos = np.array([[float(params['origin_x'])],
                                    [float(params['origin_y'])]])

        self.dist_per_pix = np.array([[float(params['dist_width'] / self._window.get_width())],
                                      [float(params['dist_height'] / self._window.get_height())]])

    def step(self, action):
        """Perform one iteration of the game loop.

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.
        """
        self._current_frame += 1
        self._entity_manager.update()

        self.s_game_over()

        # clear events for entities
        for e in self._entity_manager.get_entities():
            if e.c_events is not None:
                e.c_events.events = []

        self.s_movement(action)
        self.s_collision()

        self.s_score()

        self.render()

    def render(self):
        """Render a frame of the game."""
        surf = pygame.Surface(self._window.get_size())
        surf.fill((255, 255, 255))

        drawable = list(filter(lambda _e: _e.c_shape and _e.c_transform,
                               self._entity_manager.get_entities()))
        drawable.sort(key=lambda _e: _e.c_shape.zorder)
        for e in drawable:
            if e.c_shape is not None and e.c_transform is not None:
                e.c_shape.shape.centerx = e.c_transform.pos[0].item()
                e.c_shape.shape.centery = e.c_transform.pos[1].item()

                pygame.draw.rect(surf, e.c_shape.color, e.c_shape.shape)

        surf = pygame.transform.flip(surf, False, True)
        self._window.blit(surf, (0, 0))

        if self._render_mode == 'human':
            pygame.event.pump()
            self._clock.tick(self.render_fps)
            pygame.display.flip()

        self._img = np.transpose(np.array(pygame.surfarray.pixels3d(self._window),
                                          dtype=np.uint8),
                                 axes=(1, 0, 2))

    def get_screen_rgb(self):
        return self._img

    def get_image_size(self):
        sz = self._window.get_size()
        return sz[1], sz[0]

    def close(self):
        if self._window is not None:
            pygame.quit()


# %%% Custom Games
class BasicScore:
    def __init__(self, hazard_multiplier=5, death_scale=10, death_decay=0,
                 death_penalty=10, time_penalty=1, missed_multiplier=5,
                 target_multiplier=10, wall_penalty=5):
        self.hazard_multiplier = hazard_multiplier

        self.death_scale = death_scale
        self.death_penalty = death_penalty
        self.death_decay = death_decay

        self.time_penalty = time_penalty
        self.missed_multiplier = missed_multiplier
        self.target_multiplier = target_multiplier

        self.wall_penalty = wall_penalty

    def match_function(self, test_cap, req_cap):
        if len(req_cap) > 0:
            return sum([1 for c in test_cap if c in req_cap]) / len(req_cap)
        else:
            return 1

    def calc_score(self, t, player_lst, target_lst,
                    all_capabilities, game_over):
        """Reward function for simulated game.

        Parameters
        ----------
        player_lst : list
            Each element is an :class:`EntityBase`.
        target_lst : list
            Each elemenet is a :class:`Target`.
        params : dict
            Contains the tunable gains of the reward function.
        match_function : callable
            Function that returns a percentage of how well the capabilities match.
            Must take in two lists and return a positive number.
        all_capabilities : list
            List of all possible capabilities for the current game.
        game_over : bool
            Flag indicating if this is the last frame of the game.

        Returns
        -------
        score : float
            Amount of reward for the current frame of the game.
        """
        score = 0

        # accumulate rewards from all players
        for player in player_lst:
            s_hazard = 0
            s_target = 0
            s_death = 0
            s_wall = 0

            for e_type, info in player.c_events.events:
                if e_type == EventType.HAZARD:
                    s_hazard += -self.hazard_multiplier * info['prob']

                elif e_type == EventType.DEATH:
                    time_decay = self.death_scale * np.power(np.e, -self.death_decay * t)
                    s_death = -(time_decay * self.match_function(player.c_capabilities.capabilities,
                                                                 all_capabilities)
                                + self.death_penalty)
                    s_hazard = 0
                    s_target = 0
                    s_wall = 0
                    break

                elif e_type == EventType.TARGET:
                    target = info['target']
                    s_target = (target.c_priority.priority
                                * self.match_function(player.c_capabilities.capabilities,
                                                      target.c_capabilities.capabilities))

                elif e_type == EventType.WALL:
                    s_wall = -self.wall_penalty

            score += s_hazard + self.target_multiplier * s_target + s_death + s_wall

        # add fixed terms to reward
        r_missed = 0
        if game_over:
            for target in target_lst:
                if target.active:
                    r_missed += -target.c_priority.priority

        score += -self.time_penalty + self.missed_multiplier * r_missed

        return score


class SimpleUAV2d(Game2d):
    def __init__(self, config_file, render_mode, render_fps=None):
        super().__init__(render_mode, render_fps=render_fps)

        self._config_file = config_file
        self._scoreCls = BasicScore()
        self._all_capabilities = []

        self.parse_config_file(self._config_file)

        if self.render_fps is None:
            self.render_fps = 1 / self.dt

    def _add_capabilities(self, lst):
        for c in lst:
            if c not in self._all_capabilities:
                self._all_capabilities.append(c)

    @property
    def current_time(self):
        return self.dt * self._current_frame

    def reset(self):
        super().reset()
        self._all_capabilities = []

        self.parse_config_file(self._config_file)

    def setup_player(self, params):
        e = self._entity_manager.add_entity('player')

        # TODO: allow for other birth types
        b_params = params['birth_model']
        b_loc = np.array(b_params['location']).reshape((len(b_params['location']), 1))
        b_scale = np.diag(b_params['scale'])
        e.c_birth = CBirth(b_params['type'], b_loc, b_scale, b_params['params'])

        # TODO: allow for other dyanmics models
        d_params = params['dynamics_model']
        e.c_dynamics = CDynamics(self.dt)
        e.c_dynamics.state = np.vstack((e.c_birth.loc, 0, 0))

        e.c_transform = CTransform()

        # TODO: allow for other shape types
        s_params = params['shape_model']
        e.c_shape = CShape(s_params['type'], self._dist_to_pixels(s_params['width'], 0, False),
                           self._dist_to_pixels(s_params['height'], 1, False),
                           tuple(s_params['color']), 100)

        c_params = params['collision_model']
        e.c_collision = CCollision(self._dist_to_pixels(c_params['width'], 0, False),
                                   self._dist_to_pixels(c_params['height'], 1, False))

        e.c_events = CEvents()

        key = 'capabilities'
        if key in params:
            capabilities = params[key]
        else:
            capabilities = []
        e.c_capabilities = CCapabilities(capabilities)
        self._add_capabilities(capabilities)

    def setup_obstacles(self, params):
        for o_params in params:
            e = self._entity_manager.add_entity('obstacle')

            e.c_transform = CTransform()
            e.c_transform.pos[0] = self._dist_to_pixels(o_params['loc_x'], 0, True)
            e.c_transform.pos[1] = self._dist_to_pixels(o_params['loc_y'], 1, True)
            e.c_transform.last_pos[0] = e.c_transform.pos[0]
            e.c_transform.last_pos[1] = e.c_transform.pos[1]

            e.c_shape = CShape(o_params['shape_type'],
                               self._dist_to_pixels(o_params['width'], 0, False),
                               self._dist_to_pixels(o_params['height'], 1, False),
                               tuple(o_params['shape_color']), 1000)
            e.c_collision = CCollision(self._dist_to_pixels(o_params['collision_width'],
                                                            0, False),
                                       self._dist_to_pixels(o_params['collision_height'],
                                                            1, False))

    def setup_targets(self, params):
        for t_params in params:
            e = self._entity_manager.add_entity('target')

            e.c_transform = CTransform()
            e.c_transform.pos[0] = self._dist_to_pixels(t_params['loc_x'], 0, True)
            e.c_transform.pos[1] = self._dist_to_pixels(t_params['loc_y'], 1, True)
            e.c_transform.last_pos[0] = e.c_transform.pos[0]
            e.c_transform.last_pos[1] = e.c_transform.pos[1]

            e.c_shape = CShape(t_params['shape_type'],
                               self._dist_to_pixels(t_params['width'], 0, False),
                               self._dist_to_pixels(t_params['height'], 1, False),
                               tuple(t_params['shape_color']), 1)
            e.c_collision = CCollision(self._dist_to_pixels(t_params['collision_width'],
                                                            0, False),
                                       self._dist_to_pixels(t_params['collision_height'],
                                                            1, False))

            key = 'capabilities'
            if key in params:
                capabilities = params[key]
            else:
                capabilities = []
            e.c_capabilities = CCapabilities(capabilities)
            self._add_capabilities(capabilities)

            e.c_priority = CPriority(t_params['priority'])

    def setup_hazards(self, params):
        for h_params in params:
            e = self._entity_manager.add_entity('hazard')

            e.c_transform = CTransform()
            e.c_transform.pos[0] = self._dist_to_pixels(h_params['loc_x'], 0, True)
            e.c_transform.pos[1] = self._dist_to_pixels(h_params['loc_y'], 1, True)
            e.c_transform.last_pos[0] = e.c_transform.pos[0]
            e.c_transform.last_pos[1] = e.c_transform.pos[1]

            e.c_shape = CShape(h_params['shape_type'],
                               self._dist_to_pixels(h_params['width'], 0, False),
                               self._dist_to_pixels(h_params['height'], 1, False),
                               tuple(h_params['shape_color']), -100)
            e.c_collision = CCollision(self._dist_to_pixels(h_params['collision_width'],
                                                            0, False),
                                       self._dist_to_pixels(h_params['collision_height'],
                                                            1, False))
            pd = float(h_params['prob_of_death'])
            if pd > 1:
                pd = pd / 100.
            e.c_hazard = CHazard(pd)

    def s_movement(self, action):
        """Move entities according to their dynamics.

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        for e in self._entity_manager.get_entities():
            if e.c_transform is not None:
                e.c_transform.last_pos[0] = e.c_transform.pos[0]
                e.c_transform.last_pos[1] = e.c_transform.pos[1]

            if e.c_dynamics is not None and e.c_transform is not None:
                e.c_dynamics.last_state = e.c_dynamics.state.copy()
                e.c_dynamics.state = e.c_dynamics.dynObj.propagate_state(self.current_time,
                                                                         e.c_dynamics.last_state,
                                                                         u=action.reshape((action.size, 1)),
                                                                         state_args=e.c_dynamics.state_args,
                                                                         ctrl_args=())

                p_ii = e.c_dynamics.pos_inds
                v_ii = e.c_dynamics.vel_inds
                shape = e.c_dynamics.state[v_ii].shape
                e.c_dynamics.state[v_ii] = np.min(np.hstack((e.c_dynamics.state[v_ii],
                                                             e.c_dynamics.max_vel)),
                                                  axis=1).reshape(shape)
                e.c_dynamics.state[v_ii] = np.max(np.hstack((e.c_dynamics.state[v_ii],
                                                             e.c_dynamics.min_vel)),
                                                  axis=1).reshape(shape)

                e.c_transform.pos = self._dist_to_pixels(e.c_dynamics.state[p_ii],
                                                         [0, 1], True)
                e.c_transform.vel = self._dist_to_pixels(e.c_dynamics.state[v_ii],
                                                         [0, 1], False)

    def _get_overlap(self, bb1, bb2):
        delta = (abs(bb1.centerx - bb2.centerx), abs(bb1.centery - bb2.centery))
        ox = bb1.width / 2 + bb2.width / 2 - delta[0]
        oy = bb1.height / 2 + bb2.height / 2 - delta[1]
        return ox, oy

    def _get_last_overlap(self, pt1, pt2, bb1, bb2):
        delta = np.abs(pt1 - pt2)
        ox = bb1.width / 2 + bb2.width / 2 - delta[0].item()
        oy = bb1.height / 2 + bb2.height / 2 - delta[1].item()
        return ox, oy

    def s_collision(self):
        """Check for collisions between entities.

        Returns
        -------
        None.
        """
        rng = np.random.default_rng()

        # update all bounding boxes
        for e in self._entity_manager.get_entities():
            if e.c_transform is not None and e.c_collision is not None:
                e.c_collision.aabb.centerx = e.c_transform.pos[0].item()
                e.c_collision.aabb.centery = e.c_transform.pos[1].item()

        # check for collision of player
        for e in self._entity_manager.get_entities('player'):
            if e.c_transform is not None and e.c_collision is not None:
                p_aabb = e.c_collision.aabb
                p_trans = e.c_transform

                # check for out of bounds, stop at out of bounds
                went_oob = False
                if p_aabb.left < 0:
                    p_aabb.left = 0
                    p_trans.vel[0] = 0
                    went_oob = True
                elif p_aabb.right > self._window.get_width():
                    p_aabb.right = self._window.get_width()
                    p_trans.vel[0] = 0
                    went_oob = True

                if p_aabb.top < 0:
                    p_aabb.top = 0
                    p_trans.vel[1] = 0
                    went_oob = True
                elif p_aabb.bottom > self._window.get_height():
                    p_aabb.bottom = self._window.get_height()
                    p_trans.vel[1] = 0
                    went_oob = True

                if went_oob and e.c_events is not None:
                    e.c_events.events.append((EventType.WALL, None))

                # check for collision with wall
                for w in self._entity_manager.get_entities('obstacle'):
                    w_aabb = w.c_collision.aabb
                    ox, oy = self._get_overlap(p_aabb, w_aabb)

                    if ox > 0 and oy > 0:
                        opx, opy = self._get_last_overlap(p_trans.last_pos,
                                                          w.c_transform.last_pos,
                                                          p_aabb, w_aabb)
                        if opy > 0:
                            p_trans.vel[0] = 0
                            if p_trans.last_pos[0] < p_trans.pos[0]:
                                p_aabb.centerx -= ox
                            else:
                                p_aabb.centerx += ox
                        elif opx > 0:
                            p_trans.vel[1] = 0
                            if p_trans.last_pos[1] < p_trans.pos[1]:
                                p_aabb.centery -= oy
                            else:
                                p_aabb.centery += oy

                        if e.c_events is not None:
                            e.c_events.events.append((EventType.WALL, None))

                # check for collision with hazard
                for h in self._entity_manager.get_entities('hazard'):
                    h_aabb = h.c_collision.aabb
                    if pygame.Rect.colliderect(p_aabb, h_aabb):
                        if rng.uniform(0., 1.) < h.c_hazard.prob_of_death:
                            e.destroy()
                            e.c_events.events.append((EventType.DEATH, None))
                            break

                        else:
                            e.c_events.events.append((EventType.HAZARD,
                                                      {'prob': h.c_hazard.prob_of_death}))

                if not e.active:
                    continue

                # check for collision with target
                for ii, t in enumerate(self._entity_manager.get_entities('target')):
                    if not t.active:
                        continue

                    if pygame.Rect.colliderect(p_aabb, t.c_collision.aabb):
                        e.c_events.events.append((EventType.TARGET, {'target': t}))
                        t.destroy()
                        break

                # update position
                p_trans.pos[0] = p_aabb.centerx
                p_trans.pos[1] = p_aabb.centery

                # update state
                if e.c_dynamics is not None:
                    p_ii = e.c_dynamics.pos_inds
                    v_ii = e.c_dynamics.vel_inds

                    e.c_dynamics.state[p_ii] = self._pixels_to_dist(p_trans.pos,
                                                                    [0, 1], True)
                    e.c_dynamics.state[v_ii] = self._pixels_to_dist(p_trans.vel,
                                                                    [0, 1], False)

    def s_game_over(self):
        self.game_over = (self.current_time >= self.max_time
                          or len(self._entity_manager.get_entities('target')) == 0
                          or len(self._entity_manager.get_entities('player')) == 0)

    def s_score(self):
        self.score = self._scoreCls.calc_score(self.current_time,
                                               self._entity_manager.get_entities('player'),
                                               self._entity_manager.get_entities('target'),
                                               self._all_capabilities, self.game_over)
