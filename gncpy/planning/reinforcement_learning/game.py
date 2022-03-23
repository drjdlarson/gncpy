"""Implements basic games for RL environments."""
from abc import ABC, abstractmethod
import numpy as np
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
    def __init__(self, shape, radius, color):
        if shape.lower() == 'rect':
            self.shape = pygame.Rect((0, 0), (2 * radius, 2 * radius))

        self.color = color


class CTransform:
    def __init__(self):
        self.pos = np.nan * np.ones((2, 1))
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


# %% Systems
class Game2d(ABC):
    """Base class for defining 2d games.

    This should implement all the necessary systems (i.e. game logic) by operating
    on entities.
    """

    def __init__(self, config_file, render_mode, render_fps=60):
        pygame.init()

        self._window = None
        self._clock = pygame.time.Clock()

        self._entity_manager = EntityManager()
        self._current_frame = 0

        self._img = np.array([[]])
        self.dt = None
        self.max_time = None
        self.game_over = False

        self._render_mode = render_mode
        self._render_fps = render_fps

        with open(config_file, 'r') as fin:
            conf = yaml.safe_load(fin)

            # except yaml.YAMLError as exc:
            #     print(exc)

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

            else:
                print('Unrecognized key ({}) in config file'.format(key))

    def setup_window(self, params):
        extra = {}
        if self._render_mode !='human':
            extra['flags'] = pygame.HIDDEN

        self._window = pygame.display.set_mode((int(params['width']), int(params['height'])),
                                               **extra)

    # @abstractmethod
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
        e.c_shape = CShape(s_params['type'], s_params['radius'], tuple(s_params['color']))

        c_params = params['collision_model']
        e.c_collision = CCollision(c_params['width'], c_params['height'])

        e.c_events = CEvents()


    def setup_obstacles(self, params):
        for o_params in params:
            e = self._entity_manager.add_entity('obstacle')

            # TODO: scale from real coordinates to pixels
            e.c_transform = CTransform()
            e.c_transform.pos[0] = o_params['loc_x']
            e.c_transform.pos[1] = o_params['loc_y']

            e.c_shape = CShape(o_params['shape_type'], o_params['radius'],
                               tuple(o_params['shape_color']))
            e.c_collision = CCollision(o_params['collision_width'],
                                       o_params['collision_height'])

    # @abstractmethod
    def setup_targets(self, params):
        pass

    def setup_hazards(self, params):
        pass

    def setup_physics(self, params):
        self.dt = float(params['dt'])
        self.max_time = float(params['max_time'])

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
            if e.c_dynamics is not None and e.c_transform is not None:
                timestep = self._current_frame * self.dt
                e.c_dynamics.last_state = e.c_dynamics.state.copy()
                e.c_dynamics.state = e.c_dynamics.dynObj.propagate_state(timestep,
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

                # TODO: map from real space to pixels
                e.c_transform.pos = e.c_dynamics.state[p_ii].astype(int)
                e.c_transform.vel = e.c_dynamics.state[v_ii].astype(int)

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
                r = e.c_collision.aabb.width / 2

                # check for out of bounds, stop at out of bounds
                went_oob = False
                if e.c_transform.pos[0] - r < 0:
                    e.c_transform.pos[0] = r
                    e.c_transform.vel[0] = 0
                    went_oob = True
                elif e.c_transform.pos[0] + r > self._window.get_width():
                    e.c_transform.pos[0] = self._window.get_width() - r
                    e.c_transform.vel[0] = 0
                    went_oob = True

                if e.c_transform.pos[1] - r < 0:
                    e.c_transform.pos[1] = r
                    e.c_transform.vel[1] = 0
                    went_oob = True
                elif e.c_transform.pos[1] + r > self._window.get_height():
                    e.c_transform.pos[1] = self._window.get_height() - r
                    e.c_transform.vel[1] = 0
                    went_oob = True

                if went_oob and e.c_events is not None:
                    e.c_events.events.append((EventType.WALL, None))

                e.c_collision.aabb.centerx = e.c_transform.pos[0].item()
                e.c_collision.aabb.centery = e.c_transform.pos[1].item()

                # check for collision with wall
                for w in self._entity_manager.get_entities('obstacle'):
                    w_aabb = w.c_collision.aabb
                    if pygame.Rect.colliderect(e.c_collision.aabb, w_aabb):
                        dx = e.c_transform.pos[0] - w_aabb.centerx
                        dy = e.c_transform.pos[1] - w_aabb.centery
                        v = pygame.math.Vector2(e.c_transform.vel[0], e.c_transform.vel[1])
                        if abs(dx) > abs(dy):
                            e.c_transform.pos[0] = w_aabb.left - r if dx < 0 else w_aabb.right + r
                            if (dx < 0 and v[0] > 0) or (dx > 0 and v[0] < 0):
                                v.reflect_ip(pygame.math.Vector2(1, 0))
                        else:
                            e.c_transform.pos[1] = w_aabb.top - r if dy < 0 else w_aabb.bottom + r
                            if (dy < 0 and v[1] > 0) or (dy > 0 and v[1] < 0):
                                v.reflect_ip(pygame.math.Vector2(0, 1))
                        e.c_transform.vel[0], e.c_transform.vel[1] = v.x, v.y
                        e.c_collision.aabb.centerx = e.c_transform.pos[0].item()
                        e.c_collision.aabb.centery = e.c_transform.pos[1].item()

                        if e.c_events is not None:
                            e.c_events.events.append((EventType.WALL, None))

                # check for collision with hazard
                has_died = False
                for h in self._entity_manager.get_entities('hazard'):
                    h_aabb = h.c_collision.aabb
                    if pygame.Rect.colliderect(e.c_collision.aabb, h_aabb):
                        if rng.uniform(0., 1.) < h.c_hazard.prob_of_death:
                            e.c_events.events.append((EventType.DEATH, None))
                            has_died = True
                            break

                        else:
                            e.c_events.events.append((EventType.HAZARD,
                                                      {'prob': h.c_hazard.prob_of_death}))

                if has_died:
                    continue

                # check for collision with target
                for ii, t in enumerate(self._entity_manager.get_entities('target')):
                    if not t.active:
                        continue

                    if pygame.Rect.colliderect(e.c_collision.aabb, t.c_collision.aabb):
                        e.c_events.events.append((EventType.TARGET, {'ind': ii}))
                        t.destroy()

                if e.c_dynamics is not None:
                    p_ii = e.c_dynamics.pos_inds
                    v_ii = e.c_dynamics.vel_inds

                    # TODO: map from pixels to real space
                    e.c_dynamics.state[p_ii] = e.c_transform.pos
                    e.c_dynamics.state[v_ii] = e.c_transform.vel

    def step(self, action):
        """Perform one iteration of the game loop.

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.
        """
        self._entity_manager.update()

        self.game_over = (self._current_frame * self.dt >= self.max_time
                          # or len(self._entity_manager.get_entities('target')) == 0
                          or len(self._entity_manager.get_entities('player')) == 0)

        # clear events for entities
        for e in self._entity_manager.get_entities():
            if e.c_events is not None:
                e.c_events.events = []

        self.s_movement(action)
        self.s_collision()

        self.score = None

        self._current_frame += 1
        self.render()

    def render(self):
        """Render a frame of the game."""
        surf = pygame.Surface(self._window.get_size())
        surf.fill((255, 255, 255))

        for e in self._entity_manager.get_entities():
            if e.c_shape is not None and e.c_transform is not None:
                e.c_shape.shape.centerx = e.c_transform.pos[0].item()
                e.c_shape.shape.centery = e.c_transform.pos[1].item()

                pygame.draw.rect(surf, e.c_shape.color, e.c_shape.shape)

        surf = pygame.transform.flip(surf, False, True)
        self._window.blit(surf, (0, 0))

        if self._render_mode == 'human':
            pygame.event.pump()
            self._clock.tick(self._render_fps)
            pygame.display.flip()

        self._img = np.transpose(np.array(pygame.surfarray.pixels3d(self._window)),
                                 axes=(1, 0, 2))
