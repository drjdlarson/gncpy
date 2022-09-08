"""Implements the SimpleUAV2d game.

Included configuration files for this game are:

    * SimpleUAV2d.yaml
    * SimpleUAVHazards2d.yaml

"""
import numpy as np
import enum

import gncpy.dynamics.basic as gdyn
import gncpy.game_engine.physics2d as gphysics
import gncpy.game_engine.components as gcomp
from gncpy.game_engine.base_game import (
    BaseGame2d,
    Base2dParams,
    Shape2dParams,
    Collision2dParams,
)


class BirthModelParams:
    """Parameters for the birth model to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    type : string
        type of score system to use. Options are defined by
        :class:`gncpy.game_engine.components.CBirth`.
    extra_params : dict
        Extra parameters for the birth model. Varies depending on the type.
    location : numpy array
        Either just x/y position or full state length. Location parameter for
        the birth distribution.
    scale : numpy array
        Must be of appropriate dimension for the location. Scale parameter for
        the birth distrbution.
    randomize : bool
        Flag indicating if a random sample should be drawn from the birth
        distribution. This does not affect birth time.
    times : numpy array
        Birth times for object, these do not need to align with a game step time.
        If no value is provided then a probability must be given.
    prob : float
        Probability of birth at every time step. Only used if no times are given.
        Must be in the range (0, 1].
    """

    def __init__(self):
        self.type = ""
        self.extra_params = {}

        self.location = np.array([])
        self.scale = np.array([])

        self.randomize = False
        self.times = np.array([])
        self.prob = 0


class ControlModelParams:
    """Parameters for the birth model to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    type : string
        type of control model to use. Options are dependent on the dynamics type.
        For :code:`'DoubleIntegrator'` dynamics, options are :code:`'velocity'`.
        For :code:`'CoordinatedTurn'` dynamics, options are :code:`'velocity_turn'`.
    max_vel : float
        Maximum velocity for x/y components. Must be set for coordinated turn
        :code:`'velocity_turn'` type. Only used by double integrator
        :code:`'velocity'` type if max_vel_x and max_vel_y not set.
    max_vel_x : float
        Maximum velocity in the x direction. Used by double integrator
        :code:`'velocity'`.
    max_vel_y : float
        Maximum velocity in the y direction. Used by double integrator
        :code:`'velocity'`.
    max_turn_rate : float
        Maximum turn rate. Used by coordinated turn :code:`'velocity_turn'`.
    """

    def __init__(self):
        self.type = ""
        self.max_vel = None
        self.max_vel_x = None
        self.max_vel_y = None
        self.max_turn_rate = None


class StateConstraintParams:
    """Parameters for the state constraints to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    type : string
        Type of control model to use. Options are :code:`'none'` or
        :code:`'velocity'`.
    min_vels : numpy array
        Minimum x/y velocity.
    max_vels : numpy array
        Maximum x/y velocity.
    """

    def __init__(self):
        self.type = ""
        self.min_vels = np.array([])
        self.max_vels = np.array([])


class DynamicsParams:
    """Parameters for the state constraints to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    type : string
        Type of control model to use. Options are :code:`'DoubleIntegrator'` or
        :code:`'CoordinatedTurn'`. See :mod:`gncpy.dynamics.basic` for details
        on the dynamics.
    extra_params : dict
        Extra parameters for the dynamics object. Varies depending on the type.
    controlModel : :class:`.ControlModelParams`
        Parameters for the control model.
    stateConstraint : :class:`.StateConstraintParams`
        Parameters for the state constraints.
    """

    def __init__(self):
        self.type = ""
        self.extra_params = {}
        self.controlModel = ControlModelParams()
        self.stateConstraint = StateConstraintParams()


class PlayerParams:
    """Parameters for a player object to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    birth : :class:`.BirthModelParams`
        Parametrs for the birth model
    dynamics : :class:`.DynamicsParams`
        Parameters for the dynamics model.
    shape : :class:`gncpy.game_engine.rendering2d.Shape2dParams`
        Parameters for the shape.
    collision : :class`gncpy.game_engine.physics2d.Collision2dParams`
        Parameters for the collision bounding box.
    capabilities : list
        Each element is a string representing some capability of the player.
        Allows for modeling additional hardware some targets may require to
        get successful "hit".
    """

    def __init__(self):
        self.birth = BirthModelParams()
        self.dynamics = DynamicsParams()
        self.shape = Shape2dParams()
        self.collision = Collision2dParams()
        self.capabilities = []


class ObstacleParams:
    """Parameters for an obstacle object to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    loc_x : float
        X location of the center in real units.
    loc_y : float
        Y location of the center in real units.
    shape : :class:`gncpy.game_engine.rendering2d.Shape2dParams`
        Parameters for the shape.
    collision : :class`gncpy.game_engine.physics2d.Collision2dParams`
        Parameters for the collision bounding box.
    """

    def __init__(self):
        self.loc_x = 0
        self.loc_y = 0
        self.shape = Shape2dParams()
        self.collision = Collision2dParams()


class TargetParams:
    """Parameters for a target object to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    loc_x : float
        X location of the center in real units.
    loc_y : float
        Y location of the center in real units.
    shape : :class:`gncpy.game_engine.rendering2d.Shape2dParams`
        Parameters for the shape.
    collision : :class`gncpy.game_engine.physics2d.Collision2dParams`
        Parameters for the collision bounding box.
    capabilities : list
        Each element is a string representing some capability required by the
        target. When a player collides with a target, the more capabilities that
        match the better the score.
    priority : float
        Relative importance of reaching this target.
    order : int
        Order to reach this target reltive to other targets. Must be >= 0. All
        targets with the same order will be available at the same time.
    """

    def __init__(self):
        self.loc_x = 0
        self.loc_y = 0
        self.shape = Shape2dParams()
        self.collision = Collision2dParams()
        self.capabilities = []
        self.priority = 0
        self.order = 0


class HazardParams:
    """Parameters for a hazard object to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    loc_x : float
        X location of the center in real units.
    loc_y : float
        Y location of the center in real units.
    shape : :class:`gncpy.game_engine.rendering2d.Shape2dParams`
        Parameters for the shape.
    collision : :class`gncpy.game_engine.physics2d.Collision2dParams`
        Parameters for the collision bounding box.
    prob_of_death : float
        Probability of dying for each timestep in the hazard. Must be in the
        range [0, 1].
    """

    def __init__(self):
        self.loc_x = 0
        self.loc_y = 0
        self.shape = Shape2dParams()
        self.collision = Collision2dParams()
        self.prob_of_death = 0


class ScoreParams:
    """Parameters for the score system to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.
    See :meth:`.SimpleUAV2d.s_score` for details on what these terms mean for
    specific score systems.

    Attributes
    ----------
    type : string
        Type of control model to use. Options are :code:`'basic'`.
    hazard_multiplier : float
        Multiplier for the base hazard term.
    death_scale : float
        Scaling factor fo the death term.
    death_decay : float
        Decay factor for the death term.
    death_penalty : float
        Additional penalty for death.
    time_penalty : float
        Time penalty.
    missed_multiplier : float
        Multiplier for missing targets
    wall_penalty : float
        Penalty for hitting obstacles
    vel_penalty : float
        Penalty for having extreme velocity
    min_vel_per : float
        Minimum velocity as a percentage of the magnitude. Must be in range
        [0, 1].
    """

    def __init__(self):
        self.type = "basic"
        self.hazard_multiplier = 2
        self.death_scale = 0
        self.death_decay = 0.05
        self.death_penalty = 100
        self.time_penalty = 1
        self.missed_multiplier = 5
        self.target_multiplier = 50
        self.wall_penalty = 2
        self.vel_penalty = 1
        self.min_vel_per = 0.2


class Params(Base2dParams):
    """Main set of parameters to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.

    Attributes
    ----------
    players : list
        Each element is a :class:`.PlayerParams` object.
    targets : list
        Each element is a :class:`.TargetParams` object.
    obstacles : list
        Each element is a :class:`.ObstacleParams` object.
    hazards : list
        Each element is a :class:`.HazardsParams` object.
    score : :class:`.ScoreParams`
        Parameters for the score system.
    """

    def __init__(self):
        super().__init__()

        self.players = []
        self.targets = []
        self.obstacles = []
        self.hazards = []
        self.score = ScoreParams()


@enum.unique
class EventType(enum.Enum):
    """Define the different types of events in the game."""

    HAZARD = enum.auto()
    DEATH = enum.auto()
    TARGET = enum.auto()
    OBSTACLE = enum.auto()
    COL_PLAYER = enum.auto()

    def __str__(self):
        """Return the enum name for strings."""
        return self.name


class SimpleUAV2d(BaseGame2d):
    """Simple 2D UAV game.

    Assumes obstacles and hazards are static, and all players have the same
    state and action spaces.

    Attributes
    ----------
    cur_target_seq : int
        Current index into the target_seq list.
    target_seq : list
        Each element is an order value from the target parameters, only has
        unique values and is sorted in ascending order.
    all_capabilities : list
        Each element is a string, holds all (unique) possible values from target
        and player capabilities.
    has_random_player_birth_times : bool
        Flag indicating if any player can have random birth times.
    max_player_birth_time : float
        Maximum time a new player can be born.
    """

    def __init__(self, config_file, render_mode, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        config_file : string
            Full path of the configuratin file.
        render_mode : string
            Mode to render the game.
        **kwargs : dict
            Additional arguments for the parent classes.
        """
        super().__init__(config_file, render_mode, **kwargs)

        self.cur_target_seq = None
        self.target_seq = []
        self.all_capabilities = []

        self.has_random_player_birth_times = False
        self.max_player_birth_time = -np.inf

        self._performed_reset_spawn = False

    def register_params(self, yaml):
        """Register custom classes for this game with the yaml parser.

        Parameters
        ----------
        yaml : ruamel.yaml YAML object
            yaml parser to use.
        """
        super().register_params(yaml)
        yaml.register_class(ScoreParams)
        yaml.register_class(BirthModelParams)
        yaml.register_class(ControlModelParams)
        yaml.register_class(StateConstraintParams)
        yaml.register_class(DynamicsParams)
        yaml.register_class(PlayerParams)
        yaml.register_class(ObstacleParams)
        yaml.register_class(TargetParams)
        yaml.register_class(HazardParams)
        yaml.register_class(ScoreParams)
        yaml.register_class(Params)

    def create_obstacles(self):
        """Creates all obstacles based on config values."""
        for params in self.params.obstacles:
            e = self.entityManager.add_entity("obstacle")

            e.add_component(gcomp.CTransform)
            c_transform = e.get_component(gcomp.CTransform)
            c_transform.pos[0] = gphysics.dist_to_pixels(
                params.loc_x,
                self.dist_per_pix[0],
                min_pos=self.params.physics.min_pos[0],
            )
            c_transform.pos[1] = gphysics.dist_to_pixels(
                params.loc_y,
                self.dist_per_pix[1],
                min_pos=self.params.physics.min_pos[1],
            )
            c_transform.last_pos[0] = c_transform.pos[0]
            c_transform.last_pos[1] = c_transform.pos[1]

            e.add_component(
                gcomp.CShape,
                s_type=params.shape.type,
                w=gphysics.dist_to_pixels(params.shape.width, self.dist_per_pix[0]),
                h=gphysics.dist_to_pixels(params.shape.height, self.dist_per_pix[1]),
                color=params.shape.color,
                zorder=1000,
                fpath=params.shape.file,
            )

            if params.collision.height > 0 and params.collision.width > 0:
                e.add_component(
                    gcomp.CCollision,
                    w=gphysics.dist_to_pixels(params.collision.width, self.dist_per_pix[0]),
                    h=gphysics.dist_to_pixels(
                        params.collision.height, self.dist_per_pix[1]
                    ),
                )

    def create_hazards(self):
        """Creates all hazards based on config values."""
        for params in self.params.hazards:
            e = self.entityManager.add_entity("hazard")

            e.add_component(gcomp.CTransform)
            c_transform = e.get_component(gcomp.CTransform)
            c_transform.pos[0] = gphysics.dist_to_pixels(
                params.loc_x,
                self.dist_per_pix[0],
                min_pos=self.params.physics.min_pos[0],
            )
            c_transform.pos[1] = gphysics.dist_to_pixels(
                params.loc_y,
                self.dist_per_pix[1],
                min_pos=self.params.physics.min_pos[1],
            )
            c_transform.last_pos[0] = c_transform.pos[0]
            c_transform.last_pos[1] = c_transform.pos[1]

            e.add_component(
                gcomp.CShape,
                s_type=params.shape.type,
                w=gphysics.dist_to_pixels(params.shape.width, self.dist_per_pix[0]),
                h=gphysics.dist_to_pixels(params.shape.height, self.dist_per_pix[1]),
                color=params.shape.color,
                zorder=-100,
                fpath=params.shape.file,
            )

            if params.collision.height > 0 and params.collision.width > 0:
                e.add_component(
                    gcomp.CCollision,
                    w=gphysics.dist_to_pixels(params.collision.width, self.dist_per_pix[0]),
                    h=gphysics.dist_to_pixels(
                        params.collision.height, self.dist_per_pix[1]
                    ),
                )

            pd = float(params.prob_of_death)
            if pd > 1:
                pd = pd / 100.0
            e.add_component(gcomp.CHazard, prob_of_death=pd)

    def create_targets(self):
        """Creates current targets based on config values.

        Returns
        -------
        bool
            Flag indicating if targets were generated
        """
        if len(self.entityManager.get_entities("target")) > 0:
            return False

        if self.cur_target_seq is None:
            self.cur_target_seq = 0
        else:
            self.cur_target_seq += 1

        if self.cur_target_seq >= len(self.target_seq):
            return False

        order = self.target_seq[self.cur_target_seq]

        for params in self.params.targets:
            if params.order != order:
                continue

            e = self.entityManager.add_entity("target")

            e.add_component(gcomp.CTransform)
            c_transform = e.get_component(gcomp.CTransform)
            c_transform.pos[0] = gphysics.dist_to_pixels(
                params.loc_x,
                self.dist_per_pix[0],
                min_pos=self.params.physics.min_pos[0],
            )
            c_transform.pos[1] = gphysics.dist_to_pixels(
                params.loc_y,
                self.dist_per_pix[1],
                min_pos=self.params.physics.min_pos[1],
            )
            c_transform.last_pos[0] = c_transform.pos[0]
            c_transform.last_pos[1] = c_transform.pos[1]

            e.add_component(
                gcomp.CShape,
                s_type=params.shape.type,
                w=gphysics.dist_to_pixels(params.shape.width, self.dist_per_pix[0]),
                h=gphysics.dist_to_pixels(params.shape.height, self.dist_per_pix[1]),
                color=params.shape.color,
                zorder=1,
                fpath=params.shape.file,
            )

            if params.collision.height > 0 and params.collision.width > 0:
                e.add_component(
                    gcomp.CCollision,
                    w=gphysics.dist_to_pixels(params.collision.width, self.dist_per_pix[0]),
                    h=gphysics.dist_to_pixels(
                        params.collision.height, self.dist_per_pix[1]
                    ),
                )

            e.add_component(gcomp.CCapabilities, capabilities=params.capabilities)

            e.add_component(gcomp.CPriority, priority=params.priority)

        return True

    def get_player_pos_vel_inds(self, params=None):
        """Determines the position and velocity indices in the state vector.

        Note, this assumes all players have the same state bounds when params is
        not specified.

        Parameters
        ----------
        params : :class:`.DynamicsParams`, optional
            Player parameter structure. The default is None which uses the first
            player in the player's list.

        Returns
        -------
        pos_inds : list
            indices for the position components (x, y).
        vel_inds : list
            indices for the velocity components (x, y).
        """
        if params is None:
            params = self.params.players[0].dynamics
        if params.type == "DoubleIntegrator":
            pos_inds = [0, 1]
            vel_inds = [2, 3]
        elif params.type == "CoordinatedTurn":
            pos_inds = [0, 2]
            vel_inds = [1, 3]

        return pos_inds, vel_inds

    def get_player_state_bounds(self, params=None):
        """Calculate the bounds on the player state.

        Note, this assumes all players have the same state bounds when params is
        not specified.

        Parameters
        ----------
        params : :class:`.DynamicsParams`, optional
            Player parameter structure. The default is None which uses the first
            player in the player's list.

        Returns
        -------
        2 x N numpy array
            minimum and maximum values of the player state.
        """
        if params is None:
            params = self.params.players[0].dynamics

        pos_inds, vel_inds = self.get_player_pos_vel_inds(params=params)

        if params.type == "DoubleIntegrator":
            state_low = np.hstack(
                (self.params.physics.min_pos, np.array([-np.inf, -np.inf]))
            )
            state_high = np.hstack(
                (
                    self.params.physics.min_pos
                    + np.array(
                        [
                            self.params.physics.dist_width,
                            self.params.physics.dist_height,
                        ]
                    ),
                    np.array([np.inf, np.inf]),
                )
            )
            sParams = params.stateConstraint
            if sParams.type.lower() != "none":
                if sParams.type.lower() == "velocity":
                    state_low[vel_inds] = sParams.min_vels
                    state_high[vel_inds] = sParams.max_vels

        elif params.type == "CoordinatedTurn":
            state_low = np.hstack(
                (
                    self.params.physics.min_pos[0],
                    np.array([-np.inf]),
                    self.params.physics.min_pos[1],
                    np.array([-np.inf, -2 * np.pi]),
                )
            )
            state_high = np.hstack(
                (
                    self.params.physics.min_pos[0] + self.params.physics.dist_width,
                    np.array([np.inf]),
                    self.params.physics.min_pos[1] + self.params.physics.dist_height,
                    np.array([np.inf, 2 * np.pi]),
                )
            )
            if sParams.type.lower() != "none":
                if sParams.type.lower() == "velocity":
                    state_low[vel_inds] = sParams.min_vel * np.ones(len(vel_inds))
                    state_high[vel_inds] = sParams.min_vel * np.ones(len(vel_inds))

        else:
            raise RuntimeError("Invalid dynamics type: {}".format(sParams.type))

        return state_low, state_high

    def get_players_state(self):
        """Returns current dynamic state of all players.

        Returns
        -------
        states : dict
            Each key is the entity id and each value is a numpy array.
        """
        states = {}
        for p in self.entityManager.get_entities("player"):
            pTrans = p.get_component(gcomp.CDynamics)
            states[p.id] = pTrans.state.copy().ravel()

        return states

    def create_dynamics(self, params, cBirth):
        """Create the dynamics system for a player.

        Parameters
        ----------
        params : :class:`.DynamicsParams`
            Parameters to use when making the dynamics.
        cBirth : :class:`gncpy.game_engine.components.CBirth`
            Birth component associated with this player.

        Raises
        ------
        RuntimeError
            Incorrect parameters are set.
        NotImplementedError
            Wrong dynamics, control model and/or state constraint combo specified.

        Returns
        -------
        dynObj : :class:`gncpy.dynamics.basic.DynamicsBase`
            Dynamic object created
        pos_inds : list
            indices of the position variables in the state
        vel_inds : list
            indices of the velocity variables in the state
        state_args : tuple
            additional arguments for the dynObj propagate function
        ctrl_args
            additional arguments for the dynObj propagate function
        state_low : numpy array
            lower state bounds
        state_high : numpy array
            upper state bounds
        state0 : numpy array
            initial state of the dynObj
        """
        cls_type = getattr(gdyn, params.type)

        pos_inds, vel_inds = self.get_player_pos_vel_inds(params=params)
        kwargs = {}
        if params.type == "DoubleIntegrator":
            state_args = (self.params.physics.update_dt,)

            cParams = params.controlModel
            if cParams.type.lower() == "velocity":
                ctrl_args = ()

                def _ctrl_mod(t, x, *args):
                    if cParams.max_vel_x and cParams.max_vel_y:
                        mat = np.diag(
                            (float(cParams.max_vel_x), float(cParams.max_vel_y))
                        )
                    elif cParams.max_vel:
                        mat = cParams.max_vel * np.eye(2)
                    else:
                        raise RuntimeError(
                            "Must set max_vel or max_vel_x and max_vel_y in control model."
                        )
                    return np.vstack((np.zeros((2, 2)), mat))

            else:
                msg = "Control model type {} not implemented for dynamics {}".format(
                    cParams.type, params.type
                )
                raise NotImplementedError(msg)
            kwargs["control_model"] = _ctrl_mod

            sParams = params.stateConstraint
            if sParams.type.lower() != "none":
                if sParams.type.lower() == "velocity":

                    def _state_constraint(t, x):
                        x[vel_inds] = np.min(
                            np.vstack((x[vel_inds].ravel(), sParams.max_vels)), axis=0,
                        ).reshape((len(vel_inds), 1))
                        x[vel_inds] = np.max(
                            np.vstack((x[vel_inds].ravel(), sParams.min_vels)), axis=0,
                        ).reshape((len(vel_inds), 1))
                        return x

                else:
                    msg = "State constraint type {} not implemented for dynamics {}".format(
                        sParams.type, params.type
                    )
                    raise NotImplementedError(msg)
                kwargs["state_constraint"] = _state_constraint

        elif params.type == "CoordinatedTurn":
            state_args = ()

            cParams = params.controlModel
            if cParams.type.lower() == "velocity_turn":
                ctrl_args = ()

                def _g1(t, x, u, *args):
                    return cParams.max_vel * np.cos(x[4].item()) * u[0].item()

                def _g0(t, x, u, *args):
                    return 0

                def _g3(t, x, u, *args):
                    return cParams.max_vel * np.sin(x[4].item()) * u[0].item()

                def _g2(t, x, u, *args):
                    return 0

                def _g4(t, x, u, *args):
                    return cParams.max_turn_rate * np.pi / 180 * u[1].item()

            else:
                msg = "Control model type {} not implemented for dynamics {}".format(
                    cParams.type, params.type
                )
                raise NotImplementedError(msg)
            kwargs["control_model"] = [_g0, _g1, _g2, _g3, _g4]

            sParams = params.stateConstraint
            if sParams.type.lower() != "none":
                if sParams.type.lower() == "velocity":

                    def _state_constraint(t, x):
                        x[vel_inds] = np.min(
                            np.vstack((x[vel_inds].ravel(), sParams.max_vels)), axis=0,
                        ).reshape((-1, 1))
                        x[vel_inds] = np.max(
                            np.vstack((x[vel_inds].ravel(), sParams.min_vels)), axis=0,
                        ).reshape((-1, 1))
                        if x[4] < 0:
                            x[4] = np.mod(x[4], -2 * np.pi)
                        else:
                            x[4] = np.mod(x[4], 2 * np.pi)

                        return x

                else:
                    msg = "State constraint type {} not implemented for dynamics {}".format(
                        sParams.type, params.type
                    )
                    raise NotImplementedError(msg)
                kwargs["state_constraint"] = _state_constraint

        kwargs.update(params.extra_params)

        state_low, state_high = self.get_player_state_bounds(params=params)

        dynObj = cls_type(**kwargs)
        state0 = np.zeros((state_low.size, 1))
        val = cBirth.sample()
        if val.size == len(pos_inds):
            state0[pos_inds] = val.reshape(state0[pos_inds].shape)

            if cBirth.randomize and params.type == "CoordinatedTurn":
                state0[4] = self.rng.random() * 2 * np.pi

        elif val.size == state0.size:
            state0 = val.reshape(state0.shape)
        else:
            raise RuntimeError("Birth location must match position size or full state.")

        return (
            dynObj,
            pos_inds,
            vel_inds,
            state_args,
            ctrl_args,
            state_low,
            state_high,
            state0,
        )

    def create_player(self, params):
        """Creates a player entity.

        Parameters
        ----------
        params : :class:`gncpy.games.SimpleUAV2d.PlayerParams`
            Parameters for the player being created.

        Returns
        -------
        p : :class:`gncpy.game_engine.entities.Entity`
            Reference to the player entity that was created.
        """
        # check if using random birth time
        if params.birth.times.size == 0:
            req_spawn = self.rng.uniform(0.0, 1.0) < params.birth.prob
        else:
            diff = self.current_time - np.sort(params.birth.times)
            inds = np.where(diff >= -1e-8)[0]
            if inds.size == 0:
                return None
            min_diff = diff[inds[-1]]
            # birth times don't have to align with updates
            req_spawn = min_diff < self.params.physics.update_dt - 1e-8

        if not req_spawn:
            return None

        e = self.entityManager.add_entity("player")

        e.add_component(
            gcomp.CBirth,
            b_type=params.birth.type,
            loc=params.birth.location,
            scale=params.birth.scale,
            params=params.birth.extra_params,
            rng=self.rng,
            randomize=params.birth.randomize,
        )

        e.add_component(gcomp.CDynamics)
        cDyn = e.get_component(gcomp.CDynamics)
        (
            cDyn.dynObj,
            cDyn.pos_inds,
            cDyn.vel_inds,
            cDyn.state_args,
            cDyn.ctrl_args,
            cDyn.state_low,
            cDyn.state_high,
            cDyn.state,
        ) = self.create_dynamics(params.dynamics, e.get_component(gcomp.CBirth))

        e.add_component(gcomp.CTransform)
        cTrans = e.get_component(gcomp.CTransform)
        p_ii = cDyn.pos_inds
        v_ii = cDyn.vel_inds
        cTrans.pos = gphysics.dist_to_pixels(
            cDyn.state[p_ii], self.dist_per_pix, min_pos=self.params.physics.min_pos
        )
        if v_ii is not None:
            cTrans.vel = gphysics.dist_to_pixels(
                cDyn.state[v_ii], self.dist_per_pix
            )

        e.add_component(gcomp.CEvents)

        e.add_component(
            gcomp.CShape,
            s_type=params.shape.type,
            w=gphysics.dist_to_pixels(params.shape.width, self.dist_per_pix[0]),
            h=gphysics.dist_to_pixels(params.shape.height, self.dist_per_pix[1]),
            color=tuple(params.shape.color),
            zorder=100,
            fpath=params.shape.file,
        )

        e.add_component(
            gcomp.CCollision,
            w=gphysics.dist_to_pixels(params.collision.width, self.dist_per_pix[0]),
            h=gphysics.dist_to_pixels(
                params.collision.height, self.dist_per_pix[1]
            ),
        )

        e.add_component(gcomp.CCapabilities, capabilities=params.capabilities)

        return e

    def spawn_players(self):
        """Spawns a new player if needed."""
        for params in self.params.players:
            self.create_player(params)

    def propagate_dynamics(self, eDyn, action):
        """Propagates the dynamics with the given action.

        Parameters
        ----------
        eDyn : :class:`gncpy.game_engine.components.CDynamics`
            dynamics component to propagate.
        action : numpy array
            Control input for the dynamics object.
        """
        eDyn.state = eDyn.dynObj.propagate_state(
            self.current_time,
            eDyn.last_state.reshape((-1, 1)),
            u=action.reshape((-1, 1)),
            state_args=eDyn.state_args,
            ctrl_args=eDyn.ctrl_args,
        ).reshape((-1, 1))

    def get_player_ids(self):
        """Get the entity ids of all players.

        Returns
        -------
        list
            all entity ids of the players.
        """
        return self.entityManager.get_entity_ids(tag="player")

    def reset(self, **kwargs):
        """Resets the game to the base state.

        Parameters
        ----------
        kwargs : dict
            Additional arguments for the parent classes.
        """
        super().reset(**kwargs)

        # find list of all possible order values, and all capabilities
        self.cur_target_seq = None
        self.target_seq = []
        self.all_capabilities = []
        for t in self.params.targets:
            if t.order not in self.target_seq:
                self.target_seq.append(t.order)

            for c in t.capabilities:
                if c not in self.all_capabilities:
                    self.all_capabilities.append(c)

        self.target_seq.sort()

        # make sure all players either have birth times or birth probability and update all capabilities
        self.has_random_player_birth_times = False
        self.max_player_birth_time = -np.inf
        for ii, p in enumerate(self.params.players):
            if p.birth.times.size == 0 and p.birth.prob <= 0:
                raise RuntimeError("Player {} has invalid birth settings.".format(ii))
            self.has_random_player_birth_times = (
                self.has_random_player_birth_times or p.birth.prob > 0
            )
            if np.max(p.birth.times) > self.max_player_birth_time:
                self.max_player_birth_time = np.max(p.birth.times)

            for c in p.capabilities:
                if c not in self.all_capabilities:
                    self.all_capabilities.append(c)

        self.create_obstacles()
        self.create_hazards()
        self.create_targets()
        self.spawn_players()
        self._performed_reset_spawn = True

        self.entityManager.update()

    def s_collision(self):
        """Check for collisions between entities.

        This also handles player death if a hazard destroys a player, and
        updates the events.

        Returns
        -------
        bool
            Flag indicating if a target was hit.
        """
        hit_target = False

        # update all bounding boxes
        for e in self.entityManager.get_entities():
            if e.has_component(gcomp.CTransform) and e.has_component(gcomp.CCollision):
                c_collision = e.get_component(gcomp.CCollision)
                c_transform = e.get_component(gcomp.CTransform)
                c_collision.aabb.centerx = c_transform.pos[0].item()
                c_collision.aabb.centery = c_transform.pos[1].item()

        # check for collision of player
        for e in self.entityManager.get_entities("player"):
            p_aabb = e.get_component(gcomp.CCollision).aabb
            p_trans = e.get_component(gcomp.CTransform)
            p_events = e.get_component(gcomp.CEvents)

            # check for out of bounds, stop at out of bounds
            out_side, out_top = gphysics.clamp_window_bounds2d(
                p_aabb, p_trans, self.window.get_width(), self.window.get_height()
            )
            if out_side:
                p_events.events.append((EventType.OBSTACLE, None))
            if out_top:
                p_events.events.append((EventType.OBSTACLE, None))

            # check for collision with obstacle
            for w in self.entityManager.get_entities("obstacle"):
                if not w.has_component(gcomp.CCollision):
                    continue
                w_aabb = w.get_component(gcomp.CCollision).aabb
                if gphysics.check_collision2d(p_aabb, w_aabb):
                    gphysics.resolve_collision2d(
                        p_aabb, w_aabb, p_trans, w.get_component(gcomp.CTransform)
                    )
                    p_events.events.append((EventType.OBSTACLE, None))

            # check for collision with other players
            for otherP in self.entityManager.get_entities("player"):
                if otherP.id == e.id:
                    continue
                fixedAAABB = otherP.get_component(gcomp.CCollision).aabb
                if gphysics.check_collision2d(p_aabb, fixedAAABB):
                    gphysics.resolve_collision2d(
                        p_aabb,
                        fixedAAABB,
                        p_trans,
                        otherP.get_component(gcomp.CTransform),
                    )
                    if self.params.score.type.lower() == "basic":
                        p_events.events.append((EventType.OBSTACLE, None))
                    else:
                        p_events.events.append((EventType.COL_PLAYER, None))

            # check for collision with hazard
            for h in self.entityManager.get_entities("hazard"):
                if not h.has_component(gcomp.CCollision):
                    continue
                h_aabb = h.get_component(gcomp.CCollision).aabb
                c_hazard = h.get_component(gcomp.CHazard)
                if gphysics.check_collision2d(p_aabb, h_aabb):
                    if self.rng.uniform(0.0, 1.0) < c_hazard.prob_of_death:
                        e.destroy()
                        p_events.events.append((EventType.DEATH, None))
                        if e.id in c_hazard.entrance_times:
                            del c_hazard.entrance_times[e.id]

                    else:
                        if e.id not in c_hazard.entrance_times:
                            c_hazard.entrance_times[e.id] = self.current_time
                        p_events.events.append(
                            (
                                EventType.HAZARD,
                                {
                                    "prob": c_hazard.prob_of_death,
                                    "t_ent": c_hazard.entrance_times[e.id],
                                },
                            )
                        )
                else:
                    if e.id in c_hazard.entrance_times:
                        del c_hazard.entrance_times[e.id]

            if not e.active:
                continue

            # check for collision with target
            for t in self.entityManager.get_entities("target"):
                if not t.active:
                    continue
                if not t.has_component(gcomp.CCollision):
                    continue

                if gphysics.check_collision2d(
                    p_aabb, t.get_component(gcomp.CCollision).aabb
                ):
                    hit_target = True
                    p_events.events.append((EventType.TARGET, {"target": t}))
                    t.destroy()
                    break

            # update state
            p_dynamics = e.get_component(gcomp.CDynamics)
            p_ii = p_dynamics.pos_inds
            v_ii = p_dynamics.vel_inds

            p_dynamics.state[p_ii] = gphysics.pixels_to_dist(
                p_trans.pos, self.dist_per_pix, min_pos=self.params.physics.min_pos
            )
            if v_ii is not None:
                p_dynamics.state[v_ii] = gphysics.pixels_to_dist(
                    p_trans.vel, self.dist_per_pix
                )

        return hit_target

    def s_input(self, user_input):
        """Validate user input.

        Only allows actions that correspond to a current entity.

        Parameters
        ----------
        user_input : dict
            Each key is an entity id. Each value is a numpy array representing
            the action for that entity.

        Returns
        -------
        out : dict
            Each key is an entity id. Each value is a numpy array for the action.
        """
        ids = self.entityManager.get_entity_ids()
        out = {}
        for key, val in user_input.items():
            if key in ids:
                out[key] = val.reshape((-1, 1))
        return out

    def s_game_over(self):
        """Determines if the game has met the end conditions.

        End conditions are if all players are dead and no random births, or
        all targets have been reached, or it has past the maximum time.
        """
        n_players = len(self.entityManager.get_entities("player"))
        all_players_dead = (
            not self.has_random_player_birth_times
            and n_players == 0
            and self.current_time > self.max_player_birth_time
        )
        self.game_over = (
            self.current_time >= self.params.max_time
            or self.cur_target_seq >= len(self.target_seq)
            or all_players_dead
        )

    def s_movement(self, action):
        """Move entities according to their dynamics.

        Parameters
        ----------
        action : dict
            Each key is an entity id and its value is a 2 x 1 numpy array
            corresponding to control inputs for the given dynamics model. This
            comes from the input system.
        """
        for e in self.entityManager.get_entities():
            if e.has_component(gcomp.CTransform):
                eTrans = e.get_component(gcomp.CTransform)
                eTrans.last_pos[0] = eTrans.pos[0]
                eTrans.last_pos[1] = eTrans.pos[1]

                act_key = e.id
                if e.has_component(gcomp.CDynamics) and act_key in action.keys():
                    eDyn = e.get_component(gcomp.CDynamics)
                    eDyn.last_state = eDyn.state.copy()
                    self.propagate_dynamics(eDyn, action[act_key])

                    p_ii = eDyn.pos_inds
                    v_ii = eDyn.vel_inds
                    eTrans.pos = gphysics.dist_to_pixels(
                        eDyn.state[p_ii],
                        self.dist_per_pix,
                        min_pos=self.params.physics.min_pos,
                    )
                    if v_ii is not None:
                        eTrans.vel = gphysics.dist_to_pixels(
                            eDyn.state[v_ii], self.dist_per_pix
                        )

    def basic_reward(self):
        """Calculate the reward for a timestep for the basic reward type.

        Returns
        -------
        reward : float
            reward for the timestep.
        info : dict
            extra info useful for debugging.
        """

        def _match_function(test_cap, req_cap):
            if len(req_cap) > 0:
                return sum([1 for c in test_cap if c in req_cap]) / len(req_cap)
            else:
                return 1

        t = self.current_time

        reward = 0

        # accumulate rewards from all players
        r_vel = 0
        r_haz_cumul = 0
        r_tar_cumul = 0.0
        r_death_cumul = 0
        r_wall_cumul = 0
        r_vel_cumul = 0
        for player in self.entityManager.get_entities("player"):
            r_hazard = 0
            r_target = 0
            r_death = 0
            r_wall = 0

            p_dynamics = player.get_component(gcomp.CDynamics)
            p_events = player.get_component(gcomp.CEvents)
            p_capabilities = player.get_component(gcomp.CCapabilities)

            if p_dynamics.vel_inds is not None and len(p_dynamics.vel_inds) > 0:
                max_vel = np.linalg.norm(p_dynamics.state_high[p_dynamics.vel_inds])
                min_vel = np.linalg.norm(p_dynamics.state_low[p_dynamics.vel_inds])
                vel = np.linalg.norm(p_dynamics.state[p_dynamics.vel_inds])

                vel_per = vel / np.max((max_vel, min_vel))
                if vel_per < self.params.score.min_vel_per:
                    r_vel += -self.params.score.vel_penalty

            for e_type, info in p_events.events:
                if e_type == EventType.HAZARD:
                    r_hazard += -(
                        self.params.score.hazard_multiplier
                        * (info["prob"] * 100)
                        * (t - info["t_ent"])
                    )

                elif e_type == EventType.DEATH:
                    time_decay = self.params.score.death_scale * np.exp(
                        -self.params.score.death_decay * t
                    )
                    r_death = -(
                        time_decay
                        * _match_function(
                            p_capabilities.capabilities, self.all_capabilities
                        )
                        + self.params.score.death_penalty
                    )
                    r_hazard = 0
                    r_target = 0
                    r_wall = 0
                    r_vel = 0
                    break

                elif e_type == EventType.TARGET:
                    target = info["target"]
                    t_capabilities = target.get_component(gcomp.CCapabilities)
                    t_priority = target.get_component(gcomp.CPriority)
                    match_per = _match_function(
                        p_capabilities.capabilities, t_capabilities.capabilities
                    )
                    r_target = (
                        self.params.score.target_multiplier
                        * t_priority.priority
                        * match_per
                    )

                elif e_type == EventType.OBSTACLE:
                    r_wall += -self.params.score.wall_penalty

            r_haz_cumul += r_hazard
            r_tar_cumul += r_target
            r_death_cumul += r_death
            r_wall_cumul += r_wall
            r_vel_cumul += r_vel
            reward += r_hazard + r_target + r_death + r_wall

        # add fixed terms to reward
        r_missed = 0
        if self.game_over:
            # get all targets later in the sequence
            for target in self.params.targets:
                if target.order <= self.target_seq[self.cur_target_seq]:
                    continue
                r_missed += target.priority

            # get all remaining targets at current point in sequence
            for target in self.entityManager.get_entities("target"):
                if target.active:
                    r_missed += -target.get_component(gcomp.CPriority).priority

            r_missed *= self.params.score.missed_multiplier

        reward += -self.params.score.time_penalty + r_missed + r_vel

        info = {
            "hazard": r_haz_cumul,
            "target": r_tar_cumul,
            "death": r_death_cumul,
            "wall": r_wall_cumul,
            "missed": r_missed,
            "velocity": r_vel_cumul,
        }

        return reward, info

    def s_score(self):
        """Determines the total score for the timestep.

        Raises
        ------
        NotImplementedError
            An incorrect type is specified.

        Returns
        -------
        score : flaot
            total score for this timestep.
        info : dict
            extra info from the score system
        """
        if self.params.score.type.lower() == "basic":
            return self.basic_reward()
        else:
            msg = "Score system has no implementation for reward type {}".format(
                self.params.score.type
            )
            raise NotImplementedError(msg)

    def step(self, user_input):
        """Perform one iteration of the game loop.

        Multiple physics updates can be made between rendering calls. Also
        spwans players if needed and updates the targets based on what has been
        reached.

        Parameters
        ----------
        user_input : dict
            Each key is an integer representing the entity id that the action
            applies to. Each value is a numpy array for the action to take.

        Returns
        -------
        info : dict
            Extra infomation for debugging.
        """
        # reset handles spawn so don't call it on first step after reset
        if not self._performed_reset_spawn:
            self.spawn_players()
        self._performed_reset_spawn = False

        self.create_targets()
        return super().step(user_input)
