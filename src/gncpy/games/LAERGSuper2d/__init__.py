"""Implements a simple 2D game for the LAGER SUPER UAV.

Included configuration files for this game are:

    * LAGERSuper2dExample.yaml

"""
import numpy as np

import gncpy.dynamics.aircraft as gaircraft
from gncpy.games.SimpleUAV2d import SimpleUAV2d, EventType
import gncpy.game_engine.components as gcomp
import gncpy.game_engine.physics2d as gphysics
import gncpy.dynamics.aircraft.lager_super_bindings as super_bind
from gncpy.game_engine.physics2d import Physics2dParams as BasePhysicsParams
from gncpy.coordinate_transforms import lla_to_NED, ned_to_LLA


class DynamicsParams:
    """Parameters for the state constraints to be parsed by the yaml parser.

    The types defined in this class determine what type the parser uses.
    For more details on the dynamics model see
    :class:`gncpy.dynamics.aircraft.SimpleLAGERSuper`.

    Attributes
    ----------
    ref_lat_deg : float
        Reference latitude for transforming to NED positions, in degrees.
    ref_lon_deg : float
        Reference longitude for transforming to NED positions, in degrees.
    terrain_alt_wgs84 : float
        Altitude of the terrain according to the WGS84 model, in meters.
    wp_radius : float
        If within this radius of a waypoint, the waypoint is considered reached.
        Units of meters.
    fly_height : float
        Altitude in meters the UAV is flying at.
    ned_mag_field : numpy array
        reference values for the magnetic field in NED coordinates.
    """

    def __init__(self):
        self.ref_lat_deg = 0.0
        self.ref_lon_deg = 0.0
        self.terrain_alt_wgs84 = 0.0
        self.wp_radius = 0.0
        self.fly_height = 0.0
        self.init_height = 0.0
        self.ned_mag_field = np.array([])
        self.init_eul_angs = np.array([])


class Physics2dParams(BasePhysicsParams):
    """Parameters for the 2d physics system to be parsed by the config parser.

    The types defined in this class determine what type the parser uses. This
    overrides the base class dt and step_factor to make them read only. The
    LAGER SUPER UAV has a dt fixed by the dynamics object so there is no
    configurability for these parameters.
    """

    def __init__(self):
        super().__init__()
        self._dt = -1.0

    # make readonly since it is set by the dynamics object, properties get skipped by the config parser
    @property
    def dt(self):
        """Read only delta time for the physics system."""
        return self._dt

    @dt.setter
    def dt(self, val):
        pass

    # should be called once when creating the dynamics
    def set_dt(self, val):
        """For manually setting the delta time.

        Should only be called once when creating the dynamics since the LAGER
        SUPER UAV requires a fixed dt set by the dynamics model.
        """
        self._dt = val

    @property
    def step_factor(self):
        """Read only step factor."""
        return 1

    @step_factor.setter
    def step_factor(self, val):
        pass


class CFlyHeight:
    """Component for the height the UAV flies at.

    Attributes
    ----------
    fly_height : float
        Altitude the UAV flies at.
    """

    def __init__(self, fly_height=None):
        self.fly_height = fly_height


class SimpleLagerSUPER2d(SimpleUAV2d):
    """Simple 2D game for LAGER SUPER UAV.

    Assumes obstacles and hazards are static, and all players have the same
    state and action spaces.
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
        self.waypoints = {}

    def parse_config_file(self):
        """Parses the config file and saves the parameters."""
        super().parse_config_file()
        self.params.physics.set_dt(gaircraft.SimpleLAGERSuper().dt)

    def create_dynamics(self, params, cBirth):
        """Create the dynamics system for a player.

        Parameters
        ----------
        params : :class:`.DynamicsParams`
            Parameters to use when making the dynamics.
        cBirth : :class:`gncpy.game_engine.components.CBirth`
            Birth component associated with this player.

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
        dynObj = gaircraft.SimpleLAGERSuper()
        body_vel = np.zeros(3)
        body_rot_rate = np.zeros(3)
        init_eul_deg = np.zeros(3)
        for ii, ang in enumerate(params.init_eul_angs):
            if np.isinf(ang):
                if ii == 0:
                    init_eul_deg[ii] = self.rng.uniform(low=-180, high=180)
                elif ii == 1:
                    init_eul_deg[ii] = self.rng.uniform(low=-5, high=5)
                elif ii == 2:
                    init_eul_deg[ii] = self.rng.uniform(low=-10, high=10)

        init_xy = cBirth.sample().reshape(-1)
        init_ned_pos = np.concatenate((init_xy[1::-1], np.array([-params.init_height])))
        dynObj.set_initial_conditions(
            init_ned_pos,
            body_vel,
            init_eul_deg,
            body_rot_rate,
            params.ned_mag_field,
            ref_lat_deg=params.ref_lat_deg,
            ref_lon_deg=params.ref_lon_deg,
            terrain_alt_wgs84=params.terrain_alt_wgs84,
        )
        dynObj.waypoint_radius = params.wp_radius

        pos_inds = dynObj.state_map.ned_pos[1::-1]
        vel_inds = dynObj.state_map.body_vel[1::-1]
        state_args = ()
        ctrl_args = ()
        state0 = dynObj.state.copy().reshape((-1, 1))
        state_low = -np.inf * np.ones(state0.size)
        state_high = np.inf * np.ones(state0.size)

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

    def propagate_dynamics(self, eDyn, action):
        """Propagates the dynamics with the given action.

        Parameters
        ----------
        eDyn : :class:`gncpy.game_engine.components.CDynamics`
            dynamics component to propagate.
        action : None
            Not used.
        """
        eDyn.state = eDyn.dynObj.propagate_state(self.current_time).reshape((-1, 1))

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
        p = super().create_player(params)
        if p is not None:
            p.add_component(CFlyHeight, fly_height=params.dynamics.fly_height)

        return p

    def convert_waypoints(self, input_arr, player):
        """Converts array of x/y waypoints into form for LAGER model.

        Parameters
        ----------
        input_arr : Nx2 numpy array
            Each row is the x, y position in real units.
        player : :class:`gncpy.game_engine.entities.Entity`
            Player entity the waypoints are being created for.

        Returns
        -------
        wp_lst : list
            List of waypoint objects that can be passed to LAGER's upload waypoints.
        """
        pDyn = player.get_component(gcomp.CDynamics)
        pFly = player.get_component(CFlyHeight)

        if not isinstance(input_arr, np.ndarray):
            input_arr = np.array(input_arr)
        if input_arr.shape[1] != 2:
            input_arr = input_arr.T

        r2d = 180 / np.pi
        wp_lst = [super_bind.MissionItem() for ii in range(input_arr.shape[0])]
        for ii, (xy, wp) in enumerate(zip(input_arr, wp_lst)):
            # x is east, y is north
            lla = ned_to_LLA(
                np.hstack((xy[::-1], -pFly.fly_height)),
                pDyn.dynObj.home_lat_rad,
                pDyn.dynObj.home_lon_rad,
                pDyn.dynObj.home_alt_wgs84_m,
            )
            wp.frame = int(3)
            wp.cmd = int(16)
            wp.param1 = float(0)
            wp.param2 = float(0)
            wp.param3 = float(0)
            wp.param4 = float(0)
            wp.x = int(lla[0] * gaircraft.SimpleLAGERSuper.wp_xy_scale * r2d)
            wp.y = int(lla[1] * gaircraft.SimpleLAGERSuper.wp_xy_scale * r2d)
            wp.z = float(pFly.fly_height)
            wp.autocontinue = ii != (len(wp_lst) - 1)

        return wp_lst

    def s_input(self, user_input):
        """Validate user input.

        Passes waypoint lists on to the corresponding player.

        Parameters
        ----------
        user_input : dict
            Each key is an entity id. Each value is a Nx2 numpy array
            representing the waypoints for that entity. Each row is the x,y
            position in real units for the player.

        Returns
        -------
        dict
            empyt dictionary, not needed by any downstream functions.
        """
        if user_input is not None:
            for p in self.entityManager.get_entities("player"):
                if p.id in user_input:
                    p_dyn = p.get_component(gcomp.CDynamics)
                    wps = self.convert_waypoints(user_input[p.id], p)
                    p_dyn.dynObj.upload_waypoints(wps)

                    # if waypoints already exist then destroy them
                    if self.waypoints.get(p.id, None) is not None:
                        for wp in self.waypoints[p.id]:
                            wp.destroy()

                    # clear all waypoints
                    self.waypoints[p.id] = []

                    # add new waypoints
                    for wp in user_input[p.id]:
                        e = self.entityManager.add_entity("waypoint")

                        eTrans = e.add_component(gcomp.CTransform)
                        x = gphysics.dist_to_pixels(
                            wp[0],
                            self.dist_per_pix[0],
                            min_pos=self.params.physics.min_pos[0],
                        )
                        y = gphysics.dist_to_pixels(
                            wp[1],
                            self.dist_per_pix[1],
                            min_pos=self.params.physics.min_pos[1],
                        )
                        eTrans.pos = np.array([x, y]).reshape((2, 1))

                        e.add_component(
                            gcomp.CShape,
                            s_type="sprite",
                            fpath="flag.png",
                            w=gphysics.dist_to_pixels(0.5, self.dist_per_pix[0]),
                            h=gphysics.dist_to_pixels(1, self.dist_per_pix[1]),
                            zorder=100000,
                        )

                        self.waypoints[p.id].append(e)

        # value not needed, only need valid player keys
        return {p.id: "" for p in self.entityManager.get_entities("player")}

    def s_collision(self):
        """Check for collisions between entities.

        This also handles player death if a hazard destroys a player, collision
        with obstacle, or collision with another player. It also updates the
        events.

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

            # check for collision with obstacle
            for w in self.entityManager.get_entities("obstacle"):
                if not w.has_component(gcomp.CCollision):
                    continue
                w_aabb = w.get_component(gcomp.CCollision).aabb
                if gphysics.check_collision2d(p_aabb, w_aabb):
                    e.destroy()
                    p_events.events.append((EventType.OBSTACLE, None))
                    continue

            # check for collision with other players
            for otherP in self.entityManager.get_entities("player"):
                if otherP.id == e.id:
                    continue
                fixedAAABB = otherP.get_component(gcomp.CCollision).aabb
                if gphysics.check_collision2d(p_aabb, fixedAAABB):
                    e.destroy()
                    otherP.destroy()
                    if self.params.score.type.lower() == "basic":
                        p_events.events.append((EventType.OBSTACLE, None))
                    else:
                        p_events.events.append((EventType.COL_PLAYER, None))

            if not e.active:
                continue

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
                        continue

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

    def register_params(self, yaml):
        """Register custom classes for this game with the yaml parser.

        Parameters
        ----------
        yaml : ruamel.yaml YAML object
            yaml parser to use.
        """
        super().register_params(yaml)
        yaml.register_class(Physics2dParams)
        yaml.register_class(DynamicsParams)

    def get_player_pos_vel_inds(self, *args, **kwargs):
        """Overloads base class, not needed by this game.

        Raises
        ------
        NotImplementedError
            Not needed by this game.
        """
        raise NotImplementedError("get_player_pos_vel_inds not used.")

    def get_players_state(self, *args, **kwargs):
        """Overloads base class, not needed by this game.

        Raises
        ------
        NotImplementedError
            Not needed by this game.
        """
        raise NotImplementedError("get_players_state not used.")

    def get_player_pos_vels(self):
        """Gets all player position and velocities.

        Returns
        -------
        posvels : dict
            Each key is an entity id, each value is a numpy array of pos, vel.
        """
        posvels = {}
        for p in self.entityManager.get_entities("player"):
            p_dyn = p.get_component(gcomp.CDynamics)
            posvels[p.id] = np.hstack(
                (
                    p_dyn.dynObj.state[p_dyn.pos_inds].flatten(),
                    p_dyn.dynObj.state[p_dyn.vel_inds].flatten(),
                )
            )
        return posvels

    def get_player_modes(self):
        """Get all player flight modes.

        Returns
        -------
        dict
            Each key is an entity id, each value is an int for the flight mode.

        """
        return {
            p.id: p.get_component(gcomp.CDynamics).dynObj.current_mode
            for p in self.entityManager.get_entities("player")
        }

    def step(self, user_input):
        ret_vals = super().step(user_input)

        # update waypoints based on players
        for p in self.entityManager.get_entities("player"):
            if (
                p.id not in self.waypoints
                or self.waypoints[p.id] is None
                or len(self.waypoints[p.id]) == 0
            ):
                continue

            # if player died or not in waypoint follow mode then remove their waypoints
            pDyn = p.get_component(gcomp.CDynamics)
            if not p.active or pDyn.dynObj.current_mode != 2:
                for wp in self.waypoints[p.id]:
                    wp.destroy()
                del self.waypoints[p.id]
                continue

            # check if they reached the waypoint this step
            if pDyn.dynObj.waypoint_reached:
                self.waypoints[p.id][0].destroy()
                del self.waypoints[p.id][0]
                if len(self.waypoints) == 0:
                    del self.waypoints[p.id]

        return ret_vals
