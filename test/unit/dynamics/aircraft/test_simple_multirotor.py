import numpy as np

import gncpy.dynamics.aircraft.simple_multirotor as g_simple_multirotor


VERBOSE = False

def test_simple_multirotor():
    if VERBOSE:
        print('Testing SimpleMultirotor', flush=True)

    uav = g_simple_multirotor.SimpleMultirotor('lager_super.yaml')

    ned_pos = np.array([0, 0, -5])
    body_vel = np.array([1, 0, 0])
    eul_deg = np.array([0, 0, 0])
    body_rot_rate = np.array([0, 0, 0])
    ref_lat_deg = 33.209881
    ref_lon_deg = -87.534273
    terrain_alt_wgs84 = 0
    ned_mag_field = np.array([22.9383, -0.0337, -0.1326])
    uav.set_initial_conditions(ned_pos, body_vel, eul_deg, body_rot_rate,
                               ref_lat_deg, ref_lon_deg, terrain_alt_wgs84,
                               ned_mag_field)

    dt = 1
    state = uav.propagate_state(np.zeros(6), dt)
    x_pos_diff = state[uav.state_map.ned_pos[0]] - 1
    assert np.abs(x_pos_diff) < 0.01, 'NED x position is too far off'


def test_simple_LAGER_super():
    if VERBOSE:
        print('Testing SimpleLAGERSuper', flush=True)

    uav = g_simple_multirotor.SimpleLAGERSuper()

    ned_pos = np.array([0, 0, -5])
    body_vel = np.array([1, 0, 0])
    eul_deg = np.array([0, 0, 0])
    body_rot_rate = np.array([0, 0, 0])
    ref_lat_deg = 33.209881
    ref_lon_deg = -87.534273
    terrain_alt_wgs84 = 0
    ned_mag_field = np.array([22.9383, -0.0337, -0.1326])
    uav.set_initial_conditions(ned_pos, body_vel, eul_deg, body_rot_rate,
                               ref_lat_deg, ref_lon_deg, terrain_alt_wgs84,
                               ned_mag_field)

    dt = 1
    state = uav.propagate_state(np.zeros(6), dt)
    x_pos_diff = state[uav.state_map.ned_pos[0]] - 1
    assert np.abs(x_pos_diff) < 0.01, 'NED x position is too far off'


def test_simple_LAGER_super_custom():
    if VERBOSE:
        print('Testing SimpleLAGERSuper', flush=True)

    uav = g_simple_multirotor.SimpleLAGERSuper('lager_super.yaml')

    ned_pos = np.array([0, 0, -5])
    body_vel = np.array([1, 0, 0])
    eul_deg = np.array([0, 0, 0])
    body_rot_rate = np.array([0, 0, 0])
    ref_lat_deg = 33.209881
    ref_lon_deg = -87.534273
    terrain_alt_wgs84 = 0
    ned_mag_field = np.array([22.9383, -0.0337, -0.1326])
    uav.set_initial_conditions(ned_pos, body_vel, eul_deg, body_rot_rate,
                               ref_lat_deg, ref_lon_deg, terrain_alt_wgs84,
                               ned_mag_field)

    dt = 1
    state = uav.propagate_state(np.zeros(6), dt)
    x_pos_diff = state[uav.state_map.ned_pos[0]] - 1
    assert np.abs(x_pos_diff) < 0.01, 'NED x position is too far off'

    # check another way of specifying input args
    uav = g_simple_multirotor.SimpleLAGERSuper(params_file='lager_super.yaml')

    ned_pos = np.array([0, 0, -5])
    body_vel = np.array([1, 0, 0])
    eul_deg = np.array([0, 0, 0])
    body_rot_rate = np.array([0, 0, 0])
    ref_lat_deg = 33.209881
    ref_lon_deg = -87.534273
    terrain_alt_wgs84 = 0
    ned_mag_field = np.array([22.9383, -0.0337, -0.1326])
    uav.set_initial_conditions(ned_pos, body_vel, eul_deg, body_rot_rate,
                               ref_lat_deg, ref_lon_deg, terrain_alt_wgs84,
                               ned_mag_field)

    dt = 1
    state = uav.propagate_state(np.zeros(6), dt)
    x_pos_diff = state[uav.state_map.ned_pos[0]] - 1
    assert np.abs(x_pos_diff) < 0.01, 'NED x position is too far off'


if __name__ == "__main__":
    VERBOSE = True

    # test_simple_multirotor()
    # test_simple_LAGER_super()
    test_simple_LAGER_super_custom()
