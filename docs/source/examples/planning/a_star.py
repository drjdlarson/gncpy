def normal_a_star():
    import numpy as np
    from gncpy.planning.a_star import AStar

    # define start and end location (x, y)
    start_pos = np.array([10, 10])
    end_pos = np.array([50, 50])

    # define grid square width/height (x, y)
    grid_res = np.array([2, 2])

    # create some obstacles
    obstacles = np.array([])
    w = 1
    h = 1
    for i in range(-10, 60):
        if obstacles.size == 0:
            obstacles = np.array([[i, -10, w, h]])
            continue
        obstacles = np.concatenate((obstacles, np.array([[i, -10, w, h]])), axis=0)
    for i in range(-10, 60):
        obstacles = np.concatenate((obstacles, np.array([[60, i, w, h]])), axis=0)
    for i in range(-10, 61):
        obstacles = np.concatenate((obstacles, np.array([[i, 60, w, h]])), axis=0)
    for i in range(-10, 61):
        obstacles = np.concatenate((obstacles, np.array([[-10, i, w, h]])), axis=0)
    for i in range(-10, 40):
        obstacles = np.concatenate((obstacles, np.array([[20, i, w, h]])), axis=0)
    for i in range(0, 40):
        obstacles = np.concatenate((obstacles, np.array([[40, 60 - i, w, h]])), axis=0)

    # create some hazards
    hazards = np.array([[44, 24, 8, 15, 2], [26, 36, 9, 4, 1.75]])

    # define the bounds of the search region
    min_pos = np.min(obstacles[:, 0:2], axis=0)
    max_pos = np.max(obstacles[:, 0:2], axis=0)

    # create the class, and setup the map
    aStar = AStar()
    aStar.set_map(min_pos, max_pos, grid_res, obstacles=obstacles, hazards=hazards)

    # call the algorithm
    path, cost, fig, frame_list = aStar.plan(
        start_pos, end_pos, show_animation=True, save_animation=True
    )

    return frame_list


def beam_search():
    import numpy as np
    from gncpy.planning.a_star import AStar

    # define start and end location (x, y)
    start_pos = np.array([10, 10])
    end_pos = np.array([50, 50])

    # define grid square width/height (x, y)
    grid_res = np.array([2, 2])

    # create some obstacles
    obstacles = np.array([])
    w = 1
    h = 1
    for i in range(-10, 60):
        if obstacles.size == 0:
            obstacles = np.array([[i, -10, w, h]])
            continue
        obstacles = np.concatenate((obstacles, np.array([[i, -10, w, h]])), axis=0)
    for i in range(-10, 60):
        obstacles = np.concatenate((obstacles, np.array([[60, i, w, h]])), axis=0)
    for i in range(-10, 61):
        obstacles = np.concatenate((obstacles, np.array([[i, 60, w, h]])), axis=0)
    for i in range(-10, 61):
        obstacles = np.concatenate((obstacles, np.array([[-10, i, w, h]])), axis=0)
    for i in range(-10, 40):
        obstacles = np.concatenate((obstacles, np.array([[20, i, w, h]])), axis=0)
    for i in range(0, 40):
        obstacles = np.concatenate((obstacles, np.array([[40, 60 - i, w, h]])), axis=0)

    # create some hazards
    hazards = np.array([[44, 24, 8, 15, 2], [26, 36, 9, 4, 1.75]])

    # define the bounds of the search region
    min_pos = np.min(obstacles[:, 0:2], axis=0)
    max_pos = np.max(obstacles[:, 0:2], axis=0)

    # create the class, and setup the map
    aStar = AStar()
    aStar.set_map(min_pos, max_pos, grid_res, obstacles=obstacles, hazards=hazards)

    # turn on beam search and set max number of nodes to remember
    aStar.use_beam_search = True
    aStar.beam_search_max_nodes = 30

    # call the algorithm
    path, cost, fig, frame_list = aStar.plan(
        start_pos, end_pos, show_animation=True, save_animation=True
    )

    return frame_list


def weighted_a_star():
    import numpy as np
    from gncpy.planning.a_star import AStar

    # define start and end location (x, y)
    start_pos = np.array([10, 10])
    end_pos = np.array([50, 50])

    # define grid square width/height (x, y)
    grid_res = np.array([2, 2])

    # create some obstacles
    obstacles = np.array([])
    w = 1
    h = 1
    for i in range(-10, 60):
        if obstacles.size == 0:
            obstacles = np.array([[i, -10, w, h]])
            continue
        obstacles = np.concatenate((obstacles, np.array([[i, -10, w, h]])), axis=0)
    for i in range(-10, 60):
        obstacles = np.concatenate((obstacles, np.array([[60, i, w, h]])), axis=0)
    for i in range(-10, 61):
        obstacles = np.concatenate((obstacles, np.array([[i, 60, w, h]])), axis=0)
    for i in range(-10, 61):
        obstacles = np.concatenate((obstacles, np.array([[-10, i, w, h]])), axis=0)
    for i in range(-10, 40):
        obstacles = np.concatenate((obstacles, np.array([[20, i, w, h]])), axis=0)
    for i in range(0, 40):
        obstacles = np.concatenate((obstacles, np.array([[40, 60 - i, w, h]])), axis=0)

    # create some hazards
    hazards = np.array([[44, 24, 8, 15, 2], [26, 36, 9, 4, 1.75]])

    # define the bounds of the search region
    min_pos = np.min(obstacles[:, 0:2], axis=0)
    max_pos = np.max(obstacles[:, 0:2], axis=0)

    # create the class, and setup the map
    aStar = AStar()
    aStar.set_map(min_pos, max_pos, grid_res, obstacles=obstacles, hazards=hazards)

    # set the weight for the heuristic to use, can also override the
    # aStar.calc_weight function if the weight depends on the node.
    aStar.weight = 10

    # call the algorithm
    path, cost, fig, frame_list = aStar.plan(
        start_pos, end_pos, show_animation=True, save_animation=True
    )

    return frame_list


def run():
    import os

    fout = os.path.join(os.path.dirname(__file__), "normal_a_star.gif")
    if not os.path.isfile(fout):
        frame_list = normal_a_star()
        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=10,  # convert s to ms
            loop=0,
        )

    fout = os.path.join(os.path.dirname(__file__), "beam_search_a_star.gif")
    if not os.path.isfile(fout):
        frame_list = beam_search()
        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=10,  # convert s to ms
            loop=0,
        )

    fout = os.path.join(os.path.dirname(__file__), "weighted_a_star.gif")
    if not os.path.isfile(fout):
        frame_list = weighted_a_star()
        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=10,  # convert s to ms
            loop=0,
        )
