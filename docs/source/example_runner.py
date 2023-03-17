import sys
import os


def _run_filters():
    sys.path.insert(0, os.path.abspath("./examples/filters"))
    import kalman_filter_dynamic_object as kf_dyn
    import extended_kalman_filter_dynamic_object as ekf_dyn
    import imm_kalman_filters_dynamic_object as imm_dyn
    import StudentTFilter as StudT

    kf_dyn.run()
    ekf_dyn.run()
    imm_dyn.run()
    StudT.run()

    sys.path.pop(0)


def _run_planning():
    sys.path.insert(0, os.path.abspath("./examples/planning"))
    import a_star
    import rrt_star

    a_star.run()
    rrt_star.run()

    sys.path.pop(0)


def _run_control():
    sys.path.insert(0, os.path.abspath("./examples/control"))
    import elqr
    import lqr

    elqr.run()
    lqr.run()

    sys.path.pop(0)


def run_examples():
    _run_filters()
    _run_planning()
    _run_control()
