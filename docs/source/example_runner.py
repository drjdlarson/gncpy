import sys
import os


def run_filters():
    sys.path.insert(0, os.path.abspath("./examples/filters"))
    import kalman_filter_dynamic_object as kf_dyn
    import extended_kalman_filter_dynamic_object as ekf_dyn

    kf_dyn.run()
    ekf_dyn.run()

    sys.path.pop(0)


def run_a_star():
    sys.path.insert(0, os.path.abspath("./examples/planning"))
    import a_star
    import extended_kalman_filter_dynamic_object as ekf_dyn

    a_star.run()

    sys.path.pop(0)

def run_examples():
    run_filters()
    run_a_star()
