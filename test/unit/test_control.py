import numpy as np
import pytest
import numpy.testing as test

import gncpy.control as ctrl
import gncpy.dynamics.basic as gdyn


@pytest.mark.incremental
class TestLQR:
    def test_constructor(self):
        lqr = ctrl.LQR(time_horizon=3, hard_constraints=True)
        test.assert_equal(lqr.time_horizon, 3)
        test.assert_(lqr.hard_constraints)

    def test_set_cost(self, Q, R):
        lqr = ctrl.LQR()
        lqr.set_cost_model(Q=Q, R=R)
        test.assert_allclose(lqr.Q, Q)
        test.assert_allclose(lqr.R, R)


    def test_set_state(self):
        lqr = ctrl.LQR()
        lqr.set_state_model(np.zeros((2, 1)), dynObj=gdyn.DoubleIntegrator(), dt=0.5)
        test.assert_approx_equal(lqr.dt, 0.5)


@pytest.mark.incremental
class TestELQR:
    def test_constructor(self, Q, R):
        elqr = ctrl.ELQR(max_iters=100)
        test.assert_equal(elqr.max_iters, 100)

    def test_set_cost(self, Q, R):
        lqr = ctrl.ELQR()
        lqr.set_cost_model(Q=Q, R=R, non_quadratic_fun=lambda t, x, u, end_state, is_init, is_fin, *args: 3 * np.sum(x.ravel()**2))
        test.assert_allclose(lqr.Q, Q)
        test.assert_allclose(lqr.R, R)

    def test_set_state_lin(self):
        lqr = ctrl.ELQR()
        lqr.set_state_model(np.zeros((2, 1)), dynObj=gdyn.DoubleIntegrator(), dt=0.5)
        test.assert_approx_equal(lqr.dt, 0.5)
    
    def test_set_state_nonlin(self):
        lqr = ctrl.ELQR()
        lqr.set_state_model(np.zeros((2, 1)), dynObj=gdyn.CurvilinearMotion(), dt=0.7)
        test.assert_approx_equal(lqr.dt, 0.7)
