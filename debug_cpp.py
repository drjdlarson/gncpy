import numpy as np
import gncpy.dynamics.basic as gdyn
import gncpy.measurements as gmeas
import gncpy.filters as gfilt



if __name__ == "__main__":
    dynObj = gdyn.DoubleIntegrator()

    print(dynObj.state_names[1:5])

    x = np.array([0, 0, 1, 0])
    measMod = gmeas.StateObservation()

    filt = gfilt.KalmanFilter()
    filt.set_state_model(dyn_obj=dynObj)
    filt.set_measurement_model(measObj=measMod)
    filt.cov = np.eye(4)
    filt.proc_noise = 2 * np.eye(4)
    filt.meas_noise = 3 * np.eye(2)

    pred_out = filt.predict(1.2, x, state_mat_args=(0.3, ))

    print("filter prediction:")
    print(pred_out)

    corr_out, fit_prob = filt.correct(1.2, np.array([0.32, 0.1]), pred_out, meas_fun_args=([0, 1],))

    print("filter correction:")
    print(corr_out)
    print("filter meas fit prob:")
    print(fit_prob)



    print("done")