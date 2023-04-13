import numpy as np
import pickle
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
    print(filt.cov)

    filt.proc_noise = 2 * np.eye(4)
    filt.meas_noise = 3 * np.eye(2)

    pred_out = filt.predict(1.2, x, state_mat_args=(0.3, ))
    print(filt.cov)
    filt.cov[0, 1] = 2
    print(filt.cov)

    print("filter prediction:")
    print(pred_out)

    # corr_out, fit_prob = filt.correct(1.2, np.array([0.32, 0.1]), pred_out, meas_fun_args=([0, 1],))

    # print("filter correction:")
    # print(corr_out)
    # print("filter meas fit prob:")
    # print(fit_prob)

    # f_state = filt.save_filter_state()

    # filt2 = gfilt.KalmanFilter()
    # filt2.load_filter_state(f_state)
    # print("Loaded filter covariance difference:")
    # print(filt2.cov - filt.cov)

    # corr_out2, fit_prob2 = filt2.correct(1.2, np.array([0.32, 0.1]), pred_out, meas_fun_args=([0, 1],))
    # print("loaded filter corr_out diff:")
    # print(corr_out2 - corr_out)
    # print("loaded filter prob diff:")
    # print(fit_prob2 - fit_prob)

    print("Original filter: {:s}".format(repr(filt)))
    with open("test.pik", "wb") as fout:
        data = pickle.dumps(filt)
        fout.write(data)

    with open("test.pik", "rb") as fin:
        filtLoad = pickle.load(fin)
    print("Loaded filter: {}".format(repr(filtLoad)))

    print("loaded cov")
    print(filtLoad.cov)
    print("orig cov")
    print(filt.cov)
    filtLoad.cov[0, 2] = 5
    print("modified load cov:")
    print(filtLoad.cov)
    print("original cov:")
    print(filt.cov)

    print("done")