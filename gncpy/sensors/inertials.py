import numpy as np


class InertialBase:
    def __init__(self, **kwargs):
        self.mis_align_sig = kwargs.get('mis_align_sig', [0, 0, 0])
        self.scale_factor_sig = kwargs.get('scale_factor_sig', [0, 0, 0])

        self.corr_times = kwargs.get('corr_times', [1, 1, 1])
        self.gm_vars = kwargs.get('gm_vars', [0, 0, 0])
        self.sample_time = kwargs.get('sample_time', 1)
        self.init_bias_var = kwargs.get('init_bias_var', [0, 0, 0])

        self._sf = np.array([[]])
        self._last_bias = np.array([[]])
        self._gm_drive_sig = 0

        super().__init__(**kwargs)

    def measure(self, true):
        sf = self._calculate_scale_factor()
        return sf @ true + self._calculate_bias()

    def _calculate_bias(self):
        '''Calulates a first order Gauss-Markov process
        '''

        if self._last_bias.size == 0:
            self._last_bias = (np.array(self.init_bias_var)
                               * np.random.randn(1, 3))
            self._last_bias.reshape((3, 1))
            beta = 1 / np.array(self.corr_times)
            var = np.array(self.gm_vars)
            self._gm_drive_sig = np.sqrt(var * (1 - np.exp(-2
                                                           * self.sample_time
                                                           * beta)))

        beta = (1 / np.array(self.corr_times)).reshape((3, 1))
        w = self._gm_drive_sig * np.random.randn(3, 1)
        bias = (1 - self.sample_time * beta) * self._last_bias + w
        self._last_bias = bias
        return bias

    def _calculate_scale_factor(self):
        if self._sf.size == 0:
            sf = np.array(self.scale_factor_sig) * np.random.randn(1, 3)
            ma = np.zeros((3, 3))
            for row in range(0, 3):
                for col in range(0, 3):
                    if row == col:
                        continue
                    ma[row, col] = self.mis_align_sig[row] * np.random.randn()
            self._sf = np.eye(3) + np.diag(sf) + ma

        return self._sf


class Accelerometer(InertialBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Gyroscope(InertialBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
