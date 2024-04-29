import numpy as np


class Seeker():
    r"""Implements a model of a seeker sensor that tracks objects
    
    Notes
    -----
    todo: implement sensor models with noise
    """

    def __init__(self, sensor_noise=np.array([0])):
        """Initialize an object.

        Parameters
        ----------
        sensor_noise: float, optional
            defines the sensor noise covariance matrix for the seeker. 
            If the input is 2D (x,y -> range, bearing), R is a 2x2 matrix
            If input is 3D (x,y,z -> range, bearing, azimuth), R is 3x3
        """
        self._R = sensor_noise
        

    @property
    def R(self):
        """Read sensor noise matrix"""
        return self._R

    @R.setter
    def R(self, val):
        self._R = val

    def calc_relative_state(self, x_ego, x_target):
        r'''
        Not a 'sensor model' per-se as this isn't meant to be incorporated into a filter that observes
        x,y,z from a sensor's range, bearing, azimuth. Meant to be used if incorporating a simulation 
        where sensor noise/models/filters are not simulated. 

        Returns the relative state of the target from the ego vehicle
        '''
        return x_target - x_ego
    
